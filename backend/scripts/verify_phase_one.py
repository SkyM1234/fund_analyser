"""Live Docker smoke check using a temporary user and conversation.

Run inside the API container: python -m scripts.verify_phase_one
The chat check makes one short LLM-backed request. Test data is cleaned up.
"""
import asyncio
import json
import uuid

import httpx
from httpx_sse import aconnect_sse
from sqlalchemy import delete, select

from app.db.models import ChatSession, RefreshToken, TaskRun, User
from app.db.mysql import close_engine, get_session_factory
from app.db.redis import close_redis_client, get_redis_client
from app.services.checkpoint import close_checkpointer, get_checkpointer
from app.services.router import route_query


async def main(clarification: bool = False):
    queries = [
        ("你好，请查询159103的持仓", "fund_query"),
        ("不要推荐，只比较159103和159104的费率", "fund_query"),
        ("159103的外汇风险如何", "fund_query"),
        ("推荐一只基金", "sensitive"),
        ("今天天气如何", "out_of_scope"),
    ]
    if not clarification:
        routes = await asyncio.gather(*(route_query(query) for query, _ in queries))
        assert [route.intent for route in routes] == [expected for _, expected in queries], "Live intent classification mismatch"
        print("Live routing: five regression queries classified correctly", flush=True)
    username = "phase_one_" + uuid.uuid4().hex
    session_id = uuid.uuid4().hex
    password = uuid.uuid4().hex
    headers = {}
    async with httpx.AsyncClient(base_url="http://127.0.0.1:8800", timeout=330) as client:
        try:
            response = await client.post("/api/auth/register", json={"username": username, "password": password})
            response.raise_for_status()
            response = await client.post("/api/auth/login", json={"username": username, "password": password})
            response.raise_for_status()
            tokens = response.json()
            headers = {"Authorization": "Bearer " + tokens["access_token"]}
            responses = await asyncio.gather(*(
                client.post("/api/auth/refresh", json={"refresh_token": tokens["refresh_token"]}) for _ in range(2)
            ))
            assert sorted(response.status_code for response in responses) == [200, 401], "Refresh must have exactly one winner"
            print("Concurrent refresh: one success, one rejection", flush=True)

            events = []
            async with aconnect_sse(
                client, "POST", "/api/chat/stream", headers=headers,
                json={"session_id": session_id, "message": "它的费率呢" if clarification else "你好！"},
            ) as source:
                async for event in source.aiter_sse():
                    payload = json.loads(event.data)
                    events.append((event.event, payload))
                    if event.event in {"done", "error", "cancelled"}:
                        break
            assert events and events[-1][0] == "done", "Chat did not complete successfully"
            answer = "".join(data["delta"] for name, data in events if name == "token")
            assert answer.strip(), "Chat returned no answer"
            if clarification:
                route = next(data for name, data in events if name == "route_result")
                assert route["intent"] == "fund_query" and route["needs_clarification"] is True
                assert route["scope_basis"] == [] and route["resolved_query"]
                assert "基金代码" in answer, "Missing clarification question"
                assert not any(name in {"plan", "agent_start"} for name, _ in events), "Clarification must not start agents"
                print("Clarification: structured route event, follow-up question, no agents started", flush=True)
            async with get_session_factory()() as db:
                task = (await db.execute(select(TaskRun).where(TaskRun.session_id == session_id))).scalar_one()
                assert task.status == "SUCCESS", "done arrived before persistent SUCCESS"
                run_id = task.run_id
            replay = []
            async with aconnect_sse(client, "GET", f"/api/chat/tasks/{run_id}/stream", headers=headers) as source:
                async for event in source.aiter_sse():
                    payload = json.loads(event.data)
                    replay.append((event.event, payload))
                    if event.event in {"done", "error", "cancelled"}:
                        break
            assert replay == events, "Reconnect replay differs from the original stream"
            print("Chat: committed answer, persistent SUCCESS, identical replay", flush=True)
        finally:
            # Cancel only this script's tasks before deleting its isolated test data.
            async with get_session_factory()() as db:
                user = (await db.execute(select(User).where(User.username == username))).scalar_one_or_none()
                if user is not None:
                    user_id = user.id
                    runs = (await db.execute(select(TaskRun).where(TaskRun.user_id == user_id))).scalars().all()
                    run_ids = [task.run_id for task in runs]
                    for task in runs:
                        if task.status in {"QUEUED", "RUNNING", "LOST"}:
                            await client.post(f"/api/chat/tasks/{task.run_id}/cancel", headers=headers)
                    for _ in range(30):
                        await db.rollback()
                        active = (await db.execute(select(TaskRun.run_id).where(
                            TaskRun.user_id == user_id, TaskRun.status.in_(["QUEUED", "RUNNING", "LOST"]),
                        ))).scalars().all()
                        if not active:
                            break
                        await asyncio.sleep(1)
                    if active:
                        raise RuntimeError("Smoke task has not stopped; test data retained for diagnosis")
                    token_hashes = (await db.execute(select(RefreshToken.token_hash).where(RefreshToken.user_id == user_id))).scalars().all()
                    redis = get_redis_client()
                    for token_hash in token_hashes:
                        await redis.delete("refresh:" + token_hash)
                    for run_id in run_ids:
                        await redis.delete(f"chat:events:replay:{run_id}", f"chat:cancel:{run_id}")
                    await db.execute(delete(TaskRun).where(TaskRun.user_id == user_id))
                    await db.execute(delete(ChatSession).where(ChatSession.user_id == user_id))
                    await db.execute(delete(RefreshToken).where(RefreshToken.user_id == user_id))
                    await db.execute(delete(User).where(User.id == user_id))
                    await db.commit()
                    checkpointer = await get_checkpointer()
                    await checkpointer.adelete_thread(session_id)
                    print("Temporary smoke-test user, session and task data removed", flush=True)
            await close_checkpointer()
            await close_engine()
            await close_redis_client()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--clarification", action="store_true")
    asyncio.run(main(clarification=parser.parse_args().clarification))
