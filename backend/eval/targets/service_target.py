"""由运行中的 Docker HTTP 服务支撑的回答评测目标。"""
from __future__ import annotations

import asyncio
import json
import re
import uuid
from collections.abc import AsyncIterator

import httpx

FUND_CODE_RE = re.compile(r"(?<!\d)\d{6}(?!\d)")


def _extract_tool_fund_codes(tool_calls: list[dict]) -> list[str]:
    codes: set[str] = set()
    for tool_call in tool_calls:
        if tool_call.get("name") != "rag_search":
            continue
        fund_codes = (tool_call.get("args") or {}).get("filter_fund_code")
        if isinstance(fund_codes, str):
            fund_codes = [fund_codes]
        if not isinstance(fund_codes, list):
            continue
        codes.update(
            code
            for code in fund_codes
            if isinstance(code, str) and FUND_CODE_RE.fullmatch(code)
        )
    return sorted(codes)


async def _iter_sse(response: httpx.Response) -> AsyncIterator[tuple[str, str]]:
    """解析 SSE 事件名和多行 data 字段。"""
    event_name = "message"
    data_lines: list[str] = []

    async for line in response.aiter_lines():
        if line == "":
            if data_lines:
                yield event_name, "\n".join(data_lines)
            event_name = "message"
            data_lines = []
            continue

        if line.startswith(":"):
            continue
        if line.startswith("event:"):
            event_name = line[len("event:"):].strip()
        elif line.startswith("data:"):
            data_lines.append(line[len("data:"):].lstrip())

    if data_lines:
        yield event_name, "\n".join(data_lines)


class AnswerServiceTarget:
    """调用已部署的聊天 SSE API，并将其事件适配为评测器输出。"""

    def __init__(
        self,
        *,
        base_url: str,
        username: str,
        password: str,
        timeout_seconds: float,
        auto_register: bool = True,
        identity_pool_size: int = 1,
    ) -> None:
        self.username = username
        self.password = password
        self.auto_register = auto_register
        self.identity_pool_size = max(identity_pool_size, 1)
        self._started = False
        self._start_lock = asyncio.Lock()
        self._token_queue: asyncio.Queue[str] = asyncio.Queue(
            maxsize=self.identity_pool_size
        )
        self._client = httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            timeout=httpx.Timeout(timeout_seconds, connect=10.0),
        )

    async def start(self) -> None:
        if self._started:
            return

        async with self._start_lock:
            if self._started:
                return

            health = await self._client.get("/api/health")
            health.raise_for_status()

            for index in range(self.identity_pool_size):
                identity = (
                    self.username
                    if self.identity_pool_size == 1
                    else f"{self.username}_{index + 1}"
                )
                token = await self._authenticate(identity)
                self._token_queue.put_nowait(token)
            self._started = True

    async def _authenticate(self, username: str) -> str:
        if self.auto_register:
            register = await self._client.post(
                "/api/auth/register",
                json={
                    "username": username,
                    "email": f"{username}@eval.local",
                    "password": self.password,
                },
            )
            if register.status_code not in (201, 409):
                raise RuntimeError(
                    f"评测用户 {username} 注册失败，"
                    f"HTTP {register.status_code}: {_response_excerpt(register)}"
                )

        login = await self._client.post(
            "/api/auth/login",
            json={"username": username, "password": self.password},
        )
        if login.status_code != 200:
            raise RuntimeError(
                f"评测用户 {username} 登录失败，"
                f"HTTP {login.status_code}: {_response_excerpt(login)}"
            )
        return login.json()["access_token"]

    async def aclose(self) -> None:
        await self._client.aclose()

    async def __call__(self, inputs: dict) -> dict:
        await self.start()
        token = await self._token_queue.get()
        try:
            return await self._run_chat(inputs, token)
        finally:
            self._token_queue.put_nowait(token)

    async def _run_chat(self, inputs: dict, token: str) -> dict:
        query = inputs["query"]
        payload = {
            "message": query,
            "session_id": f"eval-{uuid.uuid4().hex}",
            "history": [],
        }
        headers = {"Authorization": f"Bearer {token}"}

        answer_parts: list[str] = []
        tool_calls: list[dict] = []
        retrieval_contexts_by_task: dict[str, list[dict]] = {}
        retrieval_context_task_order: list[str] = []
        intent: str | None = None
        received_done = False

        async with self._client.stream(
            "POST",
            "/api/chat/stream",
            json=payload,
            headers=headers,
        ) as response:
            if response.status_code != 200:
                body = (await response.aread()).decode(errors="replace")
                raise RuntimeError(
                    f"聊天服务请求失败，HTTP {response.status_code}: {body[:300]}"
                )

            async for event_name, data_text in _iter_sse(response):
                try:
                    data = json.loads(data_text)
                except json.JSONDecodeError as exc:
                    raise RuntimeError(
                        f"无法解析 SSE {event_name} 事件: {data_text[:200]}"
                    ) from exc

                if event_name == "message_start":
                    answer_parts.clear()
                elif event_name == "token":
                    delta = data.get("delta")
                    if delta:
                        answer_parts.append(str(delta))
                elif event_name == "route_result":
                    intent = data.get("intent")
                elif event_name == "tool_call":
                    tool_calls.append(
                        {
                            "name": data.get("name", ""),
                            "args": data.get("args") or {},
                            **(
                                {"agent": data["agent_name"]}
                                if data.get("agent_name")
                                else {}
                            ),
                        }
                    )
                elif event_name == "retrieval_context":
                    task_id = str(data.get("task_id") or "")
                    if not task_id:
                        continue
                    if task_id not in retrieval_contexts_by_task:
                        retrieval_context_task_order.append(task_id)
                    retrieval_contexts_by_task[task_id] = list(data.get("chunks") or [])
                elif event_name == "error":
                    raise RuntimeError(data.get("message") or "聊天服务返回 error 事件")
                elif event_name == "done":
                    if data.get("finish_reason") != "stop":
                        raise RuntimeError(f"聊天服务非正常结束: {data}")
                    received_done = True
                    break

        if not received_done:
            raise RuntimeError("聊天服务连接结束但未收到 done 事件")

        retrieved_chunks = [
            chunk
            for task_id in retrieval_context_task_order
            for chunk in retrieval_contexts_by_task[task_id]
        ]
        retrieved_contexts = [
            {
                "task_id": task_id,
                "chunks": retrieval_contexts_by_task[task_id],
            }
            for task_id in retrieval_context_task_order
        ]
        answer = "".join(answer_parts)
        return {
            "answer": answer,
            "cited_fund_codes": _extract_tool_fund_codes(tool_calls),
            "intent": intent,
            "tool_calls": tool_calls,
            "retrieved_chunks": retrieved_chunks,
            "retrieved_contexts": retrieved_contexts,
            "retrieved_chunk_ids": [chunk["id"] for chunk in retrieved_chunks],
            "retrieved_chunk_scores": [
                {
                    "id": chunk["id"],
                    "score": chunk.get("score"),
                }
                for chunk in retrieved_chunks
            ],
        }


def _response_excerpt(response: httpx.Response) -> str:
    return response.text.replace("\n", " ")[:300]
