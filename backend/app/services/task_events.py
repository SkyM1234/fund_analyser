"""Redis Pub/Sub events with replay and overlap deduplication."""
import asyncio
import json
import logging
import time
import uuid
from typing import AsyncGenerator

from celery.result import AsyncResult
from sqlalchemy import select

from app.core.celery_app import celery_app
from app.db.models import TaskRun
from app.db.mysql import get_session_factory
from app.db.redis import get_redis_client

logger = logging.getLogger(__name__)
_ALIVE_STATES = {"PENDING", "STARTED", "RETRY"}
# Preserve complete answers; trimming arbitrary frames makes replay incomplete.
_REPLAY_TTL_SECONDS = 600
_DONE_EVENTS = {"done", "error", "cancelled"}
_CANCEL_SIGNAL_TTL_SECONDS = 3600


def _channel(run_id: str) -> str:
    return f"chat:events:{run_id}"


def _replay_key(run_id: str) -> str:
    return f"chat:events:replay:{run_id}"


def _cancel_key(run_id: str) -> str:
    return f"chat:cancel:{run_id}"


async def signal_task_cancel(run_id: str) -> None:
    client = get_redis_client()
    await client.set(_cancel_key(run_id), "1", ex=_CANCEL_SIGNAL_TTL_SECONDS)


async def is_task_cancel_signalled(run_id: str) -> bool:
    return bool(await get_redis_client().exists(_cancel_key(run_id)))


def publish_event(run_id: str, event: str, data: dict) -> None:
    import redis as sync_redis
    from app.core.config import get_settings

    payload = json.dumps(
        {"id": uuid.uuid4().hex, "event": event, "data": data}, ensure_ascii=False,
    )
    client = sync_redis.Redis.from_url(
        get_settings().REDIS_URL, decode_responses=True,
        socket_connect_timeout=5, socket_timeout=5,
    )
    try:
        with client.pipeline() as pipe:
            pipe.rpush(_replay_key(run_id), payload)
            pipe.expire(_replay_key(run_id), _REPLAY_TTL_SECONDS)
            pipe.publish(_channel(run_id), payload)
            pipe.execute()
    finally:
        client.close()


async def _terminal_event(run_id: str, task_id: str) -> dict | None:
    # Recovery replaces Celery ids; the persistent run is authoritative.
    async with get_session_factory()() as db:
        task = (await db.execute(
            select(TaskRun).where(TaskRun.run_id == run_id)
        )).scalar_one_or_none()
        if task is not None:
            if task.status == "SUCCESS":
                return {"event": "done", "data": {"finish_reason": "stop"}}
            if task.status == "CANCELLED":
                return {"event": "cancelled", "data": {"message": "Task cancelled by user"}}
            if task.status in {"FAILED", "TIMED_OUT"} or (
                task.status == "LOST" and task.attempt >= task.max_attempts
            ):
                return {"event": "error", "data": {"message": "Task execution failed or timed out"}}
            return None

    state = await asyncio.to_thread(lambda: AsyncResult(task_id, app=celery_app).state)
    if state not in _ALIVE_STATES:
        return {"event": "error", "data": {"message": "Task is no longer available"}}
    return None


async def subscribe_events(
    run_id: str, task_id: str, poll_timeout: float = 30.0,
) -> AsyncGenerator[dict, None]:
    client = get_redis_client()
    pubsub = client.pubsub()
    offset = 0

    async def read_frames() -> list[dict]:
        nonlocal offset
        rows = await client.lrange(_replay_key(run_id), offset, -1)
        offset += len(rows)
        frames = []
        for raw in rows:
            try:
                frame = json.loads(raw)
                if not isinstance(frame, dict) or not isinstance(frame.get("data"), dict):
                    raise ValueError("Invalid event frame")
            except (ValueError, TypeError):
                logger.warning("Invalid task event: run_id=%s", run_id)
                continue
            frames.append(frame)
        return frames

    try:
        # Pub/Sub only wakes the reader. List offsets also deduplicate legacy
        # frames without ids, without confusing two equal token payloads.
        await pubsub.subscribe(_channel(run_id))
        for frame in await read_frames():
            yield frame
            if frame.get("event") in _DONE_EVENTS:
                return

        last_activity = time.monotonic()
        while True:
            remaining = poll_timeout - (time.monotonic() - last_activity)
            if remaining <= 0:
                terminal = await _terminal_event(run_id, task_id)
                # Drain after the status read: final tokens precede SUCCESS.
                for frame in await read_frames():
                    yield frame
                    if frame.get("event") in _DONE_EVENTS:
                        return
                if terminal is not None:
                    yield terminal
                    return
                last_activity = time.monotonic()
                continue
            message = await pubsub.get_message(
                ignore_subscribe_messages=True, timeout=remaining,
            )
            if message is None:
                continue
            last_activity = time.monotonic()
            for frame in await read_frames():
                yield frame
                if frame.get("event") in _DONE_EVENTS:
                    return
    finally:
        try:
            await pubsub.unsubscribe(_channel(run_id))
        finally:
            await pubsub.aclose()
