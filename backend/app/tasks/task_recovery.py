"""Periodic and startup recovery for task runs whose worker lease expired."""
from __future__ import annotations

import logging
import uuid

from app.core.celery_app import celery_app
from app.core.config import get_settings
from app.core.worker_lifecycle import run_coro
from app.db.mysql import get_session_factory
from app.services.checkpoint import get_checkpointer
from app.services.task_events import publish_event
from app.services.task_runs import (
    TaskRecoveryCandidate,
    list_recoverable_lost_task_runs,
    mark_expired_task_runs_lost,
    set_recovery_celery_task_id,
)
from app.tasks.chat_tasks import run_chat_turn

logger = logging.getLogger(__name__)


async def _mark_expired_runs(batch_size: int) -> list[TaskRecoveryCandidate]:
    async with get_session_factory()() as db:
        return await mark_expired_task_runs_lost(db, batch_size=batch_size)


async def _list_lost_runs(batch_size: int) -> list[TaskRecoveryCandidate]:
    async with get_session_factory()() as db:
        return await list_recoverable_lost_task_runs(db, batch_size=batch_size)


async def _record_recovery_task_id(run_id: str, celery_task_id: str) -> bool:
    async with get_session_factory()() as db:
        return await set_recovery_celery_task_id(
            db,
            run_id=run_id,
            celery_task_id=celery_task_id,
        )


def _load_checkpoint_trace(candidate: TaskRecoveryCandidate) -> list[dict]:
    session_id = candidate.request_payload.get("session_id")
    if not candidate.checkpoint_id or not session_id:
        return []

    async def _load() -> list[dict]:
        checkpointer = await get_checkpointer()
        config = {
            "configurable": {
                "thread_id": candidate.request_payload.get("session_id"),
                "checkpoint_id": candidate.checkpoint_id,
            }
        }
        checkpoint_tuple = await checkpointer.aget_tuple(config)
        if not checkpoint_tuple or not checkpoint_tuple.checkpoint:
            return []
        trace_events = checkpoint_tuple.checkpoint.get("channel_values", {}).get(
            "trace_events",
            {},
        )
        if not isinstance(trace_events, dict):
            return []
        events = trace_events.get(candidate.run_id, [])
        return [event for event in events if isinstance(event, dict)]

    try:
        return run_coro(_load(), timeout=30)
    except Exception:
        logger.exception(
            "[task_recovery] failed to load checkpoint trace; "
            "continuing with an empty recovery snapshot: run_id=%s checkpoint_id=%s",
            candidate.run_id,
            candidate.checkpoint_id,
        )
        return []


@celery_app.task(name="app.tasks.task_recovery.recover_expired_task_runs")
def recover_expired_task_runs() -> int:
    """Mark expired leases LOST and redeliver persisted requests at most once per scan."""
    settings = get_settings()
    batch_size = max(1, settings.TASK_RECOVERY_BATCH_SIZE)
    expired = run_coro(_mark_expired_runs(batch_size), timeout=30)
    lost = run_coro(_list_lost_runs(batch_size), timeout=30)
    candidates = {candidate.run_id: candidate for candidate in [*expired, *lost]}

    recovered = 0
    for candidate in candidates.values():
        celery_task_id = uuid.uuid4().hex
        recorded = run_coro(
            _record_recovery_task_id(candidate.run_id, celery_task_id),
            timeout=30,
        )
        if not recorded:
            continue

        try:
            checkpoint_trace = _load_checkpoint_trace(candidate)
            publish_event(
                candidate.run_id,
                "attempt_start",
                {
                    "attempt": candidate.attempt + 1,
                    "worker_recovery": True,
                    "checkpoint_id": candidate.checkpoint_id,
                    "checkpoint_trace": checkpoint_trace,
                },
            )
            if not candidate.checkpoint_id:
                logger.warning(
                    "[task_recovery] checkpoint_id missing; falling back to legacy "
                    "replay from request input: run_id=%s",
                    candidate.run_id,
                )
            run_chat_turn.apply_async(
                args=(
                    candidate.run_id,
                    candidate.request_payload,
                    candidate.user_id,
                ),
                task_id=celery_task_id,
                queue="agent_queue",
            )
        except Exception:
            logger.exception(
                "[task_recovery] failed to redeliver lost task: run_id=%s",
                candidate.run_id,
            )
            continue

        recovered += 1
        logger.warning(
            "[task_recovery] redelivered lost task: run_id=%s session_id=%s "
            "checkpoint_id=%s celery_task_id=%s attempt=%s/%s",
            candidate.run_id,
            candidate.request_payload.get("session_id"),
            candidate.checkpoint_id,
            celery_task_id,
            candidate.attempt + 1,
            candidate.max_attempts,
        )

    return recovered
