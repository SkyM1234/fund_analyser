"""Persistent task submission and idempotency helpers."""
from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import TaskRun


@dataclass(frozen=True)
class TaskRecoveryCandidate:
    run_id: str
    user_id: int
    request_payload: dict[str, Any]
    checkpoint_id: str | None
    attempt: int
    max_attempts: int


async def get_or_create_task_run(
    db: AsyncSession,
    *,
    user_id: int,
    session_id: str,
    idempotency_key: str,
    request_payload: dict[str, Any],
    max_attempts: int = 1,
) -> tuple[TaskRun, bool]:
    """Return (task, created), resolving concurrent unique-key inserts."""
    existing = (
        await db.execute(
            select(TaskRun).where(
                TaskRun.user_id == user_id,
                TaskRun.idempotency_key == idempotency_key,
            )
        )
    ).scalar_one_or_none()
    if existing is not None:
        return existing, False

    task = TaskRun(
        run_id=uuid.uuid4().hex,
        idempotency_key=idempotency_key,
        user_id=user_id,
        session_id=session_id,
        request_payload=request_payload,
        max_attempts=max_attempts,
    )
    db.add(task)
    try:
        await db.commit()
    except IntegrityError:
        await db.rollback()
        existing = (
            await db.execute(
                select(TaskRun).where(
                    TaskRun.user_id == user_id,
                    TaskRun.idempotency_key == idempotency_key,
                )
            )
        ).scalar_one_or_none()
        if existing is None:
            raise
        return existing, False

    return task, True


async def set_celery_task_id(
    db: AsyncSession,
    task: TaskRun,
    celery_task_id: str,
) -> TaskRun:
    """Persist the broker task id after Celery accepts the message."""
    task.celery_task_id = celery_task_id
    await db.commit()
    return task


async def set_task_checkpoint_id(
    db: AsyncSession,
    *,
    run_id: str,
    lease_token: str,
    checkpoint_id: str,
) -> bool:
    """Persist the latest checkpoint only for the current lease owner."""
    task = (
        await db.execute(
            select(TaskRun).where(TaskRun.run_id == run_id).with_for_update()
        )
    ).scalar_one_or_none()
    now = datetime.now()
    if (
        task is None
        or task.status != "RUNNING"
        or task.lease_token != lease_token
        or task.lease_expires_at is None
        or task.lease_expires_at <= now
    ):
        await db.rollback()
        return False

    task.checkpoint_id = checkpoint_id
    await db.commit()
    return True


async def request_task_cancel(
    db: AsyncSession,
    *,
    run_id: str,
    user_id: int,
) -> TaskRun | None:
    """Request cancellation and immediately finish work that has not started."""
    task = (
        await db.execute(
            select(TaskRun)
            .where(TaskRun.run_id == run_id, TaskRun.user_id == user_id)
            .with_for_update()
        )
    ).scalar_one_or_none()
    if task is None:
        await db.rollback()
        return None

    if task.status in {"SUCCESS", "FAILED", "CANCELLED", "TIMED_OUT"}:
        await db.commit()
        return task

    task.cancel_requested = True
    if task.status in {"QUEUED", "LOST"}:
        now = datetime.now()
        task.status = "CANCELLED"
        task.finished_at = now
        task.heartbeat_at = now
        task.lease_token = None
        task.lease_expires_at = None
        task.error_code = "CANCELLED_BY_USER"
        task.error_message = "Task cancellation requested by user"
    await db.commit()
    return task


async def is_task_cancel_requested(
    db: AsyncSession,
    *,
    run_id: str,
    lease_token: str,
) -> bool:
    """Read cancellation only for the worker that currently owns the lease."""
    task = (
        await db.execute(
            select(TaskRun.cancel_requested).where(
                TaskRun.run_id == run_id,
                TaskRun.status == "RUNNING",
                TaskRun.lease_token == lease_token,
            )
        )
    ).scalar_one_or_none()
    return bool(task)


async def mark_task_cancelled(
    db: AsyncSession,
    *,
    run_id: str,
    lease_token: str,
) -> bool:
    """Commit CANCELLED only when this worker still owns the active lease."""
    task = (
        await db.execute(
            select(TaskRun).where(TaskRun.run_id == run_id).with_for_update()
        )
    ).scalar_one_or_none()
    now = datetime.now()
    if (
        task is None
        or task.status != "RUNNING"
        or task.lease_token != lease_token
        or task.lease_expires_at is None
        or task.lease_expires_at <= now
        or not task.cancel_requested
    ):
        await db.rollback()
        return False

    task.status = "CANCELLED"
    task.finished_at = now
    task.heartbeat_at = now
    task.lease_token = None
    task.lease_expires_at = None
    task.error_code = "CANCELLED_BY_USER"
    task.error_message = "Task cancelled by user"
    await db.commit()
    return True


async def mark_submission_failed(
    db: AsyncSession,
    task: TaskRun,
    *,
    error_code: str,
    error_message: str,
) -> None:
    """Keep a durable record when broker submission itself fails."""
    task.status = "FAILED"
    task.error_code = error_code
    task.error_message = error_message[:1024]
    await db.commit()


async def claim_task_run(
    db: AsyncSession,
    *,
    run_id: str,
    attempt: int,
    timeout_seconds: int,
    lease_seconds: int,
) -> TaskRun | None:
    """Claim an available run and issue a fencing token for this worker."""
    task = (
        await db.execute(
            select(TaskRun).where(TaskRun.run_id == run_id).with_for_update()
        )
    ).scalar_one_or_none()
    if task is None:
        return None

    now = datetime.now()
    if task.status in {"SUCCESS", "FAILED", "CANCELLED", "TIMED_OUT"}:
        await db.rollback()
        return None
    if task.cancel_requested:
        task.status = "CANCELLED"
        task.finished_at = now
        task.heartbeat_at = now
        task.lease_token = None
        task.lease_expires_at = None
        task.error_code = "CANCELLED_BY_USER"
        task.error_message = "Task cancelled by user"
        await db.commit()
        return None
    if (
        task.status == "RUNNING"
        and task.lease_expires_at is not None
        and task.lease_expires_at > now
    ):
        await db.rollback()
        return None

    task.status = "RUNNING"
    task.attempt = max(attempt, (task.attempt or 0) + 1)
    task.started_at = task.started_at or now
    task.finished_at = None
    task.heartbeat_at = now
    task.lease_token = uuid.uuid4().hex
    task.lease_expires_at = now + timedelta(seconds=lease_seconds)
    task.deadline_at = now + timedelta(seconds=timeout_seconds)
    task.error_code = None
    task.error_message = None
    await db.commit()
    return task


async def renew_task_lease(
    db: AsyncSession,
    *,
    run_id: str,
    lease_token: str,
    lease_seconds: int,
) -> bool:
    """Renew only the lease currently owned by this worker."""
    task = (
        await db.execute(
            select(TaskRun).where(TaskRun.run_id == run_id).with_for_update()
        )
    ).scalar_one_or_none()
    now = datetime.now()
    if (
        task is None
        or task.status != "RUNNING"
        or task.lease_token != lease_token
        or task.lease_expires_at is None
        or task.lease_expires_at <= now
    ):
        await db.rollback()
        return False

    task.heartbeat_at = now
    task.lease_expires_at = now + timedelta(seconds=lease_seconds)
    await db.commit()
    return True


async def mark_task_retrying(
    db: AsyncSession,
    *,
    run_id: str,
    lease_token: str,
    error_code: str,
    error_message: str,
) -> bool:
    """Return a failed transient attempt to QUEUED before Celery retries it."""
    task = (
        await db.execute(
            select(TaskRun).where(TaskRun.run_id == run_id).with_for_update()
        )
    ).scalar_one_or_none()
    now = datetime.now()
    if (
        task is None
        or task.status != "RUNNING"
        or task.lease_token != lease_token
        or task.lease_expires_at is None
        or task.lease_expires_at <= now
    ):
        await db.rollback()
        return False

    if task.cancel_requested:
        await db.rollback()
        return False

    task.status = "QUEUED"
    task.heartbeat_at = now
    task.lease_token = None
    task.lease_expires_at = None
    task.error_code = error_code
    task.error_message = error_message[:1024]
    await db.commit()
    return True


async def mark_task_finished(
    db: AsyncSession,
    *,
    run_id: str,
    lease_token: str,
    status: str,
    error_code: str | None = None,
    error_message: str | None = None,
) -> bool:
    """Persist a terminal state only when this worker still owns the lease."""
    task = (
        await db.execute(
            select(TaskRun).where(TaskRun.run_id == run_id).with_for_update()
        )
    ).scalar_one_or_none()
    now = datetime.now()
    if (
        task is None
        or task.status != "RUNNING"
        or task.lease_token != lease_token
        or task.lease_expires_at is None
        or task.lease_expires_at <= now
    ):
        await db.rollback()
        return False
    if task.cancel_requested:
        await db.rollback()
        return False

    task.status = status
    task.heartbeat_at = now
    task.finished_at = now
    task.lease_expires_at = None
    task.error_code = error_code
    task.error_message = error_message[:1024] if error_message else None
    await db.commit()
    return True


async def mark_expired_task_runs_lost(
    db: AsyncSession,
    *,
    batch_size: int,
) -> list[TaskRecoveryCandidate]:
    """Mark expired active leases LOST and return runs eligible for redelivery."""
    now = datetime.now()
    tasks = (
        await db.execute(
            select(TaskRun)
            .where(
                TaskRun.status == "RUNNING",
                TaskRun.lease_expires_at.is_not(None),
                TaskRun.lease_expires_at <= now,
            )
            .order_by(TaskRun.lease_expires_at)
            .limit(max(1, batch_size))
            .with_for_update()
        )
    ).scalars().all()

    candidates: list[TaskRecoveryCandidate] = []
    for task in tasks:
        task.heartbeat_at = now
        task.finished_at = now
        task.lease_expires_at = None
        task.lease_token = None
        if task.cancel_requested:
            task.status = "CANCELLED"
            task.error_code = "CANCELLED_BY_USER"
            task.error_message = "Task cancelled by user"
        else:
            task.status = "LOST"
            task.error_code = "LEASE_EXPIRED"
            task.error_message = "Worker heartbeat lease expired"

        if (
            not task.cancel_requested
            and task.session_id is not None
            and isinstance(task.request_payload, dict)
            and task.attempt < task.max_attempts
        ):
            candidates.append(
                TaskRecoveryCandidate(
                    run_id=task.run_id,
                    user_id=task.user_id,
                    request_payload=task.request_payload,
                    checkpoint_id=task.checkpoint_id,
                    attempt=task.attempt,
                    max_attempts=task.max_attempts,
                )
            )

    if tasks:
        await db.commit()
    return candidates


async def set_recovery_celery_task_id(
    db: AsyncSession,
    *,
    run_id: str,
    celery_task_id: str,
) -> bool:
    """Record a redelivery id only while the run is still awaiting recovery."""
    task = (
        await db.execute(
            select(TaskRun).where(TaskRun.run_id == run_id).with_for_update()
        )
    ).scalar_one_or_none()
    if task is None or task.status != "LOST" or task.cancel_requested:
        await db.rollback()
        return False

    task.celery_task_id = celery_task_id
    await db.commit()
    return True


async def list_recoverable_lost_task_runs(
    db: AsyncSession,
    *,
    batch_size: int,
) -> list[TaskRecoveryCandidate]:
    """Return persisted LOST runs, including runs left behind during a restart."""
    tasks = (
        await db.execute(
            select(TaskRun)
            .where(
                TaskRun.status == "LOST",
                TaskRun.cancel_requested.is_(False),
                TaskRun.session_id.is_not(None),
                TaskRun.attempt < TaskRun.max_attempts,
            )
            .order_by(TaskRun.finished_at, TaskRun.id)
            .limit(max(1, batch_size))
        )
    ).scalars().all()

    return [
        TaskRecoveryCandidate(
            run_id=task.run_id,
            user_id=task.user_id,
            request_payload=task.request_payload,
            checkpoint_id=task.checkpoint_id,
            attempt=task.attempt,
            max_attempts=task.max_attempts,
        )
        for task in tasks
        if isinstance(task.request_payload, dict)
    ]
