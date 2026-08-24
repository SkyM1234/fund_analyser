import unittest
from unittest.mock import AsyncMock, MagicMock

from sqlalchemy.exc import IntegrityError

from app.db.models import TaskRun
from app.services.task_runs import (
    claim_task_run,
    get_or_create_task_run,
    mark_task_cancelled,
    mark_expired_task_runs_lost,
    mark_task_finished,
    request_task_cancel,
    renew_task_lease,
)


class _Result:
    def __init__(
        self,
        task: TaskRun | None,
        tasks: list[TaskRun] | None = None,
    ) -> None:
        self._task = task
        self._tasks = tasks or []

    def scalar_one_or_none(self) -> TaskRun | None:
        return self._task

    def scalars(self):
        return self

    def all(self) -> list[TaskRun]:
        return self._tasks


class TaskRunServiceTests(unittest.IsolatedAsyncioTestCase):
    async def test_reuses_existing_task_for_same_user_and_key(self) -> None:
        existing = TaskRun(
            run_id="existing-run",
            idempotency_key="request-1",
            user_id=1,
            session_id="session-1",
        )
        db = MagicMock()
        db.execute = AsyncMock()
        db.commit = AsyncMock()
        db.execute.return_value = _Result(existing)

        task, created = await get_or_create_task_run(
            db,
            user_id=1,
            session_id="session-1",
            idempotency_key="request-1",
            request_payload={"message": "hello", "session_id": "session-1", "history": []},
        )

        self.assertIs(task, existing)
        self.assertFalse(created)
        db.add.assert_not_called()
        db.commit.assert_not_awaited()

    async def test_allows_same_key_for_different_users(self) -> None:
        db = MagicMock()
        db.execute = AsyncMock()
        db.commit = AsyncMock()
        db.execute.return_value = _Result(None)

        task, created = await get_or_create_task_run(
            db,
            user_id=2,
            session_id="session-2",
            idempotency_key="request-1",
            request_payload={"message": "hello", "session_id": "session-2", "history": []},
        )

        self.assertTrue(created)
        self.assertEqual(task.user_id, 2)
        self.assertEqual(task.idempotency_key, "request-1")
        db.add.assert_called_once_with(task)
        db.commit.assert_awaited_once()

    async def test_concurrent_insert_conflict_reloads_existing_task(self) -> None:
        existing = TaskRun(
            run_id="winning-run",
            idempotency_key="request-1",
            user_id=1,
            session_id="session-1",
        )
        db = MagicMock()
        db.execute = AsyncMock()
        db.commit = AsyncMock()
        db.rollback = AsyncMock()
        db.execute.side_effect = [_Result(None), _Result(existing)]
        db.commit.side_effect = IntegrityError(None, None, Exception("duplicate key"))

        task, created = await get_or_create_task_run(
            db,
            user_id=1,
            session_id="session-1",
            idempotency_key="request-1",
            request_payload={"message": "hello", "session_id": "session-1", "history": []},
        )

        self.assertIs(task, existing)
        self.assertFalse(created)
        db.rollback.assert_awaited_once()

    async def test_claim_issues_lease_token_and_expiry(self) -> None:
        task = TaskRun(
            run_id="run-1",
            idempotency_key="request-1",
            user_id=1,
            session_id="session-1",
            status="QUEUED",
        )
        db = MagicMock()
        db.execute = AsyncMock(return_value=_Result(task))
        db.commit = AsyncMock()

        claimed = await claim_task_run(
            db,
            run_id="run-1",
            attempt=1,
            timeout_seconds=300,
            lease_seconds=45,
        )

        self.assertIs(claimed, task)
        self.assertEqual(task.status, "RUNNING")
        self.assertEqual(task.attempt, 1)
        self.assertIsNotNone(task.lease_token)
        self.assertIsNotNone(task.heartbeat_at)
        self.assertIsNotNone(task.lease_expires_at)
        self.assertGreater(task.lease_expires_at, task.heartbeat_at)
        db.commit.assert_awaited_once()

    async def test_renew_rejects_stale_fencing_token(self) -> None:
        task = TaskRun(
            run_id="run-1",
            idempotency_key="request-1",
            user_id=1,
            session_id="session-1",
            status="RUNNING",
            lease_token="current-token",
        )
        from datetime import datetime, timedelta

        task.lease_expires_at = datetime.now() + timedelta(seconds=30)
        db = MagicMock()
        db.execute = AsyncMock(return_value=_Result(task))
        db.commit = AsyncMock()
        db.rollback = AsyncMock()

        renewed = await renew_task_lease(
            db,
            run_id="run-1",
            lease_token="stale-token",
            lease_seconds=45,
        )

        self.assertFalse(renewed)
        self.assertEqual(task.lease_token, "current-token")
        db.commit.assert_not_awaited()
        db.rollback.assert_awaited_once()

    async def test_renew_extends_current_owner_lease(self) -> None:
        task = TaskRun(
            run_id="run-1",
            idempotency_key="request-1",
            user_id=1,
            session_id="session-1",
            status="RUNNING",
            lease_token="current-token",
        )
        from datetime import datetime, timedelta

        previous_expiry = datetime.now() + timedelta(seconds=1)
        task.lease_expires_at = previous_expiry
        db = MagicMock()
        db.execute = AsyncMock(return_value=_Result(task))
        db.commit = AsyncMock()

        renewed = await renew_task_lease(
            db,
            run_id="run-1",
            lease_token="current-token",
            lease_seconds=45,
        )

        self.assertTrue(renewed)
        self.assertGreater(task.lease_expires_at, previous_expiry)
        self.assertIsNotNone(task.heartbeat_at)
        db.commit.assert_awaited_once()

    async def test_old_token_cannot_finish_new_owner_task(self) -> None:
        task = TaskRun(
            run_id="run-1",
            idempotency_key="request-1",
            user_id=1,
            session_id="session-1",
            status="RUNNING",
            lease_token="new-owner-token",
        )
        db = MagicMock()
        db.execute = AsyncMock(return_value=_Result(task))
        db.commit = AsyncMock()
        db.rollback = AsyncMock()

        updated = await mark_task_finished(
            db,
            run_id="run-1",
            lease_token="old-owner-token",
            status="SUCCESS",
        )

        self.assertFalse(updated)
        self.assertEqual(task.status, "RUNNING")
        db.commit.assert_not_awaited()
        db.rollback.assert_awaited_once()

    async def test_queued_task_is_cancelled_immediately(self) -> None:
        task = TaskRun(
            run_id="run-1",
            idempotency_key="request-1",
            user_id=1,
            session_id="session-1",
            status="QUEUED",
        )
        db = MagicMock()
        db.execute = AsyncMock(return_value=_Result(task))
        db.commit = AsyncMock()

        cancelled = await request_task_cancel(db, run_id="run-1", user_id=1)

        self.assertIs(cancelled, task)
        self.assertEqual(task.status, "CANCELLED")
        self.assertTrue(task.cancel_requested)
        self.assertEqual(task.error_code, "CANCELLED_BY_USER")
        db.commit.assert_awaited_once()

    async def test_running_task_is_cancelled_by_current_lease_owner(self) -> None:
        from datetime import datetime, timedelta

        task = TaskRun(
            run_id="run-1",
            idempotency_key="request-1",
            user_id=1,
            session_id="session-1",
            status="RUNNING",
            lease_token="current-token",
            cancel_requested=False,
        )
        task.lease_expires_at = datetime.now() + timedelta(seconds=30)
        db = MagicMock()
        db.execute = AsyncMock(side_effect=[
            _Result(task),
            _Result(task),
        ])
        db.commit = AsyncMock()

        requested = await request_task_cancel(db, run_id="run-1", user_id=1)
        self.assertIs(requested, task)
        self.assertEqual(task.status, "RUNNING")
        self.assertTrue(task.cancel_requested)

        cancelled = await mark_task_cancelled(
            db,
            run_id="run-1",
            lease_token="current-token",
        )
        self.assertTrue(cancelled)
        self.assertEqual(task.status, "CANCELLED")
        self.assertIsNone(task.lease_token)

    async def test_stale_lease_cannot_mark_task_cancelled(self) -> None:
        from datetime import datetime, timedelta

        task = TaskRun(
            run_id="run-1",
            idempotency_key="request-1",
            user_id=1,
            status="RUNNING",
            lease_token="current-token",
            cancel_requested=True,
        )
        task.lease_expires_at = datetime.now() + timedelta(seconds=30)
        db = MagicMock()
        db.execute = AsyncMock(return_value=_Result(task))
        db.commit = AsyncMock()
        db.rollback = AsyncMock()

        cancelled = await mark_task_cancelled(
            db,
            run_id="run-1",
            lease_token="stale-token",
        )

        self.assertFalse(cancelled)
        self.assertEqual(task.status, "RUNNING")
        db.commit.assert_not_awaited()
        db.rollback.assert_awaited_once()

    async def test_expired_lease_is_marked_lost_and_becomes_recoverable(self) -> None:
        from datetime import datetime, timedelta

        payload = {"message": "hello", "session_id": "session-1", "history": []}
        task = TaskRun(
            run_id="run-1",
            idempotency_key="request-1",
            user_id=1,
            session_id="session-1",
            request_payload=payload,
            status="RUNNING",
            attempt=1,
            max_attempts=3,
            lease_token="expired-token",
        )
        task.lease_expires_at = datetime.now() - timedelta(seconds=1)
        db = MagicMock()
        db.execute = AsyncMock(return_value=_Result(None, [task]))
        db.commit = AsyncMock()

        candidates = await mark_expired_task_runs_lost(db, batch_size=20)

        self.assertEqual(task.status, "LOST")
        self.assertIsNone(task.lease_expires_at)
        self.assertEqual(task.error_code, "LEASE_EXPIRED")
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].run_id, "run-1")
        self.assertEqual(candidates[0].request_payload, payload)
        db.commit.assert_awaited_once()
