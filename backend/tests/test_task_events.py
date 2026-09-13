import json
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from app.services.task_events import _terminal_event, subscribe_events


def frame(event_id, event, data):
    return json.dumps({"id": event_id, "event": event, "data": data})


class TaskEventTests(unittest.IsolatedAsyncioTestCase):
    async def test_subscription_overlap_is_delivered_once_without_dropping_equal_tokens(self):
        overlap = frame("1", "token", {"delta": "a"})
        next_token = frame("2", "token", {"delta": "a"})
        done = frame("3", "done", {})
        pubsub = MagicMock()
        pubsub.subscribe = AsyncMock()
        pubsub.unsubscribe = AsyncMock()
        pubsub.aclose = AsyncMock()
        pubsub.get_message = AsyncMock(side_effect=[
            {"data": overlap}, {"data": next_token}, {"data": done},
        ])
        client = MagicMock()
        client.pubsub.return_value = pubsub

        async def backlog(*args):
            pubsub.subscribe.assert_awaited_once()
            return [overlap] if args[1] == 0 else [next_token, done]

        client.lrange = AsyncMock(side_effect=backlog)
        with patch("app.services.task_events.get_redis_client", return_value=client):
            events = [event async for event in subscribe_events("run", "task")]
        self.assertEqual([event["id"] for event in events], ["1", "2", "3"])
        pubsub.aclose.assert_awaited_once()

    async def test_legacy_equal_tokens_are_not_confused_with_overlap(self):
        token = json.dumps({"event": "token", "data": {"delta": "a"}})
        done = json.dumps({"event": "done", "data": {}})
        pubsub = MagicMock(subscribe=AsyncMock(), unsubscribe=AsyncMock(), aclose=AsyncMock())
        pubsub.get_message = AsyncMock(return_value={"data": token})
        client = MagicMock()
        client.pubsub.return_value = pubsub
        client.lrange = AsyncMock(side_effect=[[token], [token, done]])
        with patch("app.services.task_events.get_redis_client", return_value=client):
            events = [event async for event in subscribe_events("run", "task")]
        self.assertEqual([event["event"] for event in events], ["token", "token", "done"])
        self.assertEqual(client.lrange.await_args_list[1].args[1], 1)

    async def test_pubsub_closes_when_replay_already_has_terminal_event(self):
        pubsub = MagicMock(subscribe=AsyncMock(), unsubscribe=AsyncMock(), aclose=AsyncMock())
        client = MagicMock()
        client.pubsub.return_value = pubsub
        client.lrange = AsyncMock(return_value=[frame("1", "done", {})])
        with patch("app.services.task_events.get_redis_client", return_value=client):
            events = [event async for event in subscribe_events("run", "task")]
        self.assertEqual(events[0]["event"], "done")
        pubsub.aclose.assert_awaited_once()

    async def test_recovery_status_overrides_old_celery_task_state(self):
        task = SimpleNamespace(status="RUNNING", attempt=2, max_attempts=3)
        db = MagicMock()
        db.execute = AsyncMock(return_value=MagicMock())
        db.execute.return_value.scalar_one_or_none.return_value = task
        context = MagicMock()
        context.__aenter__ = AsyncMock(return_value=db)
        context.__aexit__ = AsyncMock()
        factory = MagicMock(return_value=context)
        with patch("app.services.task_events.get_session_factory", return_value=factory), patch(
            "app.services.task_events.AsyncResult",
        ) as result:
            self.assertIsNone(await _terminal_event("run", "old-task"))
            result.assert_not_called()
            task.status = "SUCCESS"
            self.assertEqual((await _terminal_event("run", "old-task"))["event"], "done")
            task.status = "LOST"
            task.attempt = 3
            self.assertEqual((await _terminal_event("run", "old-task"))["event"], "error")

    async def test_terminal_poll_drains_final_tokens_before_done(self):
        pubsub = MagicMock(subscribe=AsyncMock(), unsubscribe=AsyncMock(), aclose=AsyncMock())
        client = MagicMock()
        client.pubsub.return_value = pubsub
        client.lrange = AsyncMock(side_effect=[[], [frame("1", "token", {"delta": "final"})]])
        with patch("app.services.task_events.get_redis_client", return_value=client), patch(
            "app.services.task_events._terminal_event", new=AsyncMock(return_value={"event": "done", "data": {}}),
        ):
            events = [event async for event in subscribe_events("run", "task", poll_timeout=0)]
        self.assertEqual([event["event"] for event in events], ["token", "done"])
