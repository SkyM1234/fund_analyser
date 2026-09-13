import asyncio
import unittest
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from celery.exceptions import Retry
from langchain_core.messages import AIMessage

from app.models.chat import ChatRequest
from app.tasks import chat_tasks as tasks


class AnswerPublicationTests(unittest.IsolatedAsyncioTestCase):
    async def exercise_stream(self, fail=False, extra_events=()):
        graph = MagicMock()
        graph.aupdate_state = AsyncMock()
        graph.aget_state = AsyncMock(return_value=SimpleNamespace(
            next=(), values={"final_answer": "Approved answer", "messages": [AIMessage(content="Approved answer")]},
        ))

        async def events(*args, **kwargs):
            yield {"event": "on_chat_model_stream", "run_id": "draft", "metadata": {"langgraph_node": "synthesizer"}, "data": {"chunk": SimpleNamespace(content="UNAPPROVED")}}
            yield {"event": "on_chain_end", "name": "direct_answer", "metadata": {"langgraph_node": "direct_answer"}, "data": {"output": {"draft_answer": "UNAPPROVED"}}}
            for event in extra_events:
                yield event
                self.assertFalse(any(
                    call.args[1] in {"message_start", "token"}
                    for call in published.call_args_list
                ), "Answer content must remain private while the graph is running")
            if fail:
                raise ConnectionError("temporary dependency failure")

        graph.astream_events = events
        checkpoint = SimpleNamespace(aget_tuple=AsyncMock(return_value=SimpleNamespace(checkpoint={"id": "checkpoint"})))
        lock = SimpleNamespace(acquire=AsyncMock(return_value=True), release=AsyncMock())
        with ExitStack() as stack:
            stack.enter_context(patch("app.agent.multi_agent_controller.build_multi_agent_graph", return_value=graph))
            stack.enter_context(patch("app.services.checkpoint.get_checkpointer", new=AsyncMock(return_value=checkpoint)))
            stack.enter_context(patch("app.db.redis.get_redis_client", return_value=SimpleNamespace(lock=MagicMock(return_value=lock))))
            stack.enter_context(patch("app.services.mcp_client.get_mcp_client", new=AsyncMock(return_value=SimpleNamespace(get_call_stats=AsyncMock(return_value={})))))
            stack.enter_context(patch.object(tasks, "_persist_checkpoint_id", new=AsyncMock(return_value=True)))
            stack.enter_context(patch.object(tasks, "_load_replayed_trace", new=AsyncMock(return_value=[])))
            published = stack.enter_context(patch.object(tasks, "publish_event"))
            if fail:
                with self.assertRaises(ConnectionError):
                    await tasks._run_chat_turn("run", ChatRequest(message="query", session_id="session"), 1, "lease", "checkpoint")
            else:
                await tasks._run_chat_turn("run", ChatRequest(message="query", session_id="session"), 1, "lease", "checkpoint")
        lock.release.assert_awaited_once()
        return [call.args for call in published.call_args_list]

    async def test_drafts_are_not_published_and_only_committed_answer_is_sent(self):
        events = await self.exercise_stream()
        self.assertEqual([event[1] for event in events], ["message_start", "token"])
        self.assertEqual(events[-1][2]["delta"], "Approved answer")

    async def test_temporary_failure_does_not_publish_terminal_error(self):
        self.assertEqual(await self.exercise_stream(fail=True), [])

    @staticmethod
    def node_start(name, retry_count=0):
        return {
            "event": "on_chain_start",
            "name": name,
            "metadata": {"langgraph_node": name},
            "data": {"input": {"compliance_retry_count": retry_count}},
        }

    async def test_progress_precedes_approved_answer_for_both_answer_paths(self):
        for answer_node in ("direct_answer", "synthesizer"):
            with self.subTest(answer_node=answer_node):
                events = await self.exercise_stream(extra_events=[
                    self.node_start(answer_node),
                    self.node_start("compliance"),
                    self.node_start("commit_answer"),
                ])
                self.assertEqual(
                    [(event[1], event[2]) for event in events],
                    [
                        ("answer_progress", {"phase": "synthesizing"}),
                        ("answer_progress", {"phase": "reviewing"}),
                        ("answer_progress", {"phase": "finalizing"}),
                        ("message_start", {}),
                        ("token", {"delta": "Approved answer"}),
                    ],
                )

    async def test_revision_is_reviewed_again_before_publication(self):
        events = await self.exercise_stream(extra_events=[
            self.node_start("compliance"),
            {
                "event": "on_chain_end", "name": "compliance",
                "metadata": {"langgraph_node": "compliance"},
                "data": {"output": {"compliance_passed": False, "compliance_reason": "Rejected"}},
            },
            self.node_start("synthesizer", retry_count=1),
            self.node_start("compliance", retry_count=1),
            self.node_start("commit_answer"),
        ])
        self.assertEqual(
            [event[2]["phase"] for event in events if event[1] == "answer_progress"],
            ["reviewing", "revising", "reviewing", "finalizing"],
        )
        self.assertEqual([event[1] for event in events].count("retry_notice"), 1)

    async def test_progress_does_not_publish_answer_on_failure(self):
        events = await self.exercise_stream(fail=True, extra_events=[
            self.node_start("compliance"),
        ])
        self.assertEqual(events, [("run", "answer_progress", {"phase": "reviewing"})])

    async def test_nested_runnables_do_not_emit_duplicate_progress(self):
        nested = self.node_start("compliance")
        nested["name"] = "RunnableSequence"
        events = await self.exercise_stream(extra_events=[nested])
        self.assertEqual([event[1] for event in events], ["message_start", "token"])

    def test_incomplete_or_unarchived_answer_is_rejected(self):
        for snapshot in (
            SimpleNamespace(next=("compliance",), values={"final_answer": "Draft"}),
            SimpleNamespace(next=(), values={"final_answer": "New", "messages": [AIMessage(content="Old")]}),
        ):
            with self.assertRaises(RuntimeError):
                tasks._committed_answer(snapshot)


class TaskTerminalOrderingTests(unittest.TestCase):
    def test_success_event_follows_persistent_status(self):
        order = []

        async def finish(*args):
            order.append("persist")
            return True

        with patch.object(tasks, "run_coro", side_effect=lambda coro, timeout: asyncio.run(coro)), patch.object(
            tasks, "_claim_task_run", new=AsyncMock(return_value=SimpleNamespace(lease_token="lease", max_attempts=3, checkpoint_id=None, attempt=1)),
        ), patch.object(tasks, "_run_chat_turn_with_lease", new=AsyncMock()), patch.object(
            tasks, "_mark_finished", side_effect=finish,
        ), patch.object(tasks, "publish_event", side_effect=lambda *args: order.append(args[1])):
            tasks.run_chat_turn.run("run", {"message": "query", "session_id": "session"}, 1)
        self.assertEqual(order, ["persist", "done"])

    def test_retry_notice_does_not_end_stream(self):
        with patch.object(tasks, "run_coro", side_effect=lambda coro, timeout: asyncio.run(coro)), patch.object(
            tasks, "_claim_task_run", new=AsyncMock(return_value=SimpleNamespace(lease_token="lease", max_attempts=3, checkpoint_id=None, attempt=1)),
        ), patch.object(tasks, "_run_chat_turn_with_lease", new=AsyncMock(side_effect=ConnectionError("temporary"))), patch.object(
            tasks, "_mark_retrying", new=AsyncMock(return_value=True),
        ), patch.object(tasks, "_is_retryable_failure", return_value=True), patch.object(
            tasks.run_chat_turn, "retry", side_effect=Retry(),
        ), patch.object(tasks, "publish_event") as publish:
            with self.assertRaises(Retry):
                tasks.run_chat_turn.run("run", {"message": "query", "session_id": "session"}, 1)
        self.assertEqual([call.args[1] for call in publish.call_args_list], ["retry_notice"])
