import unittest
from unittest.mock import AsyncMock, patch

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from app.agent.reflection_agent import _build_retry_context, _format_retry_transcript


class RetryContextTests(unittest.IsolatedAsyncioTestCase):
    def test_transcript_keeps_tool_result_and_call_arguments(self) -> None:
        transcript = _format_retry_transcript([
            HumanMessage(content="查询基金近一年收益"),
            AIMessage(
                content="我将查询基金代码。",
                tool_calls=[{
                    "name": "rag_identify_funds",
                    "args": {"query": "基金近一年收益"},
                    "id": "call-1",
                }],
            ),
            ToolMessage(
                tool_call_id="call-1",
                name="rag_identify_funds",
                content="已确认基金代码：005827",
            ),
        ])

        self.assertIn("[human]", transcript)
        self.assertIn("rag_identify_funds", transcript)
        self.assertIn("005827", transcript)

    async def test_retry_context_uses_prior_trace_without_rewriting_query(self) -> None:
        task = {
            "task_id": "task-1",
            "description": "查询基金近一年收益率",
            "query": "基金 005827 近一年收益率",
        }
        messages = [
            HumanMessage(content="查询基金 005827 近一年收益率"),
            ToolMessage(
                tool_call_id="call-1",
                name="rag_search",
                content="近一年收益率：12.3%",
            ),
        ]
        response = type(
            "Response",
            (),
            {
                "content": "已确认基金代码 005827；无需再次查询近一年收益率。",
                "usage_metadata": None,
                "response_metadata": {},
            },
        )()

        with (
            patch("app.agent.reflection_agent.ChatOpenAI"),
            patch(
                "app.agent.reflection_agent.llm_ainvoke",
                new=AsyncMock(return_value=response),
            ) as invoke,
        ):
            context, _ = await _build_retry_context(
                task,
                messages,
                "达到最大迭代次数",
            )

        prompt = invoke.await_args.args[1][0]["content"]
        self.assertIn("基金 005827 近一年收益率", prompt)
        self.assertIn("近一年收益率：12.3%", prompt)
        self.assertIn("不可改写或扩展用户意图", prompt)
        self.assertIn("005827", context)
