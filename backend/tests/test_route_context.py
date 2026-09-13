import asyncio
import json
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

from app.agent.fund_scope_agent import fund_scope_node
from app.agent.multi_agent_controller import (
    after_fund_scope, build_multi_agent_graph, route_after_intent,
)
from app.agent.plan_validation import PlanValidationError, validate_supervisor_plan
from app.agent.supervisor import _explicit_fund_codes, _generate_new_plan
from app.services.checkpoint import _serializer
from app.services.router import RouteClassificationError, RouteResult, _parse_route, route_query
from app.tasks.chat_tasks import _is_retryable_failure
from app.tools.conversation_utils import request_history_for_prompt


def route_data(query="159103的费率", **changes):
    return {
        "intent": "fund_query",
        "scope_basis": [{"kind": "fund_code", "value": "159103", "source": "current"}],
        "resolved_query": query,
        "needs_clarification": False,
        **changes,
    }


class RouteSchemaTests(unittest.TestCase):
    def test_scope_evidence_survives_checkpoint_serialization(self):
        route = RouteResult.model_validate(route_data())
        restored = _serializer.loads_typed(_serializer.dumps_typed(route))
        self.assertEqual(restored, route)
        self.assertEqual(restored.scope_basis[0].kind, "fund_code")
        self.assertEqual(RouteResult.model_validate({"intent": "fund_query"}).scope_basis, [])

    def test_rejects_invented_or_misassigned_scope(self):
        for changes in [
            {"scope_basis": [{"kind": "fund_code", "value": "159999", "source": "current"}]},
            {"scope_basis": [{"kind": "fund_code", "value": "159103", "source": "history"}]},
            {"intent": "fund_screening"},
            {"scope_basis": []},
            {"needs_clarification": "false"},
            {"resolved_query": "159103与159999的费率"},
        ]:
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                _parse_route(json.dumps(route_data(**changes)), "159103的费率", "")

    def test_rejects_incomplete_schema(self):
        with self.assertRaises(ValueError):
            _parse_route('{"intent":"general_finance"}', "问题", "")

    def test_history_evidence_must_be_explicitly_selected(self):
        data = route_data(scope_basis=[{"kind": "fund_code", "value": "159103", "source": "history"}])
        route = _parse_route(json.dumps(data), "它的费率呢", "159103的持仓")
        self.assertEqual(route.resolved_query, "159103的费率")
        data["resolved_query"] = "159103与159104的费率"
        with self.assertRaises(ValueError):
            _parse_route(json.dumps(data), "它的费率呢", "159103和159104")

    def test_sector_with_conditions_uses_scope_branch(self):
        query = "规模超过10亿且费率低于0.5%的新能源基金"
        route = _parse_route(json.dumps(route_data(
            query, scope_basis=[{"kind": "sector", "value": "新能源", "source": "current"}],
        )), query, "")
        self.assertEqual(route_after_intent({"route_result": route}), "fund_scope")

    def test_condition_only_uses_screening_branch(self):
        for query in ["哪些基金规模超过10亿", "哪些基金持有宁德时代"]:
            route = _parse_route(json.dumps(route_data(
                query, intent="fund_screening", scope_basis=[],
            )), query, "")
            self.assertEqual(route_after_intent({"route_result": route}), "supervisor")

    def test_unresolved_reference_requests_clarification(self):
        query = "它的费率呢"
        route = _parse_route(json.dumps(route_data(
            query, scope_basis=[], needs_clarification=True,
        )), query, "")
        self.assertEqual(route_after_intent({"route_result": route}), "clarification")

    def test_missing_route_never_enters_agent(self):
        with self.assertRaises(RouteClassificationError):
            route_after_intent({})


class RequestHistoryTests(unittest.TestCase):
    def test_history_with_or_without_current_message_is_identical(self):
        history = [HumanMessage(content="159103的持仓"), AIMessage(content="持仓摘要")]
        query = "它的费率呢"
        self.assertEqual(
            request_history_for_prompt(query, history),
            request_history_for_prompt(query, history + [HumanMessage(content=query)]),
        )
        self.assertIn("159103", request_history_for_prompt(query, history))

    def test_history_window_has_three_complete_rounds(self):
        messages = []
        for i in range(4):
            messages.extend([HumanMessage(content=f"question{i}"), AIMessage(content=f"answer{i}")])
        text = request_history_for_prompt("current", messages)
        self.assertNotIn("question0", text)
        self.assertNotIn("answer0", text)
        for i in range(1, 4):
            self.assertIn(f"question{i}", text)
            self.assertIn(f"answer{i}", text)

    def test_historical_codes_are_not_automatically_allowed(self):
        messages = [HumanMessage(content="159103的持仓"), HumanMessage(content="159104的规模")]
        self.assertEqual(_explicit_fund_codes(messages), {"159104"})


class ClassificationFailureTests(unittest.IsolatedAsyncioTestCase):
    async def test_invalid_json_retries_and_recovers(self):
        with patch("app.services.router.ChatOpenAI"), patch(
            "app.services.router.llm_ainvoke", new=AsyncMock(side_effect=[
                AIMessage(content="invalid"), AIMessage(content=json.dumps(route_data())),
            ]),
        ) as invoke:
            result = await route_query("159103的费率")
        self.assertEqual(result.intent, "fund_query")
        self.assertEqual(invoke.await_count, 2)

    async def test_failure_never_defaults_to_general_finance(self):
        for result in [AIMessage(content="invalid"), ConnectionError("offline")]:
            with self.subTest(result=result), patch("app.services.router.ChatOpenAI"), patch(
                "app.services.router.llm_ainvoke", new=AsyncMock(side_effect=[result, result]),
            ) as invoke, self.assertRaises(RouteClassificationError) as failure:
                await route_query("159103的费率")
            self.assertEqual(invoke.await_count, 2)
            self.assertTrue(_is_retryable_failure(failure.exception))

    async def test_timeout_is_bounded(self):
        async def slow(*args, **kwargs):
            await asyncio.sleep(10)
        with patch("app.services.router.ChatOpenAI"), patch(
            "app.services.router.ROUTE_TIMEOUT_SECONDS", 0.01,
        ), patch("app.services.router.llm_ainvoke", side_effect=slow), self.assertRaises(RouteClassificationError):
            await route_query("159103的费率")

    async def test_cancellation_is_not_retried(self):
        with patch("app.services.router.ChatOpenAI"), patch(
            "app.services.router.llm_ainvoke", new=AsyncMock(side_effect=asyncio.CancelledError),
        ) as invoke, self.assertRaises(asyncio.CancelledError):
            await route_query("159103的费率")
        self.assertEqual(invoke.await_count, 1)


class ScopeClarificationTests(unittest.IsolatedAsyncioTestCase):
    async def test_mcp_error_text_is_not_treated_as_missing_fund(self):
        tool = MagicMock()
        tool.name = "rag_identify_funds"
        response = AIMessage(content="", tool_calls=[{
            "name": tool.name, "args": {"query": "基金名称"}, "id": "scope1",
        }])
        for error in ["工具调用失败: timeout", "工具执行失败: offline"]:
            tool.ainvoke = AsyncMock(return_value=error)
            with patch("app.agent.fund_scope_agent._get_scope_tools", return_value=[tool]), patch(
                "app.agent.fund_scope_agent.create_chat_llm",
            ), patch("app.agent.fund_scope_agent.llm_ainvoke", new=AsyncMock(return_value=response)) as invoke:
                result = await fund_scope_node({"messages": [HumanMessage(content="基金名称的规模")]})
            self.assertIsNone(result["fund_scope"])
            self.assertIn(error, result["fund_scope_error"])
            self.assertEqual(after_fund_scope(result), "planning_failure")
            invoke.assert_awaited_once()

    def test_scope_ambiguity_and_service_failure_take_different_branches(self):
        route = RouteResult(intent="fund_query")
        for scope in [
            {"funds": []},
            {"funds": [{"fund_code": "159103"}], "needs_clarification": True},
            {"funds": [{"fund_code": "159103"}], "missing_or_uncertain": ["另一只基金"]},
        ]:
            self.assertEqual(after_fund_scope({"route_result": route, "fund_scope": scope}), "clarification")
        self.assertEqual(after_fund_scope({
            "route_result": route, "fund_scope": None, "fund_scope_error": "工具不可用",
        }), "planning_failure")

    def test_incomplete_sector_candidates_can_continue(self):
        route = RouteResult.model_validate(route_data(scope_basis=[
            {"kind": "sector", "value": "新能源", "source": "current"},
        ]))
        self.assertEqual(after_fund_scope({
            "route_result": route,
            "fund_scope": {"funds": [{"fund_code": "159103"}], "coverage_status": "incomplete", "missing_or_uncertain": ["覆盖不全"]},
        }), "supervisor")

    async def test_scope_uses_resolved_query_and_basis(self):
        route = RouteResult.model_validate(route_data(scope_basis=[
            {"kind": "fund_code", "value": "159103", "source": "history"},
        ]))
        history = [HumanMessage(content="159103的持仓"), AIMessage(content="持仓摘要"), HumanMessage(content="它的费率呢")]
        response = AIMessage(content='{"funds":[{"fund_code":"159103"}],"coverage_status":"confirmed"}')
        with patch("app.agent.fund_scope_agent._get_scope_tools", return_value=[MagicMock(name="tool")]), patch(
            "app.agent.fund_scope_agent.create_chat_llm",
        ), patch("app.agent.fund_scope_agent.llm_ainvoke", new=AsyncMock(return_value=response)) as invoke:
            result = await fund_scope_node({"messages": history, "route_result": route})
        self.assertIsNone(result["fund_scope_error"])
        prompt = invoke.call_args.args[1]
        self.assertEqual(prompt[1].content, "159103的费率")
        self.assertIn("ROUTE_SCOPE_BASIS", prompt[0].content)


class GraphContextTests(unittest.IsolatedAsyncioTestCase):
    async def test_new_direct_answer_clears_old_execution_before_archiving(self):
        with patch("app.agent.multi_agent_controller.compliance_agent_node", new=AsyncMock(return_value={"compliance_passed": True})):
            graph = build_multi_agent_graph(InMemorySaver(serde=_serializer))
        config = {"configurable": {"thread_id": "new-turn"}}
        await graph.aupdate_state(config, {
            "messages": [HumanMessage(content="159103的持仓"), AIMessage(content="旧回答")],
            "plan": [{"task_id": "old", "status": "completed"}],
            "sub_results": {"old": "旧数据"},
            "fund_scope": {"funds": [{"fund_code": "159103"}]},
        }, as_node="commit_answer")
        result = await graph.ainvoke({"messages": [HumanMessage(content="你好")]}, config)
        self.assertEqual(result["plan"], [])
        self.assertEqual(result["sub_results"], {})
        self.assertIsNone(result["fund_scope"])
        self.assertEqual(result["plan_history"][-1]["plan"], [])
        self.assertEqual(result["plan_history"][-1]["results"], {})

    async def test_screening_does_not_inherit_previous_scope_or_plan(self):
        seen = []
        async def supervisor(state):
            seen.append(state)
            return {"planning_error": "test stop"}
        with patch("app.agent.multi_agent_controller.supervisor_node", supervisor):
            graph = build_multi_agent_graph(InMemorySaver(serde=_serializer))
        config = {"configurable": {"thread_id": "screening"}}
        await graph.aupdate_state(config, {
            "messages": [HumanMessage(content="159103的持仓"), AIMessage(content="旧回答")],
            "plan": [{"task_id": "old", "status": "running"}],
            "fund_scope": {"funds": [{"fund_code": "159103"}]},
            "sub_results": {"old": "旧数据"},
        }, as_node="commit_answer")
        with patch("app.agent.multi_agent_controller.route_query", new=AsyncMock(return_value=RouteResult(
            intent="fund_screening", resolved_query="哪些基金规模超过10亿",
        ))):
            await graph.ainvoke({"messages": [HumanMessage(content="哪些基金规模超过10亿")]}, config)
        self.assertEqual(len(seen), 1)
        self.assertIsNone(seen[0]["fund_scope"])
        self.assertEqual(seen[0]["plan"], [])
        self.assertEqual(seen[0]["sub_results"], {})

    async def test_checkpoint_resume_does_not_repeat_routing(self):
        direct = AsyncMock(side_effect=[ConnectionError("interrupted"), {
            "draft_answer": "你好", "compliance_passed": True, "synthesis_complete": True,
        }])
        with patch("app.agent.multi_agent_controller.direct_answer_node", direct), patch(
            "app.agent.multi_agent_controller.compliance_agent_node", new=AsyncMock(return_value={"compliance_passed": True}),
        ):
            graph = build_multi_agent_graph(InMemorySaver(serde=_serializer))
        config = {"configurable": {"thread_id": "resume"}}
        with patch("app.agent.multi_agent_controller.route_query", new=AsyncMock(return_value=RouteResult(intent="chitchat"))) as route:
            with self.assertRaises(ConnectionError):
                await graph.ainvoke({"messages": [HumanMessage(content="你好")]}, config)
            result = await graph.ainvoke(None, config)
        route.assert_awaited_once()
        self.assertEqual(result["final_answer"], "你好")

    async def test_clarification_completes_without_planning(self):
        with patch("app.agent.multi_agent_controller.supervisor_node", new=AsyncMock()) as supervisor:
            graph = build_multi_agent_graph(InMemorySaver(serde=_serializer))
        with patch("app.agent.multi_agent_controller.route_query", new=AsyncMock(return_value=RouteResult(
            intent="fund_query", needs_clarification=True, resolved_query="它的费率呢",
        ))):
            result = await graph.ainvoke({"messages": [HumanMessage(content="它的费率呢")]}, {"configurable": {"thread_id": "clarify"}})
        supervisor.assert_not_awaited()
        self.assertIn("基金代码", result["final_answer"])
        self.assertEqual(result["messages"][-1].content, result["final_answer"])


class PlanningIsolationTests(unittest.IsolatedAsyncioTestCase):
    async def test_screening_ignores_stale_scope_even_from_old_checkpoint(self):
        response = AIMessage(content=json.dumps({"plan": [{
            "task_id": "t1", "task_type": "rag_search", "assigned_agent": "rag_agent",
            "description": "按条件反查", "fund_codes": [], "query": "哪些基金持有宁德时代",
            "depends_on": [], "status": "pending",
        }], "reasoning": "检索持仓事实"}))
        with patch("app.agent.supervisor.ChatOpenAI"), patch(
            "app.agent.supervisor.llm_ainvoke", new=AsyncMock(return_value=response),
        ) as invoke:
            result = await _generate_new_plan(
                "哪些基金持有宁德时代", RouteResult(intent="fund_screening"),
                [HumanMessage(content="159103的持仓"), HumanMessage(content="哪些基金持有宁德时代")],
                {"funds": [{"fund_code": "159103"}]},
            )
        self.assertIsNone(result["planning_error"])
        self.assertIn("CONFIRMED_FUND_SCOPE:\nNone", invoke.call_args.args[1][1]["content"])

    def test_confirmed_scope_excludes_other_explicit_codes(self):
        plan = {"plan": [{
            "task_id": "t1", "task_type": "rag_search", "assigned_agent": "rag_agent",
            "description": "查询", "fund_codes": ["159104"], "query": "费率",
            "depends_on": [], "status": "pending",
        }], "reasoning": "查询"}
        with self.assertRaises(PlanValidationError):
            validate_supervisor_plan(plan, explicit_fund_codes={"159104"}, fund_scope={"funds": [{"fund_code": "159103"}]})
