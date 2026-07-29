"""Agent 端到端 target：驱动完整 LangGraph 链路。

- 每个 example 用独立 thread_id，互不污染
- 用内存 checkpointer，避免污染线上 PG 数据
- 抽出最终回答、cited_fund_codes、intent

⚠️ 多进程兼容性：
langsmith.aevaluate 可能用多进程执行 target，全局缓存会失效。
因此 MCP 初始化放在 _ensure_mcp_ready() 里，每个 worker 独立启动。

⚠️ MCP 清理警告（可忽略）：
进程退出时可能出现 anyio 的 "Attempted to exit cancel scope in a different task" 错误，
这是 MCP stdio 客户端在多进程环境下的已知问题，不影响评测结果。
"""
from __future__ import annotations

import asyncio
import logging
import re
import uuid

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver

from app.agent.multi_agent_controller import build_multi_agent_graph

logger = logging.getLogger(__name__)

FUND_CODE_RE = re.compile(r"\b\d{6}\b")

# 每个进程独立的 checkpointer 和 graph
_eval_checkpointer = MemorySaver()
_eval_graph = None
_mcp_initialized = False
_init_lock = asyncio.Lock()
_mcp_client_ref = None  # 保留 client 引用，用于清理


async def _ensure_mcp_ready():
    """确保 MCP 工具已加载（懒初始化，worker 进程首次调用时触发）。"""
    global _mcp_initialized, _mcp_client_ref
    if _mcp_initialized:
        return

    async with _init_lock:
        if _mcp_initialized:  # 双重检查
            return

        logger.info("[agent_target] 初始化 MCP 客户端...")
        from app.core.config import get_settings
        from app.services.mcp_client import MCPServerConfig, get_mcp_client, set_cached_mcp_tools
        from app.tools.mcp_adapter import load_mcp_tools
        from eval.config import get_eval_settings

        s = get_settings()
        eval_s = get_eval_settings()

        if not s.MCP_ENABLED:
            logger.warning("MCP_ENABLED=False，Agent 将无工具")
            _mcp_initialized = True
            return

        # 根据评测配置过滤 MCP 服务器
        configs = []
        for cfg in s.mcp_servers_list:
            server_name = cfg.get("name", "")
            # 如果禁用 cn-funds-mcp，则跳过该服务器
            if server_name == "cn-funds-mcp" and not eval_s.ENABLE_CN_FUNDS_MCP:
                logger.info(f"[agent_target] 跳过 {server_name}（ENABLE_CN_FUNDS_MCP=False）")
                continue

            configs.append(
                MCPServerConfig(
                    name=server_name,
                    command=cfg.get("command", ""),
                    args=cfg.get("args", []),
                    env=cfg.get("env", {}),
                    cwd=cfg.get("cwd"),
                )
            )

        client = await get_mcp_client()
        _mcp_client_ref = client  # 保留引用（暂未使用，进程退出时很难优雅关闭）
        await client.initialize(configs)
        tools = await load_mcp_tools()
        set_cached_mcp_tools(tools)
        logger.info(f"[agent_target] ✓ MCP 启动完成，{len(tools)} 个工具")
        _mcp_initialized = True


async def _get_graph():
    """懒加载 graph（在 MCP 初始化后）。"""
    global _eval_graph
    if _eval_graph is None:
        await _ensure_mcp_ready()
        from app.services.mcp_client import get_cached_mcp_tools

        tools = get_cached_mcp_tools()
        _eval_graph = build_multi_agent_graph(_eval_checkpointer)
        logger.info(f"[agent_target] Graph 已构建，绑定 {len(tools)} 个工具")
    return _eval_graph


async def agent_target(inputs: dict) -> dict:
    """运行完整 Agent，返回评测所需 outputs。

    Args:
        inputs: 从 example.inputs 传入，包含 {"query": str}

    Returns:
        {
            "answer": str,
            "cited_fund_codes": list[str],
            "intent": str | None,
            "tool_calls": list[dict],   # 调试用
        }
    """
    query = inputs["query"]
    thread_id = f"eval-{uuid.uuid4().hex[:12]}"
    config = {"configurable": {"thread_id": thread_id}}

    graph = await _get_graph()
    state = await graph.ainvoke(
        {"messages": [HumanMessage(content=query)]},
        config=config,
    )

    messages = state.get("messages", [])
    # 最后一条 AI 消息作为最终回答
    final_answer = ""
    for m in reversed(messages):
        if getattr(m, "type", None) == "ai" and getattr(m, "content", None):
            final_answer = m.content
            break

    route_result = state.get("route_result")
    intent = route_result.intent if route_result is not None else None

    cited_codes = sorted(set(FUND_CODE_RE.findall(final_answer)))

    tool_calls = [
        {"name": entry.get("name"), "args": entry.get("args")}
        for entry in state.get("tool_call_log", [])
    ]

    return {
        "answer": final_answer,
        "cited_fund_codes": cited_codes,
        "intent": intent,
        "tool_calls": tool_calls,
    }
