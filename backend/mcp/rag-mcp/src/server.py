#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""RAG MCP Server"""

import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Any, Sequence

sys.path.insert(0, str(Path(__file__).parent))

from rag_client import RagClient
from fund_code_matcher import FundCodeMatcher, load_fund_code_matcher

from mcp.server.models import InitializationOptions
from mcp.server import NotificationOptions, Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s", stream=sys.stderr)
logger = logging.getLogger("rag-mcp")

server = Server("rag-mcp")
rag_client: RagClient | None = None
fund_matcher: FundCodeMatcher | None = None
RAG_SEARCH_TYPE = "hybrid"
RAG_USE_RERANKER = True


@server.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="rag_search",
            description=(
                "检索基金年报内容。支持混合检索(dense+sparse)和重排序。\n"
                "每次调用只能指定一个基金代码：filter_fund_code 传单只基金的6位代码字符串，"
                "如 '159103'；不需要过滤时留空（全局检索），全局检索时，top_k 需要往大调。\n"
                "多基金对比时，每只基金单独调用一次本工具。"
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "查询文本"},
                    "top_k": {
                        "type": "integer",
                        "description": "返回结果数量，默认 10；小于 10 时自动按 10 处理；启用重排时粗排候选数自动为 top_k 的 3 倍",
                        "default": 10,
                    },
                    "filter_fund_code": {
                        "description": "单基金代码过滤，如 '159103'；留空表示全局检索",
                        "oneOf": [
                            {"type": "string"},
                            {"type": "null"},
                        ],
                    },
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        ),
        Tool(
            name="rag_identify_funds",
            description=(
                "两级RAG第一级：从用户问题中语义识别基金代码。\n"
                "适用场景：用户用别名、简称或模糊描述提及基金（如'汇添富的科技ETF'、'那个机器人基金'）。\n"
                "注意：若查询中已包含6位数字代码（如159103），直接使用该代码，无需调用此工具。\n"
                "返回：匹配到的基金代码列表及置信度，score >= 0.7 视为高置信度匹配。"
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "包含基金名称或描述的查询文本"},
                    "top_k": {"type": "integer", "description": "最多返回几个候选基金，默认 5", "default": 5},
                    "min_score": {
                        "type": "number",
                        "description": "最低置信度阈值，默认 0.5；提高到 0.7 可减少误匹配",
                        "default": 0.5,
                    },
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        ),
        Tool(name="rag_health", description="检查 RAG 服务健康状态", inputSchema={"type": "object", "properties": {}}),
        Tool(name="rag_stats", description="获取 RAG 服务统计信息", inputSchema={"type": "object", "properties": {}}),
        Tool(name="rag_list_funds", description="获取基金清单（从年报索引读取，兼容旧接口）", inputSchema={"type": "object", "properties": {}}),
        Tool(
            name="rag_match_fund_codes",
            description="从文本中提取基金代码（字符串匹配，已被 rag_identify_funds 语义识别所替代，保留作兜底）",
            inputSchema={"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: Any) -> Sequence[TextContent]:
    try:
        if name == "rag_search":
            return await handle_rag_search(arguments)
        elif name == "rag_identify_funds":
            return await handle_identify_funds(arguments)
        elif name == "rag_health":
            return await handle_rag_health()
        elif name == "rag_stats":
            return await handle_rag_stats()
        elif name == "rag_list_funds":
            return await handle_rag_list_funds()
        elif name == "rag_match_fund_codes":
            return await handle_match_fund_codes(arguments)
        return [TextContent(type="text", text=f"未知工具: {name}")]
    except Exception as e:
        logger.error(f"Tool {name} failed: {e}", exc_info=True)
        return [TextContent(type="text", text=f"工具执行失败: {str(e)}")]


async def handle_identify_funds(args: dict) -> Sequence[TextContent]:
    """处理 rag_identify_funds 工具调用：语义识别基金代码"""
    global rag_client
    if rag_client is None:
        rag_client = RagClient()

    results = await rag_client.identify_funds(
        query=args["query"],
        top_k=args.get("top_k", 5),
        min_score=args.get("min_score", 0.5),
    )

    if not results:
        return [TextContent(type="text", text="未从查询中识别到匹配的基金（置信度不足）")]

    lines = [f"识别到 {len(results)} 只基金:"]
    for r in results:
        lines.append(
            f"- {r['fund_code']}: {r['full_name']}  (置信度: {r['score']:.3f})"
        )

    return [TextContent(type="text", text="\n".join(lines))]


async def handle_rag_search(args: dict) -> Sequence[TextContent]:
    global rag_client
    if rag_client is None:
        rag_client = RagClient()

    # 使用 dict.get(key, default) 而不是 or，避免 False/0 被替换
    requested_top_k = args.get("top_k", 10)
    top_k = max(10, requested_top_k)
    results = await rag_client.search(
        query=args["query"],
        top_k=top_k,
        filter_fund_code=args.get("filter_fund_code"),
        search_type=RAG_SEARCH_TYPE,
        use_reranker=RAG_USE_RERANKER,
    )
    if not results:
        return [TextContent(type="text", text="未找到相关内容")]
    formatted = []
    for i, r in enumerate(results, 1):
        formatted.append(
            f"--- 结果 {i} (相似度: {r.get('score', 0):.4f}) ---\n"
            f"Chunk ID: {r.get('id', '')}\n"
            f"Chunk Index: {r.get('chunk_index', '')}\n"
            f"基金代码: {r.get('fund_code', 'N/A')}\n"
            # f"文档类型: {r.get('doc_type', 'N/A')}\n"
            # f"报告时间: {r.get('report_date', 'N/A')}\n"
            f"内容:\n{r.get('content', '')}\n"
        )
    return [TextContent(type="text", text="\n".join(formatted))]


async def handle_rag_health() -> Sequence[TextContent]:
    global rag_client
    if rag_client is None:
        rag_client = RagClient()
    data = await rag_client.health()
    return [TextContent(type="text", text=json.dumps(data, ensure_ascii=False, indent=2))]


async def handle_rag_stats() -> Sequence[TextContent]:
    global rag_client
    if rag_client is None:
        rag_client = RagClient()
    data = await rag_client.stats()
    return [TextContent(type="text", text=json.dumps(data, ensure_ascii=False, indent=2))]


async def handle_rag_list_funds() -> Sequence[TextContent]:
    global rag_client
    if rag_client is None:
        rag_client = RagClient()
    funds = await rag_client.list_funds()
    if not funds:
        return [TextContent(type="text", text="未找到基金清单")]
    fund_list = [f"{fund['code']}: {fund['name']}" for fund in funds]
    return [TextContent(type="text", text=f"共 {len(funds)} 只基金:\n" + "\n".join(fund_list))]


async def handle_match_fund_codes(args: dict) -> Sequence[TextContent]:
    global rag_client, fund_matcher
    if rag_client is None:
        rag_client = RagClient()
    if fund_matcher is None:
        logger.info("Loading fund code matcher...")
        fund_matcher = await load_fund_code_matcher(rag_client)
        logger.info(f"Loaded {len(fund_matcher.funds)} funds")
    codes = fund_matcher.match_codes(args["query"])
    if not codes:
        return [TextContent(type="text", text="未从查询中识别到基金代码或名称")]
    results = []
    for code in codes:
        fund = fund_matcher.get_fund(code)
        if fund:
            results.append(f"- {code}: {fund.short_name}")
    return [TextContent(type="text", text=f"识别到 {len(codes)} 只基金:\n" + "\n".join(results))]


async def main():
    logger.info("Starting RAG MCP Server...")
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="rag-mcp",
                server_version="0.1.0",
                capabilities=server.get_capabilities(notification_options=NotificationOptions(), experimental_capabilities={}),
            ),
        )


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Server crashed: {e}", exc_info=True)
        sys.exit(1)
