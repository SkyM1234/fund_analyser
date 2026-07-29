"""RAG 检索 target：直接调用 GPU RAG 服务的 /fund_reports/search 接口。

不走 MCP，是为了让评测和实际 Agent 行为完全解耦——专门测向量库 + 重排的质量。
若想评测"完整链路（路由→注入→MCP→GPU）"，请用 agent_target.py。

LangSmith 约定：target 函数 inputs(dict) → outputs(dict)
"""
from __future__ import annotations

import logging

import httpx

from app.core.config import get_settings

logger = logging.getLogger(__name__)


async def rag_search_target(inputs: dict) -> dict:
    """直连 GPU /fund_reports/search，返回标准化结果供 evaluator 使用。

    Args:
        inputs: 从 example.inputs 传入，包含 {"query": str, "filter_fund_code": ..., "top_k": int}

    Returns:
        {
            "results": [chunk_dict, ...],   # 给 retrieval_metrics 用
            "raw_total": int,               # GPU 端返回总数
        }
    """
    s = get_settings()

    payload = {
        "query": inputs["query"],
        "top_k": inputs.get("top_k", 10),
        "search_type": inputs.get("search_type", "hybrid"),
        "use_reranker": inputs.get("use_reranker", True),
    }

    filter_codes = inputs.get("filter_fund_code")
    if filter_codes:
        payload["filter_fund_code"] = filter_codes
    if inputs.get("min_score") is not None:
        payload["min_score"] = inputs["min_score"]

    url = f"{s.gpu_base_url}/fund_reports/search"
    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post(url, json=payload)
        resp.raise_for_status()
        data = resp.json()

    results = data.get("results") or []
    # 标准化：补充 id 字段，方便 chunk-level 相关性判定
    for r in results:
        if "id" not in r:
            r["id"] = f"{r.get('fund_code')}_{r.get('chunk_index')}"

    return {
        "results": results,
        "raw_total": data.get("total", len(results)),
    }
