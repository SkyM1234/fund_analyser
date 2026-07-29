"""基金名称识别 target：直接调用 GPU RAG 服务的 /fund_index/search 接口。

评测"两级RAG第一级"——即 rag_identify_funds 工具底层调用的语义识别能力：
从用户提供的基金名称/别名/模糊描述中识别出正确的基金代码。

与 rag_target.py（评测检索质量）解耦：本 target 只关心"名称→代码"这一跳，
不涉及向量库内容检索，避免像 retrieval.jsonl 那样通过人工预置正确
filter_fund_code 把这一跳的准确率评测"绕过"。

LangSmith 约定：target 函数 inputs(dict) → outputs(dict)
"""
from __future__ import annotations

import logging

import httpx

from app.core.config import get_settings

logger = logging.getLogger(__name__)


async def name_resolution_target(inputs: dict) -> dict:
    """直连 GPU /fund_index/search，返回识别到的基金代码列表供 evaluator 使用。

    Args:
        inputs: 从 example.inputs 传入，包含 {"query": str, "top_k": int, "min_score": float}

    Returns:
        {
            "identified_codes": [str, ...],   # 按置信度排序的识别结果
            "scores": [float, ...],           # 对应置信度
            "raw_results": [dict, ...],       # GPU 端原始返回
        }
    """
    s = get_settings()

    payload = {
        "query": inputs["query"],
        "top_k": inputs.get("top_k", 5),
        "min_score": inputs.get("min_score", 0.5),
    }

    url = f"{s.gpu_base_url}/fund_index/search"
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(url, json=payload)
        resp.raise_for_status()
        data = resp.json()

    results = data.get("results") or []

    return {
        "identified_codes": [r.get("fund_code") for r in results],
        "scores": [r.get("score") for r in results],
        "raw_results": results,
    }
