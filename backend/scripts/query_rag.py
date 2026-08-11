"""单次调用基金年报 RAG 检索接口。

直接请求 GPU RAG 服务，不经过聊天服务、Agent 或 MCP。

用法：
    python scripts/query_rag.py "159128采用什么复制策略？" --fund-code 159128
    python scripts/query_rag.py "港股科技ETF天弘的跟踪误差目标" --top-k 5
    python scripts/query_rag.py "基金经理是谁" --fund-code 159128 --output json
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import time
from typing import Any

import httpx

FUND_CODE_RE = re.compile(r"^\d{6}$")


def _default_base_url() -> str:
    host = os.getenv("GPU_HOST", "localhost")
    port = os.getenv("GPU_PORT", "8001")
    return f"http://{host}:{port}"


def _build_payload(args: argparse.Namespace) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "query": args.query,
        "top_k": args.top_k,
        "search_type": args.search_type,
        "use_reranker": not args.no_reranker,
        "rerank_top_k": args.rerank_top_k or args.top_k * 2,
    }
    if args.fund_code:
        payload["filter_fund_code"] = args.fund_code
    if args.min_score is not None:
        payload["min_score"] = args.min_score
    return payload


def _print_text_result(data: dict[str, Any], elapsed_seconds: float) -> None:
    results = data.get("results") or []
    total = data.get("total", len(results))
    print(f"命中结果：{total} 条，展示：{len(results)} 条，耗时：{elapsed_seconds:.3f}s")

    for rank, result in enumerate(results, start=1):
        score = result.get("score")
        score_text = f"{float(score):.4f}" if isinstance(score, (int, float)) else "-"
        fund_code = result.get("fund_code") or "-"
        chunk_id = result.get("id") or "-"
        chunk_index = result.get("chunk_index")
        chunk_index_text = str(chunk_index) if chunk_index is not None else "-"
        print(
            f"\n[{rank}] score={score_text} fund_code={fund_code} "
            f"chunk_index={chunk_index_text} id={chunk_id}"
        )

        content = result.get("content")
        if content:
            print(str(content))


async def _query(args: argparse.Namespace) -> tuple[dict[str, Any], float]:
    payload = _build_payload(args)
    start = time.monotonic()
    async with httpx.AsyncClient(
        base_url=args.base_url.rstrip("/"),
        timeout=httpx.Timeout(args.timeout, connect=min(args.timeout, 10.0)),
    ) as client:
        response = await client.post("/fund_reports/search", json=payload)
        response.raise_for_status()
        data = response.json()
    return data, time.monotonic() - start


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="单次调用基金年报 RAG 检索接口")
    parser.add_argument("query", help="检索问题")
    parser.add_argument(
        "--base-url",
        default=_default_base_url(),
        help="GPU RAG 服务地址，默认使用 GPU_HOST/GPU_PORT 或 http://localhost:8001",
    )
    parser.add_argument("--fund-code", help="单基金过滤代码（6位数字）")
    parser.add_argument("--top-k", type=int, default=10, help="最终返回结果数")
    parser.add_argument(
        "--search-type",
        choices=["dense", "sparse", "hybrid"],
        default="hybrid",
        help="检索类型，默认 hybrid",
    )
    parser.add_argument("--no-reranker", action="store_true", help="关闭 reranker")
    parser.add_argument(
        "--rerank-top-k",
        type=int,
        default=None,
        help="重排前候选数，默认 top_k 的两倍",
    )
    parser.add_argument("--min-score", type=float, default=None, help="最低结果分数")
    parser.add_argument("--timeout", type=float, default=60.0, help="请求超时秒数")
    parser.add_argument(
        "--output",
        choices=["text", "json"],
        default="text",
        help="输出格式，默认 text",
    )
    args = parser.parse_args()

    if args.top_k <= 0:
        parser.error("--top-k 必须大于 0")
    if args.rerank_top_k is not None and args.rerank_top_k < args.top_k:
        parser.error("--rerank-top-k 不能小于 --top-k")
    if args.timeout <= 0:
        parser.error("--timeout 必须大于 0")
    if args.fund_code and not FUND_CODE_RE.fullmatch(args.fund_code):
        parser.error("--fund-code 必须是6位数字")
    return args


def main() -> None:
    args = _parse_args()
    try:
        data, elapsed_seconds = asyncio.run(_query(args))
    except httpx.HTTPStatusError as exc:
        body = exc.response.text.replace("\n", " ")[:500]
        raise SystemExit(f"RAG 请求失败：HTTP {exc.response.status_code}，{body}") from exc
    except httpx.HTTPError as exc:
        raise SystemExit(f"RAG 请求失败：{type(exc).__name__}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise SystemExit(f"RAG 返回了无法解析的 JSON：{exc}") from exc

    if args.output == "json":
        print(json.dumps(data, ensure_ascii=False, indent=2))
    else:
        _print_text_result(data, elapsed_seconds)


if __name__ == "__main__":
    main()
