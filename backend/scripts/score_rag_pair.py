"""计算给定 query 与内容之间的 BGE Reranker 分数。

直接请求 GPU RAG 服务中已加载的 reranker，不经过 Agent、MCP 或 Milvus 检索。

用法:
    python scripts/score_rag_pair.py "基金的复制策略是什么？" "本基金采用完全复制法..."
    python scripts/score_rag_pair.py "基金经理是谁？" --content-file E:\tmp\chunk.txt
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from pathlib import Path
from typing import Any

import httpx


def _default_base_url() -> str:
    host = os.getenv("GPU_HOST", "localhost")
    port = os.getenv("GPU_PORT", "8001")
    return f"http://{host}:{port}"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="计算 query 与内容的标准化 BGE Reranker 分数"
    )
    parser.add_argument("query", help="查询问题")
    parser.add_argument("content", nargs="?", help="需要评分的内容")
    parser.add_argument(
        "--content-file",
        type=Path,
        help="从 UTF-8 文本文件读取需要评分的内容",
    )
    parser.add_argument(
        "--base-url",
        default=_default_base_url(),
        help="GPU RAG 服务地址，默认使用 GPU_HOST/GPU_PORT 或 http://localhost:8001",
    )
    parser.add_argument("--timeout", type=float, default=60.0, help="请求超时秒数")
    parser.add_argument(
        "--output",
        choices=["text", "json"],
        default="text",
        help="输出格式，默认 text",
    )
    args = parser.parse_args()

    if not args.query.strip():
        parser.error("query 不能为空")
    if bool(args.content) == bool(args.content_file):
        parser.error("必须且只能提供 content 或 --content-file")
    if args.content_file:
        if not args.content_file.is_file():
            parser.error(f"内容文件不存在: {args.content_file}")
        args.content = args.content_file.read_text(encoding="utf-8")
    if not args.content.strip():
        parser.error("content 不能为空")
    if args.timeout <= 0:
        parser.error("--timeout 必须大于 0")
    return args


async def _score(args: argparse.Namespace) -> tuple[dict[str, Any], float]:
    start = time.monotonic()
    async with httpx.AsyncClient(
        base_url=args.base_url.rstrip("/"),
        timeout=httpx.Timeout(args.timeout, connect=min(args.timeout, 10.0)),
    ) as client:
        response = await client.post(
            "/rerank/score",
            json={"query": args.query, "contents": [args.content]},
        )
        response.raise_for_status()
        data = response.json()
    return data, time.monotonic() - start


def main() -> None:
    args = _parse_args()
    try:
        data, elapsed_seconds = asyncio.run(_score(args))
    except httpx.HTTPStatusError as exc:
        body = exc.response.text.replace("\n", " ")[:500]
        raise SystemExit(
            f"Reranker 请求失败：HTTP {exc.response.status_code}，{body}"
        ) from exc
    except httpx.HTTPError as exc:
        raise SystemExit(f"Reranker 请求失败：{type(exc).__name__}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Reranker 返回了无法解析的 JSON：{exc}") from exc

    if args.output == "json":
        print(
            json.dumps(
                {
                    "query": args.query,
                    "content": args.content,
                    "score": data["scores"][0],
                    "elapsed_seconds": elapsed_seconds,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    print(f"score: {float(data['scores'][0]):.6f}")
    print(f"耗时: {elapsed_seconds:.3f}s")


if __name__ == "__main__":
    main()
