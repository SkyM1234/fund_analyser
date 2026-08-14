"""GPU 检索接口并发压测脚本。

直接向 embedding service 发起请求，绕过 FastAPI 聊天接口、Celery、Agent 和 LLM，
用于测量 GPU embedding、Milvus 检索和 reranker 的整体性能。

用法：
    # 压测基金报告检索接口（20 个请求同时发起）
    python backend/scripts/gpu_load_test.py --requests 20 --query "金融科技ETF汇添富的基金经理是谁"

    # 限制并发为 8，总共发送 100 个请求
    python backend/scripts/gpu_load_test.py --requests 100 --concurrency 8

    # 压测基金名称识别接口
    python backend/scripts/gpu_load_test.py --endpoint index --requests 20 --query "金融科技ETF汇添富"

依赖：httpx（项目已用于 rag_client，若未安装则 pip install httpx）
"""
import argparse
import asyncio
import math
import time
from dataclasses import dataclass

import httpx

DEFAULT_BASE_URL = "http://127.0.0.1:8001"


@dataclass
class RequestResult:
    request_id: str
    ok: bool = False
    error: str | None = None
    latency: float | None = None
    result_count: int | None = None


def _percentile(values: list[float], percentile: float) -> float:
    """使用线性插值计算百分位数。"""
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]

    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]

    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _build_payload(args: argparse.Namespace) -> dict:
    if args.endpoint == "index":
        return {
            "query": args.query,
            "top_k": args.top_k,
            "min_score": args.index_min_score,
        }

    return {
        "query": args.query,
        "top_k": args.top_k,
        "filter_fund_code": args.fund_code,
        "search_type": args.search_type,
        "use_reranker": not args.no_reranker,
    }


async def _run_one_request(
    client: httpx.AsyncClient,
    semaphore: asyncio.Semaphore,
    endpoint_path: str,
    request_id: str,
    payload: dict,
) -> RequestResult:
    result = RequestResult(request_id=request_id)

    async with semaphore:
        start = time.monotonic()
        try:
            resp = await client.post(endpoint_path, json=payload)
            result.latency = time.monotonic() - start

            if resp.status_code != 200:
                body = resp.text.replace("\n", " ")
                result.error = f"HTTP {resp.status_code}: {body[:200]}"
                return result

            data = resp.json()
            result.result_count = data.get("total")
            result.ok = True
        except httpx.TimeoutException:
            result.latency = time.monotonic() - start
            result.error = f"超时（>{client.timeout.read}s）"
        except Exception as e:
            result.latency = time.monotonic() - start
            result.error = f"{type(e).__name__}: {e}"

    return result


async def _warm_up(
    client: httpx.AsyncClient,
    endpoint_path: str,
    payload: dict,
    count: int,
) -> None:
    if count <= 0:
        return

    print(f"[gpu_load_test] 预热 {count} 次 -> {client.base_url}{endpoint_path}")
    for i in range(count):
        resp = await client.post(endpoint_path, json=payload)
        if resp.status_code != 200:
            body = resp.text.replace("\n", " ")
            raise RuntimeError(f"预热请求 {i + 1} 失败，HTTP {resp.status_code}: {body[:200]}")


async def main():
    parser = argparse.ArgumentParser(description="GPU 检索接口并发压测")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="embedding service 地址")
    parser.add_argument(
        "--endpoint",
        choices=["reports", "index"],
        default="reports",
        help="压测接口：reports=基金报告检索，index=基金名称识别",
    )
    parser.add_argument("--requests", type=int, default=20, help="总请求数")
    parser.add_argument(
        "--concurrency",
        type=int,
        default=None,
        help="最大并发数，默认与总请求数相同",
    )
    parser.add_argument("--warmup", type=int, default=3, help="正式压测前的串行预热次数")
    parser.add_argument(
        "--query",
        default="金融科技ETF汇添富的基金经理是谁",
        help="每个请求使用的查询文本",
    )
    parser.add_argument("--top-k", type=int, default=10, help="最终返回结果数，不能低于 10")
    parser.add_argument("--fund-code", default=None, help="报告检索使用的基金代码过滤条件")
    parser.add_argument(
        "--search-type",
        choices=["dense", "sparse", "hybrid"],
        default="hybrid",
        help="报告检索类型",
    )
    parser.add_argument("--no-reranker", action="store_true", help="关闭报告检索重排")
    parser.add_argument(
        "--index-min-score",
        type=float,
        default=0.5,
        help="基金名称识别最低置信度，仅用于 --endpoint index",
    )
    parser.add_argument("--timeout", type=float, default=120.0, help="单个请求超时（秒）")
    args = parser.parse_args()

    if args.requests <= 0:
        parser.error("--requests 必须大于 0")
    if args.warmup < 0:
        parser.error("--warmup 不能小于 0")
    if args.top_k < 10:
        parser.error("--top-k 不能低于 10")

    concurrency = args.concurrency or args.requests
    if concurrency <= 0:
        parser.error("--concurrency 必须大于 0")
    concurrency = min(concurrency, args.requests)

    endpoint_path = (
        "/fund_reports/search" if args.endpoint == "reports" else "/fund_index/search"
    )
    payload = _build_payload(args)
    limits = httpx.Limits(
        max_connections=concurrency,
        max_keepalive_connections=concurrency,
    )

    async with httpx.AsyncClient(
        base_url=args.base_url,
        timeout=args.timeout,
        limits=limits,
    ) as client:
        try:
            await _warm_up(client, endpoint_path, payload, args.warmup)
        except Exception as e:
            raise SystemExit(f"[gpu_load_test] 预热失败: {type(e).__name__}: {e}") from e

        semaphore = asyncio.Semaphore(concurrency)
        tasks = [
            _run_one_request(
                client,
                semaphore,
                endpoint_path,
                f"gpu_{i}",
                payload,
            )
            for i in range(args.requests)
        ]

        print(
            f"[gpu_load_test] 并发发起 {args.requests} 个请求"
            f"（并发上限 {concurrency}）-> {args.base_url}{endpoint_path}"
        )
        wall_start = time.monotonic()
        results: list[RequestResult] = await asyncio.gather(*tasks)
        wall_elapsed = time.monotonic() - wall_start

    print("\n" + "=" * 70)
    print(f"{'请求':<16}{'结果':<8}{'耗时(s)':<12}{'结果数':<10}备注")
    print("-" * 70)
    for result in results:
        status = "OK" if result.ok else "FAIL"
        latency = f"{result.latency:.3f}" if result.latency is not None else "-"
        result_count = str(result.result_count) if result.result_count is not None else "-"
        note = result.error or ""
        print(
            f"{result.request_id:<16}{status:<8}{latency:<12}"
            f"{result_count:<10}{note}"
        )

    ok_count = sum(1 for result in results if result.ok)
    latencies = [
        result.latency
        for result in results
        if result.ok and result.latency is not None
    ]

    print("-" * 70)
    print(f"总墙钟耗时: {wall_elapsed:.3f}s | 成功: {ok_count}/{len(results)}")
    if latencies:
        throughput = ok_count / wall_elapsed if wall_elapsed > 0 else 0.0
        print(
            f"请求耗时 avg={sum(latencies)/len(latencies):.3f}s "
            f"p50={_percentile(latencies, 0.50):.3f}s "
            f"p90={_percentile(latencies, 0.90):.3f}s "
            f"p95={_percentile(latencies, 0.95):.3f}s"
        )
        print(
            f"请求耗时 max={max(latencies):.3f}s min={min(latencies):.3f}s "
            f"| 吞吐量={throughput:.2f} req/s"
        )
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
