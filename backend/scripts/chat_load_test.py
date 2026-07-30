"""聊天 SSE 接口并发压测脚本。

认证准备不计入请求延迟，正式压测测量：
- 首 token 延迟（TTFT）
- 请求总耗时
- 成功率、延迟百分位数和吞吐量

用法：
    # 20 个请求同时发起
    python backend/scripts/chat_load_test.py --requests 20 --message "金融科技ETF汇添富的基金经理是谁"

    # 总共 100 个请求，最多 8 个并发
    python backend/scripts/chat_load_test.py --requests 100 --concurrency 8

    # 同一用户、同一会话并发，用于验证会话锁
    python backend/scripts/chat_load_test.py --requests 5 --same-session

依赖：httpx（项目已使用该依赖）
"""
import argparse
import asyncio
import json
import math
import time
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass, field

import httpx

DEFAULT_BASE_URL = "http://127.0.0.1:8800"
TEST_USER_PREFIX = "loadtest_"
TEST_PASSWORD = "LoadTest123!"


@dataclass
class TestIdentity:
    username: str
    access_token: str


@dataclass
class RequestResult:
    request_id: str
    username: str
    session_id: str
    ok: bool = False
    error: str | None = None
    first_token_latency: float | None = None
    total_latency: float | None = None
    token_events: int = 0
    event_counts: dict[str, int] = field(default_factory=dict)


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


async def _prepare_identity(
    client: httpx.AsyncClient,
    username: str,
) -> TestIdentity:
    """确保测试用户存在并登录，返回认证信息。"""
    register_resp = await client.post(
        "/api/auth/register",
        json={
            "username": username,
            "email": f"{username}@loadtest.local",
            "password": TEST_PASSWORD,
        },
    )
    if register_resp.status_code not in (201, 409):
        body = register_resp.text.replace("\n", " ")
        raise RuntimeError(
            f"用户 {username} 注册失败，HTTP {register_resp.status_code}: {body[:200]}"
        )

    login_resp = await client.post(
        "/api/auth/login",
        json={"username": username, "password": TEST_PASSWORD},
    )
    if login_resp.status_code != 200:
        body = login_resp.text.replace("\n", " ")
        raise RuntimeError(
            f"用户 {username} 登录失败，HTTP {login_resp.status_code}: {body[:200]}"
        )

    return TestIdentity(
        username=username,
        access_token=login_resp.json()["access_token"],
    )


async def _iter_sse(resp: httpx.Response) -> AsyncIterator[tuple[str, str]]:
    """按 SSE 规范解析 event 和多行 data 字段。"""
    event_name = "message"
    data_lines: list[str] = []

    async for line in resp.aiter_lines():
        if line == "":
            if data_lines:
                yield event_name, "\n".join(data_lines)
            event_name = "message"
            data_lines = []
            continue

        if line.startswith(":"):
            continue
        if line.startswith("event:"):
            event_name = line[len("event:"):].strip()
        elif line.startswith("data:"):
            data_lines.append(line[len("data:"):].lstrip())

    if data_lines:
        yield event_name, "\n".join(data_lines)


async def _consume_chat_stream(
    client: httpx.AsyncClient,
    identity: TestIdentity,
    message: str,
    session_id: str,
    result: RequestResult,
    start: float,
) -> None:
    headers = {"Authorization": f"Bearer {identity.access_token}"}
    payload = {"message": message, "session_id": session_id, "history": []}

    async with client.stream(
        "POST",
        "/api/chat/stream",
        json=payload,
        headers=headers,
    ) as resp:
        if resp.status_code != 200:
            body = (await resp.aread()).decode(errors="replace").replace("\n", " ")
            result.error = f"HTTP {resp.status_code}: {body[:200]}"
            return

        async for event_name, data_text in _iter_sse(resp):
            result.event_counts[event_name] = result.event_counts.get(event_name, 0) + 1
            try:
                data = json.loads(data_text)
            except json.JSONDecodeError:
                result.error = f"无法解析 SSE {event_name} 事件: {data_text[:160]}"
                return

            if event_name == "token":
                result.token_events += 1
                if result.first_token_latency is None and data.get("delta"):
                    result.first_token_latency = time.monotonic() - start
            elif event_name == "error":
                result.error = data.get("message", "服务端返回 error 事件")
                return
            elif event_name == "done":
                if data.get("finish_reason") == "stop":
                    result.ok = True
                else:
                    result.error = f"非正常结束: {data}"
                return

    if result.error is None and not result.ok:
        result.error = "连接结束但未收到 done 事件"


async def _run_one_request(
    client: httpx.AsyncClient,
    semaphore: asyncio.Semaphore,
    identity: TestIdentity,
    request_id: str,
    message: str,
    session_id: str,
    timeout: float,
) -> RequestResult:
    result = RequestResult(
        request_id=request_id,
        username=identity.username,
        session_id=session_id,
    )

    async with semaphore:
        start = time.monotonic()
        try:
            await asyncio.wait_for(
                _consume_chat_stream(
                    client,
                    identity,
                    message,
                    session_id,
                    result,
                    start,
                ),
                timeout=timeout,
            )
        except asyncio.TimeoutError:
            result.error = f"总耗时超时（>{timeout}s）"
        except httpx.TimeoutException:
            result.error = f"网络读取超时（>{timeout}s）"
        except Exception as exc:
            result.error = f"{type(exc).__name__}: {exc}"
        finally:
            result.total_latency = time.monotonic() - start

    return result


async def _warm_up(
    client: httpx.AsyncClient,
    identity: TestIdentity,
    message: str,
    count: int,
    timeout: float,
) -> None:
    if count <= 0:
        return

    print(f"[chat_load_test] 串行预热 {count} 次 -> {client.base_url}api/chat/stream")
    semaphore = asyncio.Semaphore(1)
    for i in range(count):
        result = await _run_one_request(
            client,
            semaphore,
            identity,
            f"warmup_{i}",
            message,
            str(uuid.uuid4()),
            timeout,
        )
        if not result.ok:
            raise RuntimeError(f"预热请求 {i + 1} 失败: {result.error}")


def _print_latency_summary(label: str, values: list[float]) -> None:
    if not values:
        return

    print(
        f"{label} avg={sum(values)/len(values):.2f}s "
        f"p50={_percentile(values, 0.50):.2f}s "
        f"p90={_percentile(values, 0.90):.2f}s "
        f"p95={_percentile(values, 0.95):.2f}s "
        f"max={max(values):.2f}s min={min(values):.2f}s"
    )


async def main() -> None:
    parser = argparse.ArgumentParser(description="聊天 SSE 接口并发压测")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="后端地址")
    parser.add_argument(
        "--requests",
        dest="requests",
        type=int,
        default=20,
        help="总请求数",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=None,
        help="最大并发数，默认与总请求数相同",
    )
    parser.add_argument("--warmup", type=int, default=0, help="正式压测前的串行预热次数")
    parser.add_argument(
        "--message",
        default="科创债ETF万家的最新净值是多少",
        help="每个请求发送的问题",
    )
    parser.add_argument(
        "--same-session",
        action="store_true",
        help="使用同一用户和 session_id，用于验证会话锁",
    )
    parser.add_argument("--timeout", type=float, default=180.0, help="单个请求总超时（秒）")
    args = parser.parse_args()

    if args.requests <= 0:
        parser.error("--requests 必须大于 0")
    if args.warmup < 0:
        parser.error("--warmup 不能小于 0")
    if args.timeout <= 0:
        parser.error("--timeout 必须大于 0")

    concurrency = args.concurrency or args.requests
    if concurrency <= 0:
        parser.error("--concurrency 必须大于 0")
    concurrency = min(concurrency, args.requests)

    identity_count = 1 if args.same_session else concurrency
    limits = httpx.Limits(
        max_connections=concurrency,
        max_keepalive_connections=concurrency,
    )

    async with httpx.AsyncClient(
        base_url=args.base_url,
        timeout=args.timeout,
        limits=limits,
    ) as client:
        print(f"[chat_load_test] 准备 {identity_count} 个测试用户（不计入压测耗时）")
        try:
            identities = await asyncio.gather(*(
                _prepare_identity(client, f"{TEST_USER_PREFIX}{i}")
                for i in range(identity_count)
            ))
            await _warm_up(
                client,
                identities[0],
                args.message,
                args.warmup,
                args.timeout,
            )
        except Exception as exc:
            raise SystemExit(
                f"[chat_load_test] 准备阶段失败: {type(exc).__name__}: {exc}"
            ) from exc

        shared_session_id = str(uuid.uuid4()) if args.same_session else None
        semaphore = asyncio.Semaphore(concurrency)
        tasks = [
            _run_one_request(
                client,
                semaphore,
                identities[0] if args.same_session else identities[i % identity_count],
                f"chat_{i}",
                args.message,
                shared_session_id or str(uuid.uuid4()),
                args.timeout,
            )
            for i in range(args.requests)
        ]

        print(
            f"[chat_load_test] 发起 {args.requests} 个请求"
            f"（并发上限 {concurrency}）-> {args.base_url}/api/chat/stream"
        )
        if shared_session_id:
            print(
                "[chat_load_test] 会话锁模式："
                f"user={identities[0].username}, session_id={shared_session_id}"
            )

        wall_start = time.monotonic()
        results: list[RequestResult] = await asyncio.gather(*tasks)
        wall_elapsed = time.monotonic() - wall_start

    print("\n" + "=" * 86)
    print(
        f"{'请求':<14}{'用户':<16}{'结果':<8}"
        f"{'首Token(s)':<14}{'总耗时(s)':<12}{'Token事件':<12}备注"
    )
    print("-" * 86)
    for result in results:
        status = "OK" if result.ok else "FAIL"
        first_token = (
            f"{result.first_token_latency:.2f}"
            if result.first_token_latency is not None
            else "-"
        )
        total = (
            f"{result.total_latency:.2f}"
            if result.total_latency is not None
            else "-"
        )
        print(
            f"{result.request_id:<14}{result.username:<16}{status:<8}"
            f"{first_token:<14}{total:<12}{result.token_events:<12}"
            f"{result.error or ''}"
        )

    ok_count = sum(1 for result in results if result.ok)
    failed_count = len(results) - ok_count
    success_rate = ok_count / len(results) * 100
    request_throughput = len(results) / wall_elapsed if wall_elapsed > 0 else 0.0
    success_throughput = ok_count / wall_elapsed if wall_elapsed > 0 else 0.0
    total_latencies = [
        result.total_latency
        for result in results
        if result.ok and result.total_latency is not None
    ]
    first_token_latencies = [
        result.first_token_latency
        for result in results
        if result.ok and result.first_token_latency is not None
    ]

    print("-" * 86)
    print(
        f"总墙钟耗时: {wall_elapsed:.2f}s | 成功: {ok_count}/{len(results)} "
        f"| 失败: {failed_count} | 成功率: {success_rate:.1f}%"
    )
    print(
        f"请求吞吐量: {request_throughput:.2f} req/s "
        f"| 成功吞吐量: {success_throughput:.2f} req/s"
    )
    _print_latency_summary("首Token延迟", first_token_latencies)
    _print_latency_summary("请求总耗时", total_latencies)
    print("=" * 86)


if __name__ == "__main__":
    asyncio.run(main())
