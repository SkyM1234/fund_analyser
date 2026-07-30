"""并发多用户压测脚本。

模拟 N 个用户同时向 /api/chat/stream 发起请求，测量：
- 整体耗时（发起请求 -> 收到 done/error）
- 首字延迟（发起请求 -> 收到第一个 token 事件），衡量排队等待时间
- 成功/失败/超时数量

用法：
    # Docker 内运行
    docker exec fund-backend python -m scripts.load_test --users 5 --message "159103的最新净值"
    # 裸机运行（在项目根目录下）
    python backend/scripts/load_test.py --users 5 --message "金融科技ETF汇添富的基金经理是谁"

依赖：httpx（项目已用于 rag_client，若未安装则 pip install httpx）
"""
import argparse
import asyncio
import json
import time
import uuid
from dataclasses import dataclass, field

import httpx

DEFAULT_BASE_URL = "http://127.0.0.1:8800"
TEST_USER_PREFIX = "loadtest_"
TEST_PASSWORD = "LoadTest123!"


@dataclass
class UserResult:
    username: str
    session_id: str
    ok: bool = False
    error: str | None = None
    first_token_latency: float | None = None
    total_latency: float | None = None
    event_counts: dict = field(default_factory=dict)


async def _ensure_user(client: httpx.AsyncClient, username: str) -> str:
    """注册（若已存在则忽略）并登录，返回 access_token。"""
    await client.post("/api/auth/register", json={
        "username": username,
        "email": f"{username}@loadtest.local",
        "password": TEST_PASSWORD,
    })
    resp = await client.post("/api/auth/login", json={
        "username": username,
        "password": TEST_PASSWORD,
    })
    resp.raise_for_status()
    return resp.json()["access_token"]


async def _run_one_user(
    base_url: str,
    username: str,
    message: str,
    session_id: str,
    timeout: float,
) -> UserResult:
    result = UserResult(username=username, session_id=session_id)
    start = time.monotonic()

    try:
        async with httpx.AsyncClient(base_url=base_url, timeout=timeout) as client:
            token = await _ensure_user(client, username)
            headers = {"Authorization": f"Bearer {token}"}

            payload = {"message": message, "session_id": session_id, "history": []}

            async with client.stream(
                "POST", "/api/chat/stream", json=payload, headers=headers
            ) as resp:
                if resp.status_code != 200:
                    result.error = f"HTTP {resp.status_code}: {await resp.aread()}"
                    return result

                async for line in resp.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    data_str = line[len("data:"):].strip()
                    if not data_str:
                        continue
                    try:
                        data = json.loads(data_str)
                    except json.JSONDecodeError:
                        continue

                    # sse_starlette 把 event 字段单独发在前一行，这里简化为按内容特征分类
                    if result.first_token_latency is None and "delta" in data:
                        result.first_token_latency = time.monotonic() - start
                    if "message" in data and "delta" not in data:
                        # error 事件
                        result.error = data["message"]
                    if data.get("finish_reason") == "stop":
                        result.ok = True
                        break

        result.total_latency = time.monotonic() - start
        if result.error is None and result.total_latency is not None and not result.ok:
            result.error = "连接结束但未收到 done 事件（可能被服务端提前关闭）"

    except httpx.TimeoutException:
        result.error = f"超时（>{timeout}s）"
        result.total_latency = time.monotonic() - start
    except Exception as e:
        result.error = f"{type(e).__name__}: {e}"
        result.total_latency = time.monotonic() - start

    return result


async def main():
    parser = argparse.ArgumentParser(description="模拟多用户并发聊天请求")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="后端地址")
    parser.add_argument("--users", type=int, default=5, help="并发用户数")
    parser.add_argument("--message", default="科创债ETF万家的最新净值是多少", help="每个用户发送的问题")
    parser.add_argument(
        "--same-session", action="store_true",
        help="所有用户共用同一个 session_id（用于验证会话锁是否生效，预期部分请求被拒绝）",
    )
    parser.add_argument("--timeout", type=float, default=180.0, help="单个请求超时（秒）")
    args = parser.parse_args()

    shared_session_id = str(uuid.uuid4()) if args.same_session else None

    tasks = []
    for i in range(args.users):
        username = f"{TEST_USER_PREFIX}{i}"
        session_id = shared_session_id or str(uuid.uuid4())
        tasks.append(_run_one_user(args.base_url, username, args.message, session_id, args.timeout))

    print(f"[load_test] 并发发起 {args.users} 个用户请求 -> {args.base_url}/api/chat/stream")
    if args.same_session:
        print(f"[load_test] 所有用户共用 session_id={shared_session_id}（测试会话锁）")
    wall_start = time.monotonic()

    results: list[UserResult] = await asyncio.gather(*tasks)

    wall_elapsed = time.monotonic() - wall_start

    print("\n" + "=" * 70)
    print(f"{'用户':<16}{'结果':<8}{'首字延迟(s)':<14}{'总耗时(s)':<12}备注")
    print("-" * 70)
    for r in sorted(results, key=lambda x: x.username):
        status = "OK" if r.ok else "FAIL"
        first_token = f"{r.first_token_latency:.2f}" if r.first_token_latency is not None else "-"
        total = f"{r.total_latency:.2f}" if r.total_latency is not None else "-"
        note = r.error or ""
        print(f"{r.username:<16}{status:<8}{first_token:<14}{total:<12}{note}")

    ok_count = sum(1 for r in results if r.ok)
    latencies = [r.total_latency for r in results if r.total_latency is not None]
    first_tokens = [r.first_token_latency for r in results if r.first_token_latency is not None]

    print("-" * 70)
    print(f"总墙钟耗时: {wall_elapsed:.2f}s | 成功: {ok_count}/{len(results)}")
    if latencies:
        print(f"总耗时 avg={sum(latencies)/len(latencies):.2f}s max={max(latencies):.2f}s min={min(latencies):.2f}s")
    if first_tokens:
        print(f"首字延迟 avg={sum(first_tokens)/len(first_tokens):.2f}s max={max(first_tokens):.2f}s min={min(first_tokens):.2f}s")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
