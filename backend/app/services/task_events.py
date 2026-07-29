"""Celery worker → FastAPI 的 SSE 事件桥接（Redis Pub/Sub + 回放缓冲）。

Worker 进程执行完整的 agent 图运行，把每个 SSE 形状的事件发布到以
run_id 为 key 的 Redis channel；FastAPI 侧订阅该 channel，将收到的帧原样
转发给浏览器的 EventSourceResponse，从而保持前端 SSE 协议逐字节不变。

同时把事件写入一个有序、有上限的 List 作为短期回放缓冲：FastAPI 订阅
pubsub 存在时间窗口（订阅生效前可能已有事件发出），订阅前先读取回放缓冲
补齐，再切换到 pubsub 实时监听，避免开头事件丢失。回放 List 设置 TTL，
用完即过期，不会无限堆积。
"""
import asyncio
import json
import logging
import time
from typing import Any, AsyncGenerator

from celery.result import AsyncResult

from app.core.celery_app import celery_app
from app.db.redis import get_redis_client

logger = logging.getLogger(__name__)

# 任务已提交但尚未被 worker 取走执行，或已被取走但还没跑完；这些状态下
# 排队等待是正常情况，不能据此判断任务已经"死了"
_ALIVE_STATES = {"PENDING", "STARTED", "RETRY"}

_REPLAY_MAX_LEN = 500  # 单次对话正常不会超过几十个事件，留足余量
_REPLAY_TTL_SECONDS = 600  # 回放缓冲存活时间，覆盖客户端断线重连的合理窗口
_DONE_EVENTS = {"done", "error"}  # 收到即视为流结束，两端都据此停止


def _channel(run_id: str) -> str:
    return f"chat:events:{run_id}"


def _replay_key(run_id: str) -> str:
    return f"chat:events:replay:{run_id}"


def publish_event(run_id: str, event: str, data: dict) -> None:
    """Worker 侧调用：同步 Redis 客户端发布一个 SSE 形状的事件。

    Celery 任务运行在 worker 的后台事件循环线程中（见 worker_lifecycle），
    但 publish 本身是极短的同步 Redis 操作，用同步 redis-py 客户端更简单，
    不需要额外经过 run_coroutine_threadsafe。
    """
    import redis as sync_redis

    from app.core.config import get_settings

    payload = json.dumps({"event": event, "data": data}, ensure_ascii=False)

    client = sync_redis.Redis.from_url(get_settings().REDIS_URL, decode_responses=True)
    try:
        pipe = client.pipeline()
        pipe.rpush(_replay_key(run_id), payload)
        pipe.expire(_replay_key(run_id), _REPLAY_TTL_SECONDS)
        pipe.publish(_channel(run_id), payload)
        pipe.execute()
    finally:
        client.close()


async def subscribe_events(run_id: str, task_id: str, poll_timeout: float = 30.0) -> AsyncGenerator[dict, None]:
    """FastAPI 侧调用：异步生成器，产出 {"event", "data"} 字典，直到 done/error。

    先读回放缓冲里已有的事件（覆盖"任务先跑起来、FastAPI 还没订阅上"的窗口），
    再进入 pubsub 实时监听；对每条消息都做去重（按回放缓冲长度）以避免重复
    转发同一事件。poll_timeout 秒内没有任何新事件时，用 Celery 的 AsyncResult
    查一次任务真实状态：任务还在排队/执行中（PENDING/STARTED/RETRY）就继续等，
    只有任务已经不再运行（SUCCESS/FAILURE/REVOKED 等终态）却没发布 done/error
    （worker 异常退出）时才提前结束，避免客户端连接永久挂起。

    task_id 是 Celery 任务自身的 ID（`AsyncResult.id`），与 run_id（Redis
    channel/回放缓冲的标识）是两个独立的概念：run_id 早在任务被 worker 取走
    执行之前就已生成，此时 Redis 上还没有任何该 run_id 的记录，不能拿"Redis
    key 是否存在"来推断任务是否已结束——排队中和已结束在 Redis 层面看起来
    完全一样，只有问 Celery 本身才能区分（--pool=solo 单并发下，任务排队
    超过 poll_timeout 是正常情况，见并发压测暴露的问题）。
    """
    redis_client = get_redis_client()
    replay_key = _replay_key(run_id)
    channel = _channel(run_id)

    delivered = 0

    async def _emit_raw(raw: str) -> dict | None:
        nonlocal delivered
        delivered += 1
        try:
            frame = json.loads(raw)
        except json.JSONDecodeError:
            logger.error(f"[task_events] 无法解析事件 payload: run_id={run_id}")
            return None
        return frame

    # 1) 补齐订阅生效前已发出的事件
    backlog = await redis_client.lrange(replay_key, 0, -1)
    for raw in backlog:
        frame = await _emit_raw(raw)
        if frame is not None:
            yield frame
            if frame.get("event") in _DONE_EVENTS:
                return

    # 2) 切到 pubsub 实时监听剩余事件
    pubsub = redis_client.pubsub()
    await pubsub.subscribe(channel)
    # subscribe() 不会读走 SUBSCRIBE 命令的确认帧（redis-py 故意留到 get_message
    # 里再读，避免吞掉已经到达的正常消息），所以订阅后第一次 get_message 几乎总是
    # 立刻返回 None（读到的是确认帧，被 ignore_subscribe_messages 过滤掉），而不是
    # 真的等了 poll_timeout 秒。这里用 last_activity 记录真实的最后一次活动时间，
    # 只有真实空闲达到 poll_timeout 才做存活性判断，避免刚订阅上就被误判为任务已死。
    last_activity = time.monotonic()
    try:
        while True:
            remaining = poll_timeout - (time.monotonic() - last_activity)
            if remaining <= 0:
                # 真实空闲已达到 poll_timeout：直接问 Celery 任务是否已经不再运行。
                # 排队中（PENDING）或执行中（STARTED/RETRY）都继续等——在 --pool=solo
                # 单并发下，前面还有任务在跑时排队超过 poll_timeout 是完全正常的，
                # 不能当成任务已死。只有终态（SUCCESS/FAILURE/REVOKED 等）却没发布
                # done/error（worker 异常退出、没走到 finally 发布 error 之前就被杀）
                # 才提前结束，避免客户端连接永久挂起。
                result = AsyncResult(task_id, app=celery_app)
                if result.state not in _ALIVE_STATES:
                    logger.warning(
                        f"[task_events] 任务已处于终态但未发布 done/error: "
                        f"run_id={run_id}, task_id={task_id}, state={result.state}"
                    )
                    return
                last_activity = time.monotonic()
                continue

            message = await pubsub.get_message(ignore_subscribe_messages=True, timeout=remaining)
            if message is None:
                continue
            last_activity = time.monotonic()

            raw = message["data"]
            frame = await _emit_raw(raw)
            if frame is not None:
                yield frame
                if frame.get("event") in _DONE_EVENTS:
                    return
    finally:
        await pubsub.unsubscribe(channel)
        await pubsub.aclose()
