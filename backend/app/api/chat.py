"""SSE 流式问答接口（带 Checkpoint）。

事件类型（SSE event 字段），采用带消息生命周期的标准协议：
- message_start  新的一条助手消息开始（含首次生成与合规重试后的重新生成）；
                 前端收到后应重置当前正在渲染的消息内容，而不是追加
- token          LLM 生成的增量文本 {"delta": "..."}，追加到当前消息
- retry_notice   合规检查未通过，即将重新生成 {"reason": "..."}；
                 随后必然紧跟一个新的 message_start
- route_result   路由识别完成 {"intent": "..."}
- plan_created   Supervisor 完成规划 {"plan": [...], "reasoning": "..."}
- agent_start    子 Agent 节点开始执行 {"agent_name": "...", "task_id": "...", "description": "..."}
- agent_end      子 Agent 节点执行完毕 {"agent_name": "...", "task_id": "...", "status": "completed|failed"}
- tool_call      Agent 决定调用工具 {"name": "...", "args": {...}, "agent_name": "..."}
- tool_result    工具返回结果 {"name": "...", "output": "..."}
- retrieval_context RAG Agent 实际传给 LLM 的最终 chunk metadata
                   {"agent_name": "...", "task_id": "...", "chunks": [...]}
- tool_retry     Agent 内部触发重试 {"agent_name": "...", "task_id": "...", "attempt": 2, "reason": "..."}
- done           结束 {"finish_reason": "stop"}
- error          异常 {"message": "..."}

背景：Compliance 节点判定不通过时会回到 synthesizer 重新生成答案（见
multi_agent_controller.check_compliance）。synthesizer 每次执行都会产出一段完整的
LLM 流，若不加区分地转发，前端会把重试前后的两段内容都渲染出来。这里通过
message_start / retry_notice 显式标记消息边界，交由前端在收到 message_start 时
清空当前消息缓冲区。
"""
import json
import logging
import asyncio
import uuid
from typing import AsyncGenerator

from fastapi import APIRouter, Depends, Header, HTTPException, Request
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession
from sse_starlette.sse import EventSourceResponse

from app.core.security import get_current_user
from app.core.config import get_settings
from app.core.celery_app import celery_app
from app.db.models import ChatSession, User
from app.db.mysql import get_db
from app.models.chat import ChatRequest
from app.services.task_events import (
    publish_event,
    signal_task_cancel,
    subscribe_events,
)
from app.services.task_runs import (
    get_task_run_for_user,
    get_or_create_task_run,
    mark_submission_failed,
    request_task_cancel,
    set_celery_task_id,
)
from app.tasks.chat_tasks import run_chat_turn

logger = logging.getLogger(__name__)
router = APIRouter()


def _sse(event: str, data: dict) -> dict:
    return {"event": event, "data": json.dumps(data, ensure_ascii=False)}


async def _ensure_session_ownership(db: AsyncSession, thread_id: str, user_id: int, first_message: str) -> None:
    """确保 thread_id 在 MySQL 中有归属记录：不存在则创建，存在则校验属于当前用户。"""
    owned = (await db.execute(select(ChatSession).where(ChatSession.thread_id == thread_id))).scalar_one_or_none()
    if owned is None:
        try:
            db.add(ChatSession(user_id=user_id, thread_id=thread_id, title=first_message[:30]))
            await db.commit()
        except IntegrityError:
            # 并发下两个请求同时用同一个新 thread_id 抢先插入；回滚后重新查一次即可
            await db.rollback()
            owned = (await db.execute(select(ChatSession).where(ChatSession.thread_id == thread_id))).scalar_one_or_none()
            if owned is None or owned.user_id != user_id:
                raise HTTPException(status_code=403, detail="无权访问该会话")
    elif owned.user_id != user_id:
        raise HTTPException(status_code=403, detail="无权访问该会话")


async def _relay_from_task(run_id: str, task_id: str, request: Request) -> AsyncGenerator[dict, None]:
    """订阅 Redis 上 run_id 对应的事件流，原样转发给浏览器，直到 done/error。

    实际的 agent 执行发生在 Celery worker 进程里（app.tasks.chat_tasks），
    本生成器只负责把 worker 通过 publish_event 发出的事件转发给 SSE 连接。
    """
    logger.info(f"[chat] 开始转发任务事件: run_id={run_id}")
    try:
        async for frame in subscribe_events(run_id, task_id):
            if await request.is_disconnected():
                logger.warning(f"[chat] 客户端断开连接: run_id={run_id}")
                break
            yield _sse(frame["event"], frame["data"])
    except Exception:
        logger.exception(f"[chat] 转发任务事件异常: run_id={run_id}")
        yield _sse("error", {"message": "内部错误，请稍后重试"})
    logger.info(f"[chat] 事件转发结束: run_id={run_id}")


@router.get("/chat/tasks/{run_id}/stream")
async def resume_chat_task_stream(
    run_id: str,
    request: Request,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Reattach to an existing task after a browser reconnect."""
    task_run = await get_task_run_for_user(db, run_id=run_id, user_id=user.id)
    if task_run is None:
        raise HTTPException(status_code=404, detail="Task not found")

    # The session detail and task submission are separate database writes.
    # During that small window the browser can discover an active QUEUED run
    # before the Celery id has been persisted. Wait briefly instead of turning
    # a recoverable reconnect into a terminal frontend error.
    for _ in range(20):
        if task_run.celery_task_id:
            break
        if task_run.status not in {"QUEUED", "RUNNING", "LOST"}:
            raise HTTPException(status_code=409, detail="Task is no longer active")
        await asyncio.sleep(0.25)
        task_run = await get_task_run_for_user(db, run_id=run_id, user_id=user.id)
        if task_run is None:
            raise HTTPException(status_code=404, detail="Task not found")

    if not task_run.celery_task_id:
        raise HTTPException(status_code=409, detail="Task is still being submitted")

    return EventSourceResponse(
        _relay_from_task(run_id, task_run.celery_task_id, request)
    )


@router.post("/chat/stream")
async def chat_stream(
    req: ChatRequest,
    request: Request,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
    idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
):
    # 归属校验放在建立 SSE 流之前做，这样越权/新建会话失败会是干净的 403 响应，
    # 而不是流已经开始后再中途报错
    await _ensure_session_ownership(db, req.session_id, user.id, req.message)

    # Old clients can still submit work, but only requests with the same header
    # can be deduplicated across network retries.
    idempotency_key = idempotency_key or uuid.uuid4().hex
    if len(idempotency_key) > 128:
        raise HTTPException(status_code=400, detail="Idempotency-Key too long")

    req_payload = {
        "message": req.message,
        "session_id": req.session_id,
        "history": [{"role": h.role, "content": h.content} for h in req.history],
    }
    task_run, created = await get_or_create_task_run(
        db,
        user_id=user.id,
        session_id=req.session_id,
        idempotency_key=idempotency_key,
        request_payload=req_payload,
        max_attempts=get_settings().CELERY_TASK_MAX_ATTEMPTS,
    )
    if not created:
        if not task_run.celery_task_id:
            raise HTTPException(status_code=409, detail="Task is being submitted")
        return EventSourceResponse(
            _relay_from_task(task_run.run_id, task_run.celery_task_id, request)
        )

    run_id = task_run.run_id
    try:
        async_result = run_chat_turn.delay(run_id, req_payload, user.id)
        await set_celery_task_id(db, task_run, async_result.id)
    except Exception as exc:
        logger.exception(f"[chat] Failed to submit Celery task: run_id={run_id}")
        await mark_submission_failed(
            db,
            task_run,
            error_code="BROKER_SUBMIT_FAILED",
            error_message=str(exc),
        )
        raise HTTPException(status_code=503, detail="Task submission failed") from exc
    logger.info(
        f"[chat] 已分派任务: run_id={run_id}, task_id={async_result.id}, "
        f"session_id={req.session_id}, user_id={user.id}"
    )

    # Expose the identifiers needed by the cooperative cancellation API.
    publish_event(
        run_id,
        "started",
        {"run_id": run_id, "task_id": async_result.id},
    )
    return EventSourceResponse(_relay_from_task(run_id, async_result.id, request))


@router.post("/chat/tasks/{run_id}/cancel")
async def cancel_chat_task(
    run_id: str,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Request cooperative cancellation for one task run."""
    task_run = await request_task_cancel(
        db,
        run_id=run_id,
        user_id=user.id,
    )
    if task_run is None:
        raise HTTPException(status_code=404, detail="Task not found")

    if task_run.celery_task_id and task_run.status not in {
        "SUCCESS",
        "FAILED",
        "TIMED_OUT",
    }:
        try:
            await signal_task_cancel(run_id)
        except Exception:
            logger.exception(
                "[chat] failed to signal task cancellation: run_id=%s",
                run_id,
            )
        try:
            celery_app.control.revoke(
                task_run.celery_task_id,
                terminate=False,
            )
        except Exception:
            logger.exception(
                "[chat] failed to revoke cancelled task: run_id=%s",
                run_id,
            )

    if task_run.status == "CANCELLED":
        publish_event(
            run_id,
            "cancelled",
            {"message": "Task cancelled by user"},
        )
    status = task_run.status if task_run.status in {
        "SUCCESS",
        "FAILED",
        "CANCELLED",
        "TIMED_OUT",
    } else "CANCEL_REQUESTED"
    return {
        "run_id": task_run.run_id,
        "status": status,
        "cancel_requested": task_run.cancel_requested,
    }
