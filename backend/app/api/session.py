"""会话管理接口。

归属模型：MySQL 的 sessions 表记录 thread_id -> user_id 的归属关系（及标题缓存）；
PostgreSQL 的 checkpoints 表存 LangGraph 的执行状态快照。所有接口先在 MySQL 里
校验 thread_id 是否属于当前登录用户，通过后才去 Postgres 读写，避免任何用户能
凭空猜一个 thread_id 就看到/删掉别人的会话。
"""
import logging
import json
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.security import get_current_user
from app.db.models import ChatSession, User
from app.db.mysql import get_db
from app.services.checkpoint import get_checkpointer
from app.services.task_runs import get_active_task_run

logger = logging.getLogger(__name__)
router = APIRouter()


async def _load_replay_trace(run_id: str) -> list[dict[str, Any]]:
    """加载进行中的 trace 事件（最终 checkpoint 尚不可用时）。"""
    from app.db.redis import get_redis_client

    trace_types = {"agent_thought", "tool_call", "tool_result", "tool_retry"}
    events: list[dict[str, Any]] = []
    try:
        frames = await get_redis_client().lrange(f"chat:events:replay:{run_id}", 0, -1)
    except Exception:
        logger.exception("[session] failed to load replay trace: run_id=%s", run_id)
        return []

    for raw in frames:
        try:
            frame = json.loads(raw)
        except (TypeError, json.JSONDecodeError):
            continue
        if (
            not isinstance(frame, dict)
            or frame.get("event") not in trace_types
            or not isinstance(frame.get("data"), dict)
        ):
            continue
        events.append({"type": frame["event"], **frame["data"]})

    by_id: dict[str, dict[str, Any]] = {}
    for index, event in enumerate(events):
        event_id = event.get("event_id")
        key = str(event_id) if event_id else f"{event.get('type', '')}:{event.get('sequence', index)}"
        by_id[key] = event

    def sequence(event: dict[str, Any]) -> int:
        try:
            return int(event.get("sequence", 0))
        except (TypeError, ValueError):
            return 0

    return sorted(by_id.values(), key=sequence)


class SessionItem(BaseModel):
    thread_id: str
    first_message: str | None
    last_checkpoint: str | None
    checkpoint_count: int
    created_at: str | None


class SessionDetail(BaseModel):
    thread_id: str
    messages: list[dict[str, Any]]
    checkpoint_count: int
    active_task: dict[str, Any] | None = None


def _build_execution_summaries(
    channel_values: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """根据 checkpoint 状态构建可重放的执行步骤。"""
    plan_summary: list[dict[str, Any]] = []
    agent_summary: list[dict[str, Any]] = []

    fund_scope = channel_values.get("fund_scope")
    fund_scope_error = channel_values.get("fund_scope_error")
    if fund_scope is not None or fund_scope_error:
        agent_summary.append({
            "agent_name": "fund_scope_agent",
            "task_id": "fund_scope",
            "description": "确认当前问题涉及的基金范围",
            "status": "completed" if fund_scope is not None else "failed",
            # 重放时让这个非 plan 步骤排在计划中的 worker 之前。
            "sequence": -1,
        })

    current_plan = channel_values.get("plan", []) or []
    for task in current_plan:
        if not isinstance(task, dict):
            continue
        plan_summary.append({
            "task_id": task.get("task_id", ""),
            "task_type": task.get("task_type", ""),
            "description": task.get("description", ""),
            "assigned_agent": task.get("assigned_agent", ""),
            "fund_codes": task.get("fund_codes", []),
        })
        agent_summary.append({
            "agent_name": task.get("assigned_agent", ""),
            "task_id": task.get("task_id", ""),
            "description": task.get("description", ""),
            "status": task.get("status", "completed"),
        })

    return plan_summary, agent_summary


async def _get_owned_session(db: AsyncSession, thread_id: str, user_id: int) -> ChatSession:
    """校验 thread_id 属于当前用户，返回对应的 sessions 行；不存在/不属于该用户则 404。"""
    owned = (await db.execute(
        select(ChatSession).where(ChatSession.thread_id == thread_id, ChatSession.user_id == user_id)
    )).scalar_one_or_none()
    if owned is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return owned


@router.get("/sessions", response_model=list[SessionItem])
async def list_sessions(
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """列出当前用户的所有会话（按最后更新时间倒序）。"""
    owned_rows = (await db.execute(
        select(ChatSession)
        .where(ChatSession.user_id == user.id)
        .order_by(ChatSession.updated_at.desc())
    )).scalars().all()

    if not owned_rows:
        return []

    checkpointer = await get_checkpointer()
    sessions = []
    for row in owned_rows:
        thread_id = row.thread_id
        first_message = None
        last_checkpoint = None
        checkpoint_count = 0

        try:
            config = {"configurable": {"thread_id": thread_id, "checkpoint_ns": ""}}
            cp_tuple = await checkpointer.aget_tuple(config)
            if cp_tuple and cp_tuple.checkpoint:
                last_checkpoint = cp_tuple.checkpoint.get("id")
                msgs = cp_tuple.checkpoint.get("channel_values", {}).get("messages", [])
                checkpoint_count = len(msgs)
                for msg in msgs:
                    if hasattr(msg, "type") and hasattr(msg, "content"):
                        mtype, mcontent = msg.type, msg.content
                    elif isinstance(msg, dict):
                        mtype, mcontent = msg.get("type"), msg.get("content", "")
                    else:
                        continue

                    if mtype == "human" and str(mcontent).strip():
                        first_message = str(mcontent)[:30]
                        break
        except Exception as e:
            logger.warning(f"提取 {thread_id} 首条消息失败: {e}")

        sessions.append(SessionItem(
            thread_id=thread_id,
            first_message=first_message or row.title,
            last_checkpoint=last_checkpoint,
            checkpoint_count=checkpoint_count,
            created_at=row.created_at.isoformat() if row.created_at else None,
        ))

    return sessions


async def _load_session_detail(
    thread_id: str,
    db: AsyncSession | None = None,
    user_id: int | None = None,
) -> SessionDetail:
    """从 Postgres checkpoint 还原某个 thread_id 的完整消息列表，供普通用户和管理员接口共用。"""
    checkpointer = await get_checkpointer()

    try:
        config = {"configurable": {"thread_id": thread_id, "checkpoint_ns": ""}}
        checkpoint_tuple = await checkpointer.aget_tuple(config)

        if not checkpoint_tuple or not checkpoint_tuple.checkpoint:
            raise HTTPException(status_code=404, detail="Session not found")

        checkpoint = checkpoint_tuple.checkpoint
        channel_values = checkpoint.get("channel_values", {})
        messages = channel_values.get("messages", [])
        checkpoint_id = checkpoint.get("id")
        message_types = [
            (
                getattr(message, "type", None)
                or message.get("type")
                if isinstance(message, dict)
                else getattr(message, "type", type(message).__name__)
            )
            for message in messages
        ]
        logger.info(
            "[session] loaded checkpoint: thread_id=%s checkpoint_id=%s "
            "message_count=%s message_types=%s",
            thread_id,
            checkpoint_id,
            len(messages),
            message_types,
        )
        if messages and not any(message_type == "ai" for message_type in message_types):
            logger.warning(
                "[session] checkpoint has no AIMessage yet; task state will determine "
                "whether frontend should create a pending assistant: "
                "thread_id=%s checkpoint_id=%s",
                thread_id,
                checkpoint_id,
            )
        tool_call_log = channel_values.get("tool_call_log", []) or []
        trace_events_by_run = channel_values.get("trace_events", {}) or {}

        formatted_messages = []
        tool_log_index = 0
        for msg in messages:
            if hasattr(msg, "type") and hasattr(msg, "content"):
                role = msg.type
                content = msg.content
                tool_calls = getattr(msg, "tool_calls", None) or []
            elif isinstance(msg, dict):
                role = msg.get("type")
                content = msg.get("content", "")
                tool_calls = msg.get("tool_calls", [])
            else:
                continue

            if role == "human":
                formatted_messages.append({
                    "role": "user",
                    "content": str(content),
                    "tools": []
                })
            elif role == "ai":
                tools = []
                if tool_calls:
                    for tc in tool_calls:
                        if isinstance(tc, dict):
                            tools.append({
                                "name": tc.get("name", ""),
                                "args": tc.get("args", {}),
                                "tool_call_id": tc.get("id", ""),
                            })
                        elif hasattr(tc, "get"):
                            tools.append({
                                "name": tc.get("name", ""),
                                "args": tc.get("args", {}),
                                "tool_call_id": tc.get("id", ""),
                            })

                for tool in tools:
                    while tool_log_index < len(tool_call_log):
                        log_item = tool_call_log[tool_log_index]
                        tool_log_index += 1
                        if log_item.get("name") != tool.get("name"):
                            continue
                        tool["agent_name"] = log_item.get("agent", "")
                        tool["task_id"] = log_item.get("task_id", "")
                        break

                if str(content).strip() or tools:
                    formatted_messages.append({
                        "role": "assistant",
                        "content": str(content),
                        "tools": tools
                    })
            elif role == "tool":
                if formatted_messages and formatted_messages[-1]["role"] == "assistant":
                    tool_name = msg.name if hasattr(msg, "name") else msg.get("name", "") if isinstance(msg, dict) else ""
                    tool_call_id = (
                        msg.tool_call_id
                        if hasattr(msg, "tool_call_id")
                        else msg.get("tool_call_id", "")
                        if isinstance(msg, dict)
                        else ""
                    )
                    output = str(content)
                    for tool in formatted_messages[-1]["tools"]:
                        if (
                            tool_call_id
                            and tool.get("tool_call_id") == tool_call_id
                        ) or (
                            not tool_call_id
                            and tool["name"] == tool_name
                            and "output" not in tool
                        ):
                            tool["output"] = output
                            break

        # trace_events 按用户轮次与最终 assistant 消息配对。旧 checkpoint 没有该字段，
        # 保持既有的工具消息回放路径。
        trace_runs = (
            list(trace_events_by_run.values())
            if isinstance(trace_events_by_run, dict)
            else []
        )
        user_turns: list[int] = [
            index
            for index, message in enumerate(formatted_messages)
            if message["role"] == "user"
        ]
        for turn_index, user_index in enumerate(user_turns):
            if turn_index >= len(trace_runs):
                break
            next_user_index = (
                user_turns[turn_index + 1]
                if turn_index + 1 < len(user_turns)
                else len(formatted_messages)
            )
            assistant_indexes = [
                index
                for index in range(user_index + 1, next_user_index)
                if formatted_messages[index]["role"] == "assistant"
            ]
            if assistant_indexes:
                formatted_messages[assistant_indexes[-1]]["trace_events"] = trace_runs[turn_index]

        active_task = None
        if db is not None and user_id is not None:
            task = await get_active_task_run(
                db,
                session_id=thread_id,
                user_id=user_id,
            )
            if task is not None:
                active_task = {
                    "run_id": task.run_id,
                    "task_id": task.celery_task_id,
                    "status": task.status,
                    "attempt": task.attempt,
                }
                replay_trace = await _load_replay_trace(task.run_id)
                if replay_trace:
                    if trace_runs:
                        trace_runs[-1] = replay_trace
                    else:
                        trace_runs = [replay_trace]

                    last_assistant = next(
                        (
                            message
                            for message in reversed(formatted_messages)
                            if message["role"] == "assistant"
                        ),
                        None,
                    )
                    if last_assistant is not None:
                        last_assistant["trace_events"] = replay_trace

        # checkpoint 可能只包含用户消息，此时第一条 AI 消息或 trace 事件
        # 尚未持久化。根据持久的任务状态创建 pending 的 assistant，
        # 这样重连时始终有一个事件接收端。
        if (
            (active_task is not None or trace_runs)
            and formatted_messages
            and formatted_messages[-1]["role"] == "user"
        ):
            formatted_messages.append({
                "role": "assistant",
                "content": "",
                "tools": [],
                "trace_events": trace_runs[-1] if trace_runs else [],
                "agents": [],
                "plan": [],
                "pending": True,
            })

        plan_summary, agent_summary = _build_execution_summaries(channel_values)
        if formatted_messages and (plan_summary or agent_summary):
            last_assistant = next(
                (
                    message
                    for message in reversed(formatted_messages)
                    if message["role"] == "assistant"
                ),
                None,
            )
            if last_assistant is not None:
                if plan_summary:
                    last_assistant["plan"] = plan_summary
                if agent_summary:
                    last_assistant["agents"] = agent_summary

        return SessionDetail(
            thread_id=thread_id,
            messages=formatted_messages,
            checkpoint_count=len(messages),
            active_task=active_task,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("get session failed")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/sessions/{thread_id}", response_model=SessionDetail)
async def get_session(
    thread_id: str,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """获取某个会话的所有历史消息（仅当前用户拥有的会话）。"""
    await _get_owned_session(db, thread_id, user.id)
    return await _load_session_detail(thread_id, db, user.id)


@router.delete("/sessions/{thread_id}")
async def delete_session(
    thread_id: str,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """删除某个会话（仅当前用户拥有的会话）：同时清理 MySQL 归属记录和 Postgres checkpoint。"""
    owned = await _get_owned_session(db, thread_id, user.id)
    checkpointer = await get_checkpointer()
    pool = checkpointer.conn

    try:
        async with pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute("DELETE FROM checkpoints WHERE thread_id = %s", (thread_id,))
                deleted_count = cur.rowcount
                await cur.execute("DELETE FROM checkpoint_blobs WHERE thread_id = %s", (thread_id,))
                await cur.execute("DELETE FROM checkpoint_writes WHERE thread_id = %s", (thread_id,))

        await db.delete(owned)
        await db.commit()

        return {"deleted": deleted_count, "thread_id": thread_id}
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("delete session failed")
        raise HTTPException(status_code=500, detail=str(e))


class RewindRequest(BaseModel):
    message_index: int = Field(..., description="要回溯到的消息索引（保留到该索引之前的消息）")


def _checkpoint_message_type(message: Any) -> str | None:
    if hasattr(message, "type"):
        return message.type
    if isinstance(message, dict):
        return message.get("type")
    return None


def _checkpoint_messages(checkpoint: dict[str, Any]) -> list[Any]:
    return checkpoint.get("channel_values", {}).get("messages", []) or []


async def _list_checkpoint_history(checkpointer: Any, thread_id: str) -> list[Any]:
    config = {"configurable": {"thread_id": thread_id, "checkpoint_ns": ""}}
    return [item async for item in checkpointer.alist(config)]


async def _clear_thread_checkpoints(checkpointer: Any, thread_id: str) -> int:
    pool = checkpointer.conn
    async with pool.connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute("DELETE FROM checkpoints WHERE thread_id = %s", (thread_id,))
            deleted_count = cur.rowcount
            await cur.execute("DELETE FROM checkpoint_blobs WHERE thread_id = %s", (thread_id,))
            await cur.execute("DELETE FROM checkpoint_writes WHERE thread_id = %s", (thread_id,))
    return deleted_count


def _select_rewind_checkpoint(
    history: list[Any],
    target_user_count: int,
) -> Any | None:
    """返回所选用户轮次之前的最新状态。"""
    for item in history:
        checkpoint = getattr(item, "checkpoint", None)
        if not checkpoint:
            continue
        messages = _checkpoint_messages(checkpoint)
        user_count = sum(
            _checkpoint_message_type(message) == "human"
            for message in messages
        )
        has_assistant = any(
            _checkpoint_message_type(message) == "ai"
            for message in messages
        )
        if user_count == target_user_count and (
            target_user_count == 0 or has_assistant
        ):
            return item
    return None


@router.post("/sessions/{thread_id}/rewind")
async def rewind_session(
    thread_id: str,
    req: RewindRequest,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """从 LangGraph checkpoint 历史回退，不信任前端历史。"""
    await _get_owned_session(db, thread_id, user.id)
    checkpointer = await get_checkpointer()

    try:
        if req.message_index < 0:
            raise HTTPException(status_code=400, detail="Invalid message index")

        history = await _list_checkpoint_history(checkpointer, thread_id)
        latest_checkpoint = next(
            (item.checkpoint for item in history if getattr(item, "checkpoint", None)),
            None,
        )
        if latest_checkpoint is None:
            raise HTTPException(status_code=404, detail="Session checkpoint not found")

        latest_messages = _checkpoint_messages(latest_checkpoint)
        user_positions = [
            index
            for index, message in enumerate(latest_messages)
            if _checkpoint_message_type(message) == "human"
        ]
        if req.message_index >= len(user_positions):
            raise HTTPException(status_code=400, detail="Message index is not a user message")

        target_position = user_positions[req.message_index]
        target_user_count = sum(
            _checkpoint_message_type(message) == "human"
            for message in latest_messages[:target_position]
        )
        selected_item = _select_rewind_checkpoint(history, target_user_count)

        deleted_count = await _clear_thread_checkpoints(checkpointer, thread_id)
        if selected_item is not None:
            restore_config = {
                "configurable": {
                    "thread_id": thread_id,
                    "checkpoint_ns": "",
                }
            }
            await checkpointer.aput(
                restore_config,
                selected_item.checkpoint,
                selected_item.metadata,
                selected_item.checkpoint.get("channel_versions", {}),
            )

        logger.info(
            "Rewound session %s: deleted=%s target_message_index=%s restored=%s",
            thread_id,
            deleted_count,
            req.message_index,
            selected_item is not None,
        )
        return {
            "thread_id": thread_id,
            "message_index": req.message_index,
            "deleted_checkpoints": deleted_count,
            "restored": selected_item is not None,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("rewind session failed")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/sessions/{thread_id}/rewind/legacy")
async def rewind_session(
    thread_id: str,
    req: RewindRequest,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """回溯会话到指定消息位置，删除所有 checkpoint，由前端负责保持截断后的消息。"""
    await _get_owned_session(db, thread_id, user.id)
    checkpointer = await get_checkpointer()
    pool = checkpointer.conn

    try:
        async with pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "DELETE FROM checkpoints WHERE thread_id = %s AND checkpoint_ns = ''",
                    (thread_id,)
                )
                deleted_count = cur.rowcount
                await cur.execute(
                    "DELETE FROM checkpoint_blobs WHERE thread_id = %s",
                    (thread_id,)
                )
                await cur.execute(
                    "DELETE FROM checkpoint_writes WHERE thread_id = %s",
                    (thread_id,)
                )

        logger.info(f"回溯会话 {thread_id}: 删除了 {deleted_count} 个 checkpoint，消息截断到索引 {req.message_index}")

        return {
            "thread_id": thread_id,
            "message_index": req.message_index,
            "deleted_checkpoints": deleted_count
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("rewind session failed")
        raise HTTPException(status_code=500, detail=str(e))
