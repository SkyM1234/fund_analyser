"""会话管理接口。

归属模型：MySQL 的 sessions 表记录 thread_id -> user_id 的归属关系（及标题缓存）；
PostgreSQL 的 checkpoints 表存 LangGraph 的执行状态快照。所有接口先在 MySQL 里
校验 thread_id 是否属于当前登录用户，通过后才去 Postgres 读写，避免任何用户能
凭空猜一个 thread_id 就看到/删掉别人的会话。
"""
import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.security import get_current_user
from app.db.models import ChatSession, User
from app.db.mysql import get_db
from app.services.checkpoint import get_checkpointer

logger = logging.getLogger(__name__)
router = APIRouter()


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


async def _load_session_detail(thread_id: str) -> SessionDetail:
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

        formatted_messages = []
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

        return SessionDetail(
            thread_id=thread_id,
            messages=formatted_messages,
            checkpoint_count=len(messages),
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
    return await _load_session_detail(thread_id)


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


@router.post("/sessions/{thread_id}/rewind")
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
