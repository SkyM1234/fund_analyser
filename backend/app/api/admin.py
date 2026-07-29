"""管理员接口：用户管理 / 会话管理 / token 管理。

所有接口都挂在 get_current_admin 依赖之下（role != "admin" 时 403）。
会话相关接口直接按 thread_id 操作，不做归属校验（管理员可访问任意用户的会话）。
"""
import logging
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.security import get_current_admin, hash_password, revoke_all_access_tokens
from app.db.models import ChatSession, RefreshToken, User
from app.db.mysql import get_db
from app.db.redis import get_redis_client
from app.models.admin import (
    AdminDeleteUserResponse,
    AdminResetPasswordRequest,
    AdminSessionItem,
    AdminSetActiveRequest,
    AdminTokenItem,
    AdminUserItem,
)
from app.api.session import SessionDetail, _load_session_detail
from app.services.checkpoint import get_checkpointer

logger = logging.getLogger(__name__)
router = APIRouter()

_REFRESH_TOKEN_KEY_PREFIX = "refresh:"


@router.get("/admin/users", response_model=list[AdminUserItem])
async def list_users(
    admin: User = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db),
):
    """列出所有用户及其会话数量。"""
    rows = (await db.execute(
        select(User, func.count(ChatSession.id))
        .outerjoin(ChatSession, ChatSession.user_id == User.id)
        .group_by(User.id)
        .order_by(User.id)
    )).all()

    return [
        AdminUserItem(
            id=user.id,
            username=user.username,
            email=user.email,
            role=user.role,
            is_active=user.is_active,
            created_at=user.created_at,
            session_count=session_count,
        )
        for user, session_count in rows
    ]


async def _get_user_or_404(db: AsyncSession, user_id: int) -> User:
    user = (await db.execute(select(User).where(User.id == user_id))).scalar_one_or_none()
    if user is None:
        raise HTTPException(status_code=404, detail="用户不存在")
    return user


@router.patch("/admin/users/{user_id}/active", response_model=AdminUserItem)
async def set_user_active(
    user_id: int,
    req: AdminSetActiveRequest,
    admin: User = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db),
):
    """启用/停用某个用户账号。停用管理员自己会被拒绝，避免误操作把自己锁出。停用时撤销所有 token 强制下线。"""
    if user_id == admin.id and not req.is_active:
        raise HTTPException(status_code=400, detail="不能停用自己的账号")

    user = await _get_user_or_404(db, user_id)
    user.is_active = req.is_active

    if not req.is_active:
        # 停用时撤销所有 token，强制该用户立即下线
        tokens = (await db.execute(
            select(RefreshToken).where(RefreshToken.user_id == user_id, RefreshToken.revoked == False)  # noqa: E712
        )).scalars().all()
        redis_client = get_redis_client()
        for t in tokens:
            t.revoked = True
            await redis_client.delete(f"{_REFRESH_TOKEN_KEY_PREFIX}{t.token_hash}")
        await revoke_all_access_tokens(user_id)
        logger.info(f"[admin] 停用用户 user_id={user_id}，已撤销 {len(tokens)} 个 refresh token")

    await db.commit()

    session_count = (await db.execute(
        select(func.count(ChatSession.id)).where(ChatSession.user_id == user_id)
    )).scalar_one()

    logger.info(f"[admin] user_id={user_id} is_active={req.is_active} (操作者 admin_id={admin.id})")
    return AdminUserItem(
        id=user.id, username=user.username, email=user.email, role=user.role,
        is_active=user.is_active, created_at=user.created_at, session_count=session_count,
    )


@router.delete("/admin/users/{user_id}", response_model=AdminDeleteUserResponse)
async def delete_user(
    user_id: int,
    admin: User = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db),
):
    """删除用户及其所有关联数据（会话、refresh token、checkpoint）。管理员不能删除自己。"""
    if user_id == admin.id:
        raise HTTPException(status_code=400, detail="不能删除自己的账号")

    user = await _get_user_or_404(db, user_id)
    deleted_username = user.username

    # 撤销该用户所有 access token（Redis），使其立即失效
    await revoke_all_access_tokens(user_id)

    # 清除所有 refresh token 的 Redis 有效性标记
    tokens = (await db.execute(
        select(RefreshToken).where(RefreshToken.user_id == user_id)
    )).scalars().all()
    redis_client = get_redis_client()
    for t in tokens:
        await redis_client.delete(f"{_REFRESH_TOKEN_KEY_PREFIX}{t.token_hash}")

    # 清理该用户所有会话的 Postgres checkpoint 数据
    sessions = (await db.execute(
        select(ChatSession.thread_id).where(ChatSession.user_id == user_id)
    )).scalars().all()
    thread_ids = list(sessions)
    if thread_ids:
        try:
            checkpointer = await get_checkpointer()
            pool = checkpointer.conn
            async with pool.connection() as conn:
                async with conn.cursor() as cur:
                    for tid in thread_ids:
                        await cur.execute("DELETE FROM checkpoints WHERE thread_id = %s", (tid,))
                        await cur.execute("DELETE FROM checkpoint_blobs WHERE thread_id = %s", (tid,))
                        await cur.execute("DELETE FROM checkpoint_writes WHERE thread_id = %s", (tid,))
        except Exception as e:
            logger.exception(f"[admin] 删除用户 {user_id} 的 checkpoint 数据时出错: {e}")
            raise HTTPException(status_code=500, detail="清理会话数据失败，请重试")

    # 删除用户（MySQL cascade 会自动清理 sessions 和 refresh_tokens 行）
    await db.delete(user)
    await db.commit()

    logger.info(f"[admin] 删除用户 user_id={user_id} username={deleted_username}，清理了 {len(thread_ids)} 个会话 (操作者 admin_id={admin.id})")
    return AdminDeleteUserResponse(deleted_user_id=user_id, deleted_username=deleted_username)


@router.post("/admin/users/{user_id}/reset-password", status_code=204)
async def reset_password(
    user_id: int,
    req: AdminResetPasswordRequest,
    admin: User = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db),
):
    """重置某个用户的密码，并撤销其所有 refresh token（强制重新登录）。"""
    user = await _get_user_or_404(db, user_id)
    user.password_hash = await hash_password(req.new_password)

    tokens = (await db.execute(
        select(RefreshToken).where(RefreshToken.user_id == user_id, RefreshToken.revoked == False)  # noqa: E712
    )).scalars().all()
    redis_client = get_redis_client()
    for t in tokens:
        t.revoked = True
        await redis_client.delete(f"{_REFRESH_TOKEN_KEY_PREFIX}{t.token_hash}")

    await db.commit()
    logger.info(f"[admin] 重置密码 user_id={user_id} 并撤销 {len(tokens)} 个 refresh token (操作者 admin_id={admin.id})")
    await revoke_all_access_tokens(user_id)


@router.get("/admin/users/{user_id}/tokens", response_model=list[AdminTokenItem])
async def list_user_tokens(
    user_id: int,
    admin: User = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db),
):
    """列出某个用户当前未过期的 refresh token（即登录设备数）。"""
    await _get_user_or_404(db, user_id)
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    rows = (await db.execute(
        select(RefreshToken)
        .where(RefreshToken.user_id == user_id, RefreshToken.expires_at > now, RefreshToken.revoked == False)  # noqa: E712
        .order_by(RefreshToken.created_at.desc())
    )).scalars().all()
    return [
        AdminTokenItem(id=r.id, expires_at=r.expires_at, created_at=r.created_at, revoked=r.revoked)
        for r in rows
    ]


@router.delete("/admin/tokens/{token_id}", status_code=204)
async def revoke_token(
    token_id: int,
    admin: User = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db),
):
    """撤销单个 refresh token（踢下线）。"""
    token = (await db.execute(
        select(RefreshToken).where(RefreshToken.id == token_id)
    )).scalar_one_or_none()
    if token is None:
        raise HTTPException(status_code=404, detail="Token 不存在")

    token.revoked = True
    redis_client = get_redis_client()
    await redis_client.delete(f"{_REFRESH_TOKEN_KEY_PREFIX}{token.token_hash}")
    await db.commit()
    logger.info(f"[admin] 撤销 token_id={token_id} user_id={token.user_id} (操作者 admin_id={admin.id})")
    await revoke_all_access_tokens(token.user_id)


@router.get("/admin/sessions", response_model=list[AdminSessionItem])
async def list_all_sessions(
    admin: User = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db),
):
    """列出所有用户的所有会话（按最后更新时间倒序）。"""
    rows = (await db.execute(
        select(ChatSession, User.username)
        .join(User, User.id == ChatSession.user_id)
        .order_by(ChatSession.updated_at.desc())
    )).all()
    return [
        AdminSessionItem(
            thread_id=cs.thread_id, user_id=cs.user_id, username=username,
            title=cs.title, created_at=cs.created_at, updated_at=cs.updated_at,
        )
        for cs, username in rows
    ]


@router.get("/admin/sessions/{thread_id}", response_model=SessionDetail)
async def get_any_session(
    thread_id: str,
    admin: User = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db),
):
    """获取任意用户会话的完整消息（不校验归属）。"""
    exists = (await db.execute(
        select(ChatSession).where(ChatSession.thread_id == thread_id)
    )).scalar_one_or_none()
    if exists is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return await _load_session_detail(thread_id)


@router.delete("/admin/sessions/{thread_id}")
async def delete_any_session(
    thread_id: str,
    admin: User = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db),
):
    """删除任意用户的会话（不校验归属）：同时清理 MySQL 归属记录和 Postgres checkpoint。"""
    owned = (await db.execute(
        select(ChatSession).where(ChatSession.thread_id == thread_id)
    )).scalar_one_or_none()
    if owned is None:
        raise HTTPException(status_code=404, detail="Session not found")

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
        logger.info(f"[admin] 删除会话 thread_id={thread_id} user_id={owned.user_id} (操作者 admin_id={admin.id})")
        return {"deleted": deleted_count, "thread_id": thread_id}
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("admin delete session failed")
        raise HTTPException(status_code=500, detail=str(e))
