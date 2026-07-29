"""密码哈希 + JWT 编解码 + 当前用户依赖注入。"""
import uuid
from datetime import datetime, timedelta, timezone
from typing import Literal

import asyncio
import bcrypt
import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.db.mysql import get_db
from app.db.models import User
from app.db.redis import get_redis_client

_bearer = HTTPBearer(auto_error=False)

# access token 无法像 refresh token 一样单独撤销（不落库、不查 Redis 逐个校验，否则失去 JWT 免查库的意义）。
# 折中方案：记录"此用户在此时间点之前签发的 access token 全部失效"，get_current_user 用 token 的 iat 比对。
# 管理员重置密码 / 踢下线时写入这个时间戳，代价是最多有 1 次请求的竞态窗口（可接受）。
_USER_REVOKED_BEFORE_KEY_PREFIX = "revoked_before:"


async def revoke_all_access_tokens(user_id: int) -> None:
    """使某用户当前所有 access token 立即失效（配合撤销 refresh token 使用，达到真正的“踢下线”）。"""
    redis_client = get_redis_client()
    settings = get_settings()
    now_ts = datetime.now(timezone.utc).timestamp()
    ttl_seconds = int(timedelta(minutes=settings.JWT_ACCESS_TOKEN_EXPIRE_MINUTES).total_seconds())
    await redis_client.set(f"{_USER_REVOKED_BEFORE_KEY_PREFIX}{user_id}", str(now_ts), ex=ttl_seconds)


async def hash_password(password: str) -> str:
    """bcrypt 哈希是 CPU 密集型阻塞调用（约 100-300ms），丢线程池避免卡住事件循环。"""
    return await asyncio.to_thread(
        lambda: bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")
    )


async def verify_password(password: str, password_hash: str) -> bool:
    return await asyncio.to_thread(
        lambda: bcrypt.checkpw(password.encode("utf-8"), password_hash.encode("utf-8"))
    )


def _create_token(user_id: int, token_type: Literal["access", "refresh"], expires_delta: timedelta) -> str:
    settings = get_settings()
    now = datetime.now(timezone.utc)
    payload = {
        "sub": str(user_id),
        "type": token_type,
        "iat": now,
        "exp": now + expires_delta,
        # jti：同一用户同一秒内可能连续登录/刷新签发多个 token，iat+exp+sub 会完全相同，
        # 若不加随机数，refresh_token 落库时会撞 token_hash 唯一索引
        "jti": uuid.uuid4().hex,
    }
    return jwt.encode(payload, settings.JWT_SECRET_KEY, algorithm=settings.JWT_ALGORITHM)


def create_access_token(user_id: int) -> str:
    settings = get_settings()
    return _create_token(user_id, "access", timedelta(minutes=settings.JWT_ACCESS_TOKEN_EXPIRE_MINUTES))


def create_refresh_token(user_id: int) -> str:
    settings = get_settings()
    return _create_token(user_id, "refresh", timedelta(days=settings.JWT_REFRESH_TOKEN_EXPIRE_DAYS))


def decode_token(token: str, expected_type: Literal["access", "refresh"]) -> int:
    """解码并校验 token，返回 user_id；失败抛 401。"""
    settings = get_settings()
    try:
        payload = jwt.decode(token, settings.JWT_SECRET_KEY, algorithms=[settings.JWT_ALGORITHM])
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Token 已过期")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Token 无效")

    if payload.get("type") != expected_type:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Token 类型错误")

    try:
        return int(payload["sub"])
    except (KeyError, ValueError):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Token 无效")


def _decode_payload(token: str, expected_type: Literal["access", "refresh"]) -> dict:
    settings = get_settings()
    try:
        payload = jwt.decode(token, settings.JWT_SECRET_KEY, algorithms=[settings.JWT_ALGORITHM])
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Token 已过期")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Token 无效")

    if payload.get("type") != expected_type:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Token 类型错误")
    return payload


async def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer),
    db: AsyncSession = Depends(get_db),
) -> User:
    """FastAPI 依赖：从 Authorization: Bearer <token> 解析出当前登录用户。"""
    if credentials is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="缺少认证信息")

    payload = _decode_payload(credentials.credentials, expected_type="access")
    try:
        user_id = int(payload["sub"])
        issued_at = float(payload["iat"])
    except (KeyError, ValueError):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Token 无效")

    redis_client = get_redis_client()
    revoked_before = await redis_client.get(f"{_USER_REVOKED_BEFORE_KEY_PREFIX}{user_id}")
    if revoked_before is not None and issued_at < float(revoked_before):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="登录状态已失效，请重新登录")

    user = (await db.execute(select(User).where(User.id == user_id))).scalar_one_or_none()
    if user is None or not user.is_active:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="用户不存在或已停用")

    return user


async def get_current_admin(user: User = Depends(get_current_user)) -> User:
    """FastAPI 依赖：在 get_current_user 基础上要求 role=admin，否则 403。"""
    if user.role != "admin":
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="需要管理员权限")
    return user
