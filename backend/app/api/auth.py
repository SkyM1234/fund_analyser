"""用户注册 / 登录 / token 刷新 / 登出接口。"""
import hashlib
import logging
from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.core.security import (
    create_access_token,
    create_refresh_token,
    decode_token,
    get_current_user,
    hash_password,
    verify_password,
)
from app.db.mysql import get_db
from app.db.models import RefreshToken, User
from app.db.redis import get_redis_client
from app.models.auth import LoginRequest, RefreshRequest, RegisterRequest, TokenResponse, UserResponse

logger = logging.getLogger(__name__)
router = APIRouter()

# refresh token 有效性判断走 Redis（O(1) EXISTS，取代原来的 MySQL SELECT）；
# MySQL 中的 RefreshToken 行仍然写入，仅作为审计/持久化留痕，不再参与鉴权热路径判断。
_REFRESH_TOKEN_KEY_PREFIX = "refresh:"


def _hash_token(token: str) -> str:
    """刷新令牌落库前做单向哈希（sha256，可精确匹配查重/撤销，无需 bcrypt 的加盐比较）。"""
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def _utcnow_naive() -> datetime:
    """DB 的 DateTime 列不带时区，统一存/比 naive UTC，避免跟 tz-aware 时间比较时报错。"""
    return datetime.now(timezone.utc).replace(tzinfo=None)


async def _issue_tokens(user_id: int, db: AsyncSession) -> TokenResponse:
    access_token = create_access_token(user_id)
    refresh_token = create_refresh_token(user_id)
    token_hash = _hash_token(refresh_token)

    settings = get_settings()
    ttl_seconds = int(timedelta(days=settings.JWT_REFRESH_TOKEN_EXPIRE_DAYS).total_seconds())

    db.add(RefreshToken(
        user_id=user_id,
        token_hash=token_hash,
        expires_at=_utcnow_naive() + timedelta(days=settings.JWT_REFRESH_TOKEN_EXPIRE_DAYS),
    ))
    await db.commit()

    # 鉴权热路径的有效性标记，TTL 与令牌到期时间一致，到期自动失效
    redis_client = get_redis_client()
    await redis_client.set(f"{_REFRESH_TOKEN_KEY_PREFIX}{token_hash}", "valid", ex=ttl_seconds)

    return TokenResponse(access_token=access_token, refresh_token=refresh_token)


@router.post("/auth/register", status_code=status.HTTP_201_CREATED)
async def register(req: RegisterRequest, db: AsyncSession = Depends(get_db)):
    existing = (await db.execute(select(User).where(User.username == req.username))).scalar_one_or_none()
    if existing is not None:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="用户名已存在")

    user = User(username=req.username, email=req.email, password_hash=await hash_password(req.password))
    db.add(user)
    await db.commit()

    logger.info(f"[auth] 新用户注册: username={req.username}, user_id={user.id}")
    return {"message": "注册成功，请登录"}


@router.post("/auth/login", response_model=TokenResponse)
async def login(req: LoginRequest, db: AsyncSession = Depends(get_db)):
    user = (await db.execute(select(User).where(User.username == req.username))).scalar_one_or_none()
    if user is None or not await verify_password(req.password, user.password_hash):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="用户名或密码错误")
    if not user.is_active:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="账号已停用")

    return await _issue_tokens(user.id, db)


@router.post("/auth/refresh", response_model=TokenResponse)
async def refresh(req: RefreshRequest, db: AsyncSession = Depends(get_db)):
    user_id = decode_token(req.refresh_token, expected_type="refresh")
    token_hash = _hash_token(req.refresh_token)

    redis_client = get_redis_client()
    valid = await redis_client.exists(f"{_REFRESH_TOKEN_KEY_PREFIX}{token_hash}")
    if not valid:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="刷新令牌无效或已过期")

    # 检查账号是否仍为活跃状态（停用/删除后 refresh 也应失效）
    user = (await db.execute(select(User).where(User.id == user_id))).scalar_one_or_none()
    if user is None or not user.is_active:
        await redis_client.delete(f"{_REFRESH_TOKEN_KEY_PREFIX}{token_hash}")
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="账号已停用或不存在")

    # 旋转：撤销旧的刷新令牌（Redis 热路径 + MySQL 审计留痕），签发新的一对（防止刷新令牌被重放）
    await redis_client.delete(f"{_REFRESH_TOKEN_KEY_PREFIX}{token_hash}")
    stored = (await db.execute(
        select(RefreshToken).where(RefreshToken.token_hash == token_hash)
    )).scalar_one_or_none()
    if stored is not None:
        stored.revoked = True

    tokens = await _issue_tokens(user_id, db)
    return tokens


@router.post("/auth/logout", status_code=status.HTTP_204_NO_CONTENT)
async def logout(req: RefreshRequest, db: AsyncSession = Depends(get_db)):
    token_hash = _hash_token(req.refresh_token)

    redis_client = get_redis_client()
    await redis_client.delete(f"{_REFRESH_TOKEN_KEY_PREFIX}{token_hash}")

    stored = (await db.execute(
        select(RefreshToken).where(RefreshToken.token_hash == token_hash)
    )).scalar_one_or_none()
    if stored is not None:
        stored.revoked = True
        await db.commit()


@router.get("/auth/me", response_model=UserResponse)
async def me(user: User = Depends(get_current_user)):
    return UserResponse(id=user.id, username=user.username, email=user.email, role=user.role)
