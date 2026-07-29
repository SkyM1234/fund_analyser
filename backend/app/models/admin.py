"""Pydantic 模型：管理员接口请求/响应。"""
import datetime

from pydantic import BaseModel, Field


class AdminUserItem(BaseModel):
    id: int
    username: str
    email: str | None
    role: str
    is_active: bool
    created_at: datetime.datetime
    session_count: int


class AdminSetActiveRequest(BaseModel):
    is_active: bool


class AdminResetPasswordRequest(BaseModel):
    new_password: str = Field(..., min_length=6, max_length=128)


class AdminSessionItem(BaseModel):
    thread_id: str
    user_id: int
    username: str
    title: str | None
    created_at: datetime.datetime
    updated_at: datetime.datetime


class AdminDeleteUserResponse(BaseModel):
    deleted_user_id: int
    deleted_username: str


class AdminTokenItem(BaseModel):
    id: int
    expires_at: datetime.datetime
    created_at: datetime.datetime
    revoked: bool
