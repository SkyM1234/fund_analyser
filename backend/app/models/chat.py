"""Pydantic 模型：聊天请求/响应。"""
from typing import Literal

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    message: str = Field(..., description="本轮用户输入")
    session_id: str = Field(..., description="会话 ID（UUID，前端生成或复用）")
    history: list[ChatMessage] = Field(default_factory=list, description="历史消息（兼容旧版，优先从 checkpoint 恢复）")
