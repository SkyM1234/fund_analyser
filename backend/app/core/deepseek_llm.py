"""DeepSeek 特有的推理内容兼容层。"""
from __future__ import annotations

from typing import Any, Optional, Union

import openai
from langchain_core.messages import AIMessageChunk
from langchain_core.outputs import ChatGenerationChunk, ChatResult
from langchain_openai import ChatOpenAI

from app.core.config import get_settings


class DeepSeekChatOpenAI(ChatOpenAI):
    """在 ChatOpenAI 解析响应时保留 DeepSeek 的 reasoning_content。"""

    def _get_request_payload(
        self,
        input_: Any,
        *,
        stop: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> dict:
        payload = super()._get_request_payload(input_, stop=stop, **kwargs)
        language = get_settings().LLM_THINKING_LANGUAGE.strip()
        if not language:
            return payload

        instruction = (
            f"请使用{language}进行思考。思考过程中的解释、判断和计划都使用{language}，"
            "不要因为工具调用或技术术语改用英文。"
        )
        for message in payload.get("messages", []):
            if message.get("role") == "system":
                content = message.get("content", "")
                message["content"] = f"{content}\n\n{instruction}"
                break
        else:
            payload["messages"].insert(0, {"role": "system", "content": instruction})
        return payload

    def _create_chat_result(
        self,
        response: Union[dict, openai.BaseModel],
        generation_info: Optional[dict] = None,
    ) -> ChatResult:
        result = super()._create_chat_result(response, generation_info)
        if not isinstance(response, openai.BaseModel):
            return result

        choices = getattr(response, "choices", None)
        if not choices:
            return result

        message = choices[0].message
        reasoning = getattr(message, "reasoning_content", None)
        if reasoning is None:
            model_extra = getattr(message, "model_extra", None)
            if isinstance(model_extra, dict):
                reasoning = model_extra.get("reasoning_content") or model_extra.get(
                    "reasoning"
                )

        if isinstance(reasoning, str) and reasoning.strip() and result.generations:
            result.generations[0].message.additional_kwargs["reasoning_content"] = (
                reasoning
            )
        return result

    def _convert_chunk_to_generation_chunk(
        self,
        chunk: dict,
        default_chunk_class: type,
        base_generation_info: Optional[dict],
    ) -> Optional[ChatGenerationChunk]:
        generation_chunk = super()._convert_chunk_to_generation_chunk(
            chunk,
            default_chunk_class,
            base_generation_info,
        )
        choices = chunk.get("choices")
        if not choices or generation_chunk is None:
            return generation_chunk

        message = generation_chunk.message
        if not isinstance(message, AIMessageChunk):
            return generation_chunk

        delta = choices[0].get("delta", {})
        reasoning = delta.get("reasoning_content")
        if reasoning is None:
            reasoning = delta.get("reasoning")
        if isinstance(reasoning, str):
            message.additional_kwargs["reasoning_content"] = reasoning
        return generation_chunk


def create_chat_llm(*, temperature: float | None = None) -> DeepSeekChatOpenAI:
    """创建项目 LLM，并显式开启 DeepSeek 思考模式。"""
    settings = get_settings()
    kwargs: dict[str, Any] = {
        "base_url": settings.LLM_BASE_URL,
        "api_key": settings.LLM_API_KEY,
        "model": settings.LLM_MODEL,
        "temperature": settings.LLM_TEMPERATURE if temperature is None else temperature,
    }
    if settings.LLM_THINKING_ENABLED:
        kwargs["extra_body"] = {"thinking": {"type": "enabled"}}
    return DeepSeekChatOpenAI(**kwargs)
