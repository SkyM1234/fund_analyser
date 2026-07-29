"""全局 LLM 调用并发上限。

多用户并发聊天时，11 处 ChatOpenAI 调用点（各 agent 节点 + 路由分类器）互相独立，
没有任何调用节流。高并发下可能瞬间打出远超上游（DeepSeek）速率限制的请求量，
导致连锁 429。这里提供一个进程级共享信号量，所有 .ainvoke() 调用统一从这里过一遍。

注意：这是单进程内的软上限，多 worker 部署时每个进程各自持有一个信号量，
总并发上限是 CONCURRENCY × workers 数，需要按实际部署的 worker 数相应调低。
"""
import asyncio

from app.core.config import get_settings

_semaphore: asyncio.Semaphore | None = None


def get_llm_semaphore() -> asyncio.Semaphore:
    global _semaphore
    if _semaphore is None:
        settings = get_settings()
        _semaphore = asyncio.Semaphore(settings.LLM_MAX_CONCURRENCY)
    return _semaphore


async def llm_ainvoke(llm, messages):
    """包装 ChatOpenAI.ainvoke，统一走全局并发信号量。"""
    async with get_llm_semaphore():
        return await llm.ainvoke(messages)
