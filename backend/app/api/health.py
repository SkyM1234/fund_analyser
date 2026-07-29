"""健康检查：同时探测 GPU 服务和 LLM 配置是否就位。"""
from fastapi import APIRouter

from app.core.config import get_settings
from app.services.rag_client_mcp import get_rag_client

router = APIRouter()


@router.get("/health")
async def health():
    s = get_settings()
    result = {
        "app": "ok",
        "llm_configured": bool(s.LLM_API_KEY),
        "llm_model": s.LLM_MODEL,
        "gpu": {"reachable": False},
    }
    try:
        gpu = await get_rag_client().health()
        result["gpu"] = {"reachable": True, **gpu}
    except Exception as e:
        result["gpu"]["error"] = str(e)
    return result
