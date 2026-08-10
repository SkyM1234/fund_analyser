"""MCP 管理 API - 监控和管理 MCP 工具调用。"""
import logging
from fastapi import APIRouter, Depends, HTTPException

from app.core.security import get_current_admin, get_current_user
from app.db.models import User
from app.services.mcp_client import get_mcp_client

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/mcp", tags=["mcp"])


@router.get("/stats")
async def get_mcp_stats(user: User = Depends(get_current_user)):
    """获取当前用户在当前限流窗口内的 MCP 工具调用统计信息。

    Returns:
        包含以下信息的字典：
        - total_calls: 当前窗口内总调用次数
        - max_total_calls: 每窗口最大总调用次数限制（None 表示无限制）
        - max_calls_per_tool: 每窗口单个工具最大调用次数限制（None 表示无限制）
        - tool_calls: 各工具在当前窗口内的调用次数统计
        - remaining_total_calls: 当前窗口剩余可用总调用次数（None 表示无限制）
    """
    try:
        mcp_client = await get_mcp_client()
        stats = await mcp_client.get_call_stats(user_id=str(user.id))
        return {
            "status": "success",
            "data": stats
        }
    except Exception as e:
        logger.error(f"获取 MCP 统计信息失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/reset-stats")
async def reset_mcp_stats(user: User = Depends(get_current_admin)):
    """重置当前管理员的 MCP 工具调用计数器。"""
    try:
        mcp_client = await get_mcp_client()
        await mcp_client.reset_call_counts(user_id=str(user.id))
        return {
            "status": "success",
            "message": "MCP 调用计数器已重置"
        }
    except Exception as e:
        logger.error(f"重置 MCP 统计信息失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/tools")
async def get_mcp_tools(user: User = Depends(get_current_user)):
    """获取所有可用的 MCP 工具列表。

    Returns:
        工具列表，包含工具名称、描述等信息
    """
    try:
        mcp_client = await get_mcp_client()
        tools = await mcp_client.list_all_tools()
        return {
            "status": "success",
            "data": {
                "count": len(tools),
                "tools": tools
            }
        }
    except Exception as e:
        logger.error(f"获取 MCP 工具列表失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))
