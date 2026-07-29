"""FastAPI 入口。"""
import logging
import sys
from contextlib import asynccontextmanager
from logging.handlers import RotatingFileHandler
from pathlib import Path

# Windows 下 uvicorn 默认使用 ProactorEventLoop，与 psycopg 的异步连接池不兼容
# （会导致 checkpointer 连接池 PoolTimeout，进而使 SSE 响应中途断开）。
# 必须在任何事件循环创建之前设置该策略；且仅当启动方式不会覆盖 loop_factory
# 时才会生效——用 `uvicorn app.main:app` 直接启动时需加上 `--loop none`，
# 或使用本文件底部的 `python -m app.main` 启动方式。
if sys.platform == "win32":
    import asyncio
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import admin, auth, chat, health, session, mcp
from app.core.config import get_settings

# 配置日志：同时输出到控制台和文件
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)

# 创建根 logger
root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)

# 格式化器
formatter = logging.Formatter(
    "%(asctime)s %(levelname)-8s [%(name)s:%(lineno)d] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

# 控制台处理器
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(formatter)

# 文件处理器（按大小轮转）
file_handler = RotatingFileHandler(
    log_dir / "fund_api.log",
    maxBytes=10 * 1024 * 1024,  # 10MB
    backupCount=5,
    encoding="utf-8"
)
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(formatter)

# 添加处理器（避免 `python -m app.main` 启动时模块被 uvicorn 重新 import 导致 handler 重复叠加）
if not root_logger.handlers:
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)

# 降低第三方库日志级别
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("uvicorn.access").setLevel(logging.WARNING)

settings = get_settings()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理：启动时初始化 MCP，关闭时清理。"""
    logger = logging.getLogger(__name__)

    # === 启动逻辑 ===
    try:
        from app.db.redis import get_redis_client

        redis_client = get_redis_client()
        await redis_client.ping()
        logger.info("✓ Redis 连接正常")
    except Exception as e:
        logger.error(f"✗ Failed to connect to Redis: {e}")

    if not settings.MCP_ENABLED:
        logger.info("MCP is disabled")
    else:
        mcp_servers = settings.mcp_servers_list
        if not mcp_servers:
            logger.info("No MCP servers configured")
        else:
            try:
                from app.services.mcp_client import get_mcp_client, MCPServerConfig, set_cached_mcp_tools
                from app.tools.mcp_adapter import load_mcp_tools

                # 转换配置
                server_configs = [
                    MCPServerConfig(
                        name=s.get("name", ""),
                        command=s.get("command", ""),
                        args=s.get("args", []),
                        env=s.get("env", {}),
                        cwd=s.get("cwd"),
                    )
                    for s in mcp_servers
                ]

                # 初始化 MCP 客户端
                mcp_client = await get_mcp_client()
                await mcp_client.initialize(server_configs)

                logger.info(f"✓ MCP client initialized with {len(server_configs)} servers")

                # 加载并缓存 MCP 工具
                mcp_tools = await load_mcp_tools()
                set_cached_mcp_tools(mcp_tools)

                if mcp_tools:
                    logger.info(f"✓ Loaded and cached {len(mcp_tools)} MCP tools")
                else:
                    logger.warning("No MCP tools loaded")

            except Exception as e:
                logger.error(f"✗ Failed to initialize MCP client: {e}")
                import traceback
                logger.error(traceback.format_exc())

    yield  # 应用运行中

    # === 关闭逻辑 ===
    try:
        from app.services.checkpoint import close_checkpointer
        await close_checkpointer()
    except Exception as e:
        logger.error(f"✗ Failed to close checkpoint pool: {e}")

    try:
        from app.db.mysql import close_engine
        await close_engine()
    except Exception as e:
        logger.error(f"✗ Failed to close MySQL engine: {e}")

    try:
        from app.db.redis import close_redis_client
        await close_redis_client()
    except Exception as e:
        logger.error(f"✗ Failed to close Redis client: {e}")

    if not settings.MCP_ENABLED:
        return

    try:
        from app.services.mcp_client import get_mcp_client

        mcp_client = await get_mcp_client()
        await mcp_client.close()

        logger.info("✓ MCP client closed")
    except Exception as e:
        logger.error(f"✗ Failed to close MCP client: {e}")


app = FastAPI(title="Fund Analyser Backend", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router, prefix="/api", tags=["health"])
app.include_router(auth.router, prefix="/api", tags=["auth"])
app.include_router(chat.router, prefix="/api", tags=["chat"])
app.include_router(session.router, prefix="/api", tags=["session"])
app.include_router(admin.router, prefix="/api", tags=["admin"])
app.include_router(mcp.router, prefix="/api", tags=["mcp"])


@app.get("/")
async def root():
    return {"name": "fund-analyser-backend", "version": "0.1.0"}


if __name__ == "__main__":
    # 推荐的 Windows 启动方式：python -m app.main
    # 用 loop="none" 让 uvicorn 不覆盖事件循环的 loop_factory，
    # 从而使上面设置的 WindowsSelectorEventLoopPolicy 真正生效。
    # 直接传 app 对象而非 "app.main:app" 字符串，避免 uvicorn 内部
    # 再次 import 本模块（那样会导致模块级代码执行两次，例如日志 handler 重复叠加）。
    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8800,
        loop="none",
    )
