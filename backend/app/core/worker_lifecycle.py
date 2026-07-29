"""Celery worker 进程生命周期：完整复刻 app.main 的 lifespan。

每个 worker 进程是独立的 OS 进程，不与 FastAPI 共享任何内存状态，因此需要
自己的 MCP client（stdio 子进程连接）、checkpoint 连接池、MySQL engine、
Redis client——初始化逻辑与 app.main.lifespan 完全一致。

关键约束：MCP 的 stdio 连接、psycopg 的 AsyncConnectionPool 都绑定在创建它们
的事件循环上，不能跨循环使用；而 Celery 任务函数本身是同步调用（无论
solo/prefork/gevent pool）。因此本模块在 worker 进程内启动一个专属的后台
线程，其中运行一个长驱的事件循环（loop 在进程生命周期内只创建一次），所有
初始化 / 任务协程 / 关闭逻辑都通过 run_coroutine_threadsafe 提交到这个循环
执行，从而保证 MCP/checkpoint 连接始终在同一个循环上使用。
"""
import asyncio
import logging
import sys
import threading

logger = logging.getLogger(__name__)

# Windows 下 psycopg 异步连接池与默认 ProactorEventLoop 不兼容，worker 内的
# 后台事件循环同样需要该策略（原因与 app.main 顶部注释一致）。
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

_loop: asyncio.AbstractEventLoop | None = None
_loop_thread: threading.Thread | None = None


def get_worker_loop() -> asyncio.AbstractEventLoop:
    """返回本 worker 进程专属的后台事件循环（未初始化则报错）。"""
    if _loop is None:
        raise RuntimeError("worker 事件循环尚未初始化，worker_init 信号是否已触发？")
    return _loop


def run_coro(coro, timeout: float | None = None):
    """在 worker 的后台事件循环上执行协程，阻塞当前（Celery 任务）线程直到完成。

    Celery 的 soft_time_limit/time_limit 通过向主线程发信号实现（SIGUSR1/
    SIGALRM），而 Python 信号处理器只在主线程运行；协程本身跑在后台循环线程
    上不会被信号打断，也不会被同线程的 future.cancel() 真正中断（它跑在
    另一个线程的事件循环里，取消必须发生在那个循环内部）。因此这里在提交
    前用 asyncio.wait_for 包一层：超时时 asyncio 会在协程内部正确地抛出
    CancelledError 并等待其清理（如释放分布式锁的 finally 块），而不是把
    协程留在后台循环里变成孤儿任务。
    """
    loop = get_worker_loop()
    coro_to_run = asyncio.wait_for(coro, timeout=timeout) if timeout else coro
    future = asyncio.run_coroutine_threadsafe(coro_to_run, loop)
    return future.result()


def _run_loop_forever(loop: asyncio.AbstractEventLoop) -> None:
    asyncio.set_event_loop(loop)
    loop.run_forever()


async def _startup() -> None:
    from app.core.config import get_settings

    settings = get_settings()

    try:
        from app.db.redis import get_redis_client

        redis_client = get_redis_client()
        await redis_client.ping()
        logger.info("✓ [worker] Redis 连接正常")
    except Exception as e:
        logger.error(f"✗ [worker] Failed to connect to Redis: {e}")

    if not settings.MCP_ENABLED:
        logger.info("[worker] MCP is disabled")
        return

    mcp_servers = settings.mcp_servers_list
    if not mcp_servers:
        logger.info("[worker] No MCP servers configured")
        return

    try:
        from app.services.mcp_client import get_mcp_client, MCPServerConfig, set_cached_mcp_tools
        from app.tools.mcp_adapter import load_mcp_tools

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

        mcp_client = await get_mcp_client()
        await mcp_client.initialize(server_configs)

        logger.info(f"✓ [worker] MCP client initialized with {len(server_configs)} servers")

        mcp_tools = await load_mcp_tools()
        set_cached_mcp_tools(mcp_tools)

        if mcp_tools:
            logger.info(f"✓ [worker] Loaded and cached {len(mcp_tools)} MCP tools")
        else:
            logger.warning("[worker] No MCP tools loaded")

    except Exception as e:
        logger.error(f"✗ [worker] Failed to initialize MCP client: {e}")
        import traceback
        logger.error(traceback.format_exc())


async def _shutdown() -> None:
    from app.core.config import get_settings

    settings = get_settings()

    try:
        from app.services.checkpoint import close_checkpointer
        await close_checkpointer()
    except Exception as e:
        logger.error(f"✗ [worker] Failed to close checkpoint pool: {e}")

    try:
        from app.db.mysql import close_engine
        await close_engine()
    except Exception as e:
        logger.error(f"✗ [worker] Failed to close MySQL engine: {e}")

    try:
        from app.db.redis import close_redis_client
        await close_redis_client()
    except Exception as e:
        logger.error(f"✗ [worker] Failed to close Redis client: {e}")

    if not settings.MCP_ENABLED:
        return

    try:
        from app.services.mcp_client import get_mcp_client

        mcp_client = await get_mcp_client()
        await mcp_client.close()

        logger.info("✓ [worker] MCP client closed")
    except Exception as e:
        logger.error(f"✗ [worker] Failed to close MCP client: {e}")


def on_worker_process_init(**kwargs) -> None:
    """worker_init 信号处理：启动后台事件循环线程，并在其上运行启动逻辑。"""
    global _loop, _loop_thread

    _loop = asyncio.new_event_loop()
    _loop_thread = threading.Thread(
        target=_run_loop_forever, args=(_loop,), name="celery-worker-aio-loop", daemon=True
    )
    _loop_thread.start()

    future = asyncio.run_coroutine_threadsafe(_startup(), _loop)
    future.result()
    logger.info("✓ [worker] 进程内资源初始化完成")


def on_worker_process_shutdown(**kwargs) -> None:
    """worker_shutdown 信号处理：在后台循环上执行关闭逻辑，再停止循环。"""
    global _loop, _loop_thread

    if _loop is None:
        return

    try:
        future = asyncio.run_coroutine_threadsafe(_shutdown(), _loop)
        future.result(timeout=30)
    except Exception as e:
        logger.error(f"✗ [worker] 关闭流程出错: {e}")
    finally:
        _loop.call_soon_threadsafe(_loop.stop)
        if _loop_thread is not None:
            _loop_thread.join(timeout=10)
        _loop = None
        _loop_thread = None
        logger.info("✓ [worker] 进程内资源已清理")
