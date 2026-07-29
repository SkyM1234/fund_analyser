"""MCP 客户端管理器 - 连接和管理多个 MCP 服务器。

支持：
- 启动和连接多个 MCP 服务器（stdio、SSE）
- 动态发现和注册工具
- 统一的工具调用接口
- 生命周期管理
"""
import json
import logging
from typing import Any
from contextlib import AsyncExitStack

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from app.db.redis import get_redis_client

logger = logging.getLogger(__name__)


# 全局缓存 MCP 工具（避免重复加载）
_cached_mcp_tools: list = []


class MCPServerConfig:
    """MCP 服务器配置。"""
    
    def __init__(
        self,
        name: str,
        command: str,
        args: list[str] | None = None,
        env: dict[str, str] | None = None,
        cwd: str | None = None,
    ):
        self.name = name
        self.command = command
        self.args = args or []
        self.env = env or {}
        self.cwd = cwd


class MCPClient:
    """MCP 客户端管理器。"""

    def __init__(
        self,
        max_total_calls: int | None = None,
        max_calls_per_tool: int | None = None,
        rate_limit_window_seconds: int = 60,
        cache_ttl_seconds: int = 30,
        cacheable_tools: set[str] | None = None,
    ):
        """初始化 MCP 客户端。

        Args:
            max_total_calls: 每用户每滚动窗口的全局最大调用次数（None 表示无限制）
            max_calls_per_tool: 每用户每工具每滚动窗口最大调用次数（None 表示无限制）
            rate_limit_window_seconds: 限流滚动窗口长度（秒）
            cache_ttl_seconds: 实时行情类工具结果缓存 TTL（秒）
            cacheable_tools: 允许缓存结果的工具名集合
        """
        self.sessions: dict[str, ClientSession] = {}
        self.exit_stack = AsyncExitStack()
        self._initialized = False

        # 调用次数限制配置（按用户分桶，存储在 Redis，跨进程/重启保持一致）
        self.max_total_calls = max_total_calls
        self.max_calls_per_tool = max_calls_per_tool
        self.rate_limit_window_seconds = rate_limit_window_seconds

        # 实时行情缓存配置
        self.cache_ttl_seconds = cache_ttl_seconds
        self.cacheable_tools = cacheable_tools or set()
    
    async def initialize(self, server_configs: list[MCPServerConfig]) -> None:
        """初始化并连接所有配置的 MCP 服务器。
        
        Args:
            server_configs: MCP 服务器配置列表
        """
        if self._initialized:
            logger.warning("MCP client already initialized")
            return
        
        for config in server_configs:
            try:
                await self._connect_server(config)
                logger.info(f"✓ Connected to MCP server: {config.name}")
            except Exception as e:
                logger.error(f"✗ Failed to connect to {config.name}: {e}")
        
        self._initialized = True
    
    async def _connect_server(self, config: MCPServerConfig) -> None:
        """连接单个 MCP 服务器。"""
        # 创建 stdio 连接
        server_params = StdioServerParameters(
            command=config.command,
            args=config.args,
            env=config.env,
            cwd=config.cwd,
        )
        
        stdio_transport = await self.exit_stack.enter_async_context(
            stdio_client(server_params)
        )
        
        read_stream, write_stream = stdio_transport
        session = await self.exit_stack.enter_async_context(
            ClientSession(read_stream, write_stream)
        )
        
        # 初始化会话
        await session.initialize()
        
        self.sessions[config.name] = session
    
    async def list_all_tools(self) -> list[dict[str, Any]]:
        """列出所有已连接服务器的工具。
        
        Returns:
            工具列表，每个工具包含：name, description, server, input_schema
        """
        all_tools = []
        
        for server_name, session in self.sessions.items():
            try:
                result = await session.list_tools()
                for tool in result.tools:
                    all_tools.append({
                        "name": tool.name,
                        "description": tool.description or "",
                        "server": server_name,
                        "input_schema": tool.inputSchema,
                    })
            except Exception as e:
                logger.error(f"Failed to list tools from {server_name}: {e}")
        
        return all_tools
    
    async def call_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        server_name: str | None = None,
        user_id: str | None = None,
    ) -> Any:
        """调用 MCP 工具。

        Args:
            tool_name: 工具名称
            arguments: 工具参数
            server_name: 服务器名称（如不指定则搜索所有服务器）
            user_id: 发起调用的用户 id，用于按用户分桶限流（None 归入 anonymous 桶）

        Returns:
            工具执行结果
        """
        # 如果指定了服务器名称
        if server_name:
            if server_name not in self.sessions:
                raise ValueError(f"MCP server not found: {server_name}")
            return await self._call_tool_on_server(
                self.sessions[server_name], tool_name, arguments, server_name, user_id
            )

        # 否则尝试所有服务器
        for name, session in self.sessions.items():
            try:
                return await self._call_tool_on_server(session, tool_name, arguments, name, user_id)
            except Exception:
                continue

        raise ValueError(f"Tool not found in any server: {tool_name}")

    def _rate_limit_keys(self, user_id: str, tool_name: str) -> tuple[str, str]:
        bucket = user_id or "anonymous"
        return f"mcp:calls:{bucket}:total", f"mcp:calls:{bucket}:{tool_name}"

    async def _check_and_incr_rate_limit(self, user_id: str | None, tool_name: str) -> None:
        """按用户分桶的滚动窗口限流（Redis INCR + EXPIRE）。"""
        if self.max_total_calls is None and self.max_calls_per_tool is None:
            return

        redis_client = get_redis_client()
        total_key, tool_key = self._rate_limit_keys(user_id or "anonymous", tool_name)

        if self.max_total_calls is not None:
            total_calls = await redis_client.incr(total_key)
            if total_calls == 1:
                await redis_client.expire(total_key, self.rate_limit_window_seconds)
            if total_calls > self.max_total_calls:
                raise ValueError(
                    f"已达到全局最大调用次数限制：{self.max_total_calls}/"
                    f"{self.rate_limit_window_seconds}s。当前调用次数：{total_calls}"
                )

        if self.max_calls_per_tool is not None:
            tool_calls = await redis_client.incr(tool_key)
            if tool_calls == 1:
                await redis_client.expire(tool_key, self.rate_limit_window_seconds)
            if tool_calls > self.max_calls_per_tool:
                raise ValueError(
                    f"工具 '{tool_name}' 已达到最大调用次数限制：{self.max_calls_per_tool}/"
                    f"{self.rate_limit_window_seconds}s。当前调用次数：{tool_calls}"
                )

    @staticmethod
    def _cache_key(server_name: str, tool_name: str, arguments: dict[str, Any]) -> str:
        args_repr = json.dumps(arguments, sort_keys=True, ensure_ascii=False)
        return f"mcp:cache:{server_name}:{tool_name}:{args_repr}"

    async def _call_tool_on_server(
        self,
        session: ClientSession,
        tool_name: str,
        arguments: dict[str, Any],
        server_name: str,
        user_id: str | None = None,
    ) -> Any:
        """在特定服务器上调用工具。"""
        redis_client = get_redis_client()
        cacheable = tool_name in self.cacheable_tools
        cache_key = self._cache_key(server_name, tool_name, arguments) if cacheable else None

        if cache_key:
            cached = await redis_client.get(cache_key)
            if cached is not None:
                logger.info(f"工具调用缓存命中: {tool_name}")
                return json.loads(cached)

        # 缓存未命中才计入限流并真正调用
        await self._check_and_incr_rate_limit(user_id, tool_name)

        result = await session.call_tool(tool_name, arguments)

        # 提取文本内容
        if result.content:
            if len(result.content) == 1:
                extracted = result.content[0].text
            else:
                extracted = [item.text for item in result.content]
        else:
            extracted = None

        logger.info(f"工具调用成功: {tool_name}, user={user_id or 'anonymous'}")

        if cache_key:
            await redis_client.set(cache_key, json.dumps(extracted, ensure_ascii=False), ex=self.cache_ttl_seconds)

        return extracted

    async def get_call_stats(self, user_id: str | None = None) -> dict[str, Any]:
        """获取指定用户当前窗口内的调用统计信息（从 Redis 读取）。"""
        redis_client = get_redis_client()
        total_key, _ = self._rate_limit_keys(user_id or "anonymous", "")
        total_calls = await redis_client.get(total_key)

        tool_calls: dict[str, int] = {}
        prefix = f"mcp:calls:{user_id or 'anonymous'}:"
        async for key in redis_client.scan_iter(match=f"{prefix}*"):
            if key == total_key:
                continue
            value = await redis_client.get(key)
            tool_calls[key[len(prefix):]] = int(value) if value else 0

        total_calls_int = int(total_calls) if total_calls else 0
        return {
            "user_id": user_id or "anonymous",
            "window_seconds": self.rate_limit_window_seconds,
            "total_calls": total_calls_int,
            "max_total_calls": self.max_total_calls,
            "max_calls_per_tool": self.max_calls_per_tool,
            "tool_calls": tool_calls,
            "remaining_total_calls": (
                max(self.max_total_calls - total_calls_int, 0)
                if self.max_total_calls is not None
                else None
            ),
        }

    async def reset_call_counts(self, user_id: str | None = None) -> None:
        """重置指定用户（或全部用户）的调用计数器。"""
        redis_client = get_redis_client()
        pattern = f"mcp:calls:{user_id}:*" if user_id else "mcp:calls:*"
        keys = [key async for key in redis_client.scan_iter(match=pattern)]
        if keys:
            await redis_client.delete(*keys)
        logger.info(f"调用计数器已重置: {user_id or 'all users'}")

    async def close(self) -> None:
        """关闭所有连接。"""
        await self.exit_stack.aclose()
        self.sessions.clear()
        self._initialized = False
        logger.info("MCP client closed")


# 全局实例
_mcp_client: MCPClient | None = None


async def get_mcp_client() -> MCPClient:
    """获取全局 MCP 客户端实例。"""
    from app.core.config import get_settings

    global _mcp_client
    if _mcp_client is None:
        settings = get_settings()
        _mcp_client = MCPClient(
            max_total_calls=settings.MCP_MAX_TOTAL_CALLS,
            max_calls_per_tool=settings.MCP_MAX_CALLS_PER_TOOL,
            rate_limit_window_seconds=settings.MCP_RATE_LIMIT_WINDOW_SECONDS,
            cache_ttl_seconds=settings.MCP_CACHE_TTL_SECONDS,
            cacheable_tools=settings.MCP_CACHEABLE_TOOLS,
        )
    return _mcp_client


def get_cached_mcp_tools() -> list:
    """获取缓存的 MCP 工具列表。"""
    return _cached_mcp_tools


def set_cached_mcp_tools(tools: list) -> None:
    """设置缓存的 MCP 工具列表。"""
    global _cached_mcp_tools
    _cached_mcp_tools = tools
