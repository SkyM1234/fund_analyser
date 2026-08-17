"""配置加载（pydantic-settings，仅从系统环境变量读取）

约定：
- 默认值写在类里；
- 敏感值（如 LLM_API_KEY）通过环境变量注入，不落盘。
"""
from functools import lru_cache
import os
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(extra="ignore")

    # LLM
    LLM_BASE_URL: str = "https://api.deepseek.com/v1"
    LLM_API_KEY: str = os.getenv("DEEPSEEK_API_KEY", "")
    LLM_MODEL: str = "deepseek-v4-flash"
    LLM_TEMPERATURE: float = 0.2
    LLM_MAX_CONCURRENCY: int = int(os.getenv("LLM_MAX_CONCURRENCY", "10"))  # 单进程内并发 LLM 调用上限

    # GPU
    GPU_HOST: str = os.getenv("GPU_HOST", "localhost")
    GPU_PORT: int = 8001

    # Agent
    AGENT_TIMEOUT: int = 300

    # PostgreSQL (Checkpoint，部署在 GPU 电脑)
    # Docker 部署时 POSTGRES_HOST 指向 compose 中的 postgres 服务名，而非 GPU_HOST
    POSTGRES_HOST: str = os.getenv("POSTGRES_HOST", "")
    POSTGRES_PORT: int = int(os.getenv("POSTGRES_PORT", "5432"))

    @property
    def POSTGRES_URI(self) -> str:
        host = self.POSTGRES_HOST or self.GPU_HOST
        return f"postgresql://fund_user:fund_pass@{host}:{self.POSTGRES_PORT}/fund_chat"

    # PostgreSQL 连接池（AsyncConnectionPool）
    PG_POOL_MIN_SIZE: int = int(os.getenv("PG_POOL_MIN_SIZE", "5"))
    PG_POOL_MAX_SIZE: int = int(os.getenv("PG_POOL_MAX_SIZE", "20"))
    PG_POOL_TIMEOUT: float = float(os.getenv("PG_POOL_TIMEOUT", "30"))
    PG_POOL_MAX_IDLE: float = float(os.getenv("PG_POOL_MAX_IDLE", "300"))
    PG_POOL_MAX_LIFETIME: float = float(os.getenv("PG_POOL_MAX_LIFETIME", "1800"))
    PG_POOL_NUM_WORKERS: int = int(os.getenv("PG_POOL_NUM_WORKERS", "3"))

    # MySQL（业务数据：用户、会话索引、合规/用量审计）
    # 默认跟随 GPU_HOST（与 PostgreSQL/Redis 同机 Docker 部署）
    MYSQL_HOST: str = os.getenv("MYSQL_HOST", "")
    MYSQL_PORT: int = int(os.getenv("MYSQL_PORT", "3306"))
    MYSQL_USER: str = os.getenv("MYSQL_USER", "root")
    MYSQL_PASSWORD: str = os.getenv("MYSQL_PASSWORD", "")
    MYSQL_DATABASE: str = os.getenv("MYSQL_DATABASE", "fund_analyser")

    @property
    def MYSQL_URI(self) -> str:
        host = self.MYSQL_HOST or self.GPU_HOST
        return (
            f"mysql+asyncmy://{self.MYSQL_USER}:{self.MYSQL_PASSWORD}"
            f"@{host}:{self.MYSQL_PORT}/{self.MYSQL_DATABASE}"
        )

    # Redis（缓存 / 限流 / 分布式锁）；默认跟随 GPU_HOST，Docker Redis 未设置密码
    @property
    def REDIS_URL(self) -> str:
        return os.getenv("REDIS_URL", f"redis://{self.GPU_HOST}:6379/0")

    # Celery（agent 任务队列）；broker/result backend 用独立 db，避免与业务缓存/锁的
    # key 空间混在一起（尤其是 FLUSHDB 误操作时不会互相牵连）
    @property
    def CELERY_BROKER_URL(self) -> str:
        return os.getenv("CELERY_BROKER_URL", f"redis://{self.GPU_HOST}:6379/1")

    @property
    def CELERY_RESULT_BACKEND(self) -> str:
        return os.getenv("CELERY_RESULT_BACKEND", f"redis://{self.GPU_HOST}:6379/2")

    CELERY_TASK_SOFT_TIME_LIMIT: int = int(os.getenv("CELERY_TASK_SOFT_TIME_LIMIT", "330"))
    CELERY_TASK_TIME_LIMIT: int = int(os.getenv("CELERY_TASK_TIME_LIMIT", "360"))
    CELERY_RESULT_EXPIRES: int = int(os.getenv("CELERY_RESULT_EXPIRES", "3600"))

    # JWT（内部签名密钥，非外部凭证；未设置环境变量时使用开发默认值）
    JWT_SECRET_KEY: str = os.getenv("JWT_SECRET_KEY", "dev-secret-change-me")
    JWT_ALGORITHM: str = "HS256"
    JWT_ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    JWT_REFRESH_TOKEN_EXPIRE_DAYS: int = 7

    # Server
    APP_HOST: str = "0.0.0.0"
    APP_PORT: int = 8800
    CORS_ORIGINS: str = "http://localhost:5173"

    @property
    def gpu_base_url(self) -> str:
        return f"http://{self.GPU_HOST}:{self.GPU_PORT}"

    @property
    def cors_list(self) -> list[str]:
        return [o.strip() for o in self.CORS_ORIGINS.split(",") if o.strip()]

    # MCP 配置
    MCP_ENABLED: bool = True  # 是否启用 MCP
    MCP_MAX_TOTAL_CALLS: int | None = 20  # 每用户每窗口的全局最大调用次数（None 表示无限制）
    MCP_MAX_CALLS_PER_TOOL: int | None = 10  # 每用户每工具每窗口最大调用次数（None 表示无限制）
    MCP_RATE_LIMIT_WINDOW_SECONDS: int = 60  # 限流滚动窗口长度（秒）
    MCP_SERVERS: str | None = None  # 由 mcp_servers_list 方法动态生成

    # 实时行情类工具缓存（短 TTL，减少重复调用 & 计入限流的次数）
    MCP_CACHE_TTL_SECONDS: int = 60
    MCP_CACHEABLE_TOOLS: set[str] = {
        "search_fund",
        "get_fund_estimate",
        "get_fund_info",
        "get_fund_valuation_detail",
        "get_fund_position",
    }

    # 显式 server → agent 工具映射；key 为 assigned_agent 值，value 为允许使用的 MCP server 名列表。
    # 优先级高于名称前缀过滤，可通过环境变量 MCP_AGENT_TOOL_SERVERS（JSON 字符串）覆盖。
    MCP_AGENT_TOOL_SERVERS: dict[str, list[str]] = {
        "rag_agent": ["rag-mcp"],
        "market_agent": ["cn-funds-mcp"],
    }

    # 按工具名排除：即使工具所属 server 已被允许，这里列出的工具名仍会被过滤掉。
    # 用于屏蔽当前场景不需要的能力（如持仓管理、定时提醒），而不必改动 MCP server 源码。
    MCP_EXCLUDED_TOOLS: set[str] = {
        "save_portfolio",
        "remove_portfolio",
        "get_portfolio",
        "get_portfolio_profit",
        "set_reminder",
        "get_reminders",
        "remove_reminder",
        "check_reminders",
    }

    @property
    def mcp_servers_list(self) -> list[dict]:
        """解析 MCP 服务器配置。

        环境变量示例：
        MCP_SERVERS='[
            {"name": "brave-search", "command": "npx", "args": ["-y", "@modelcontextprotocol/server-brave-search"], "env": {"BRAVE_API_KEY": "xxx"}},
            {"name": "fetch", "command": "uvx", "args": ["mcp-server-fetch"]}
        ]'

        支持的字段：
        - name (必填): 服务器唯一标识
        - command (必填): 启动命令
        - args (可选): 命令行参数数组，默认 []
        - env (可选): 环境变量字典，默认 {}
        - cwd (可选): 工作目录，默认 None
        """

        # 默认配置：动态使用当前的 GPU_HOST 和 GPU_PORT
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

        default_servers = [
            {
                "name": "cn-funds-mcp",
                "command": "node",
                "args": ["src/index.js"],
                "cwd": os.path.join(project_root, "mcp", "cn-funds-mcp-master"),
                "env": {}
            },
            {
                "name": "rag-mcp",
                "command": "python",
                "args": ["src/server.py"],
                "cwd": os.path.join(project_root, "mcp", "rag-mcp"),
                "env": {
                    "GPU_HOST": self.GPU_HOST,  # 动态使用当前配置
                    "GPU_PORT": str(self.GPU_PORT)
                }
            }
        ]

        return default_servers


@lru_cache
def get_settings() -> Settings:
    return Settings()
