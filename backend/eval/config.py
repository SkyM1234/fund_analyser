"""评测框架配置。

按需从 .env / 系统环境变量读取。Judge LLM 与业务 LLM 解耦，避免同源偏差。
"""
from functools import lru_cache
import os

from pydantic_settings import BaseSettings, SettingsConfigDict


class EvalSettings(BaseSettings):
    """评测专用设置。"""

    # ===== LangSmith =====
    LANGSMITH_API_KEY: str = os.getenv("LANGSMITH_API_KEY", "")
    LANGSMITH_PROJECT: str = "fund-analyser-eval"
    LANGSMITH_TRACING: bool = True
    LANGSMITH_ENDPOINT: str = "https://api.smith.langchain.com"

    # ===== LLM Judge =====
    JUDGE_LLM_BASE_URL: str = "https://api.deepseek.com/v1"
    JUDGE_LLM_API_KEY: str = os.getenv("DEEPSEEK_API_KEY", "")
    JUDGE_LLM_MODEL: str = "deepseek-v4-pro"
    JUDGE_LLM_TEMPERATURE: float = 0.0

    # ===== 数据集 =====
    DATASET_RETRIEVAL_NAME: str = "fund-rag-retrieval-v1"
    DATASET_ANSWER_NAME: str = "fund-agent-answer-v1"
    DATASET_NAME_RESOLUTION_NAME: str = "fund-name-resolution-v1"

    # ===== 执行 =====
    EVAL_MAX_CONCURRENCY: int = 1

    # ===== MCP 配置 =====
    ENABLE_CN_FUNDS_MCP: bool = True  # 是否启用 cn-funds-mcp（可通过命令行参数覆盖）

    def export_langsmith_env(self) -> None:
        """把 LangSmith 配置写入进程环境变量，供 langsmith SDK 自动读取。"""
        os.environ["LANGSMITH_API_KEY"] = self.LANGSMITH_API_KEY
        os.environ["LANGSMITH_PROJECT"] = self.LANGSMITH_PROJECT
        os.environ["LANGSMITH_TRACING"] = "true" if self.LANGSMITH_TRACING else "false"
        os.environ["LANGSMITH_ENDPOINT"] = self.LANGSMITH_ENDPOINT

@lru_cache
def get_eval_settings() -> EvalSettings:
    return EvalSettings()
