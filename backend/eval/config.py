"""评测框架配置。

按需从 .env / 系统环境变量读取。Judge LLM 与业务 LLM 解耦，避免同源偏差。
"""
from functools import lru_cache
import os
from pathlib import Path

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class EvalSettings(BaseSettings):
    """评测专用设置。"""

    model_config = SettingsConfigDict(
        env_file=Path(__file__).with_name(".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ===== LangSmith =====
    LANGSMITH_API_KEY: str = ""
    LANGSMITH_PROJECT: str = "fund-analyser-eval"
    LANGSMITH_TRACING: bool = True
    LANGSMITH_ENDPOINT: str = "https://api.smith.langchain.com"

    # ===== LLM Judge =====
    JUDGE_LLM_BASE_URL: str = "https://api.deepseek.com/v1"
    JUDGE_LLM_API_KEY: str = Field(
        default="",
        validation_alias=AliasChoices("JUDGE_LLM_API_KEY", "DEEPSEEK_API_KEY"),
    )
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

    GPU_HOST: str = "localhost"
    GPU_PORT: int = 8001

    @property
    def gpu_base_url(self) -> str:
        return f"http://{self.GPU_HOST}:{self.GPU_PORT}"

    def export_langsmith_env(self) -> None:
        """把 LangSmith 配置写入进程环境变量，供 langsmith SDK 自动读取。"""
        api_key = self.LANGSMITH_API_KEY.strip()
        if not api_key:
            env_path = Path(__file__).with_name(".env")
            raise RuntimeError(
                f"未配置 LANGSMITH_API_KEY，请在 {env_path} 或系统环境变量中设置有效值"
            )

        os.environ["LANGSMITH_API_KEY"] = api_key
        print(f"[OK] LangSmith API Key 已注入环境变量 LANGSMITH_API_KEY（长度={len(api_key)}）")
        os.environ["LANGSMITH_PROJECT"] = self.LANGSMITH_PROJECT
        os.environ["LANGSMITH_TRACING"] = "true" if self.LANGSMITH_TRACING else "false"
        os.environ["LANGSMITH_ENDPOINT"] = self.LANGSMITH_ENDPOINT

@lru_cache
def get_eval_settings() -> EvalSettings:
    return EvalSettings()
