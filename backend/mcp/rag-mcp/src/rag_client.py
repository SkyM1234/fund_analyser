"""GPU RAG 服务异步客户端

封装对 GPU 服务器上 RAG API 的 HTTP 访问。
"""
import os
from typing import Any

import httpx


class RagClient:
    """RAG 服务 HTTP 客户端"""
    
    def __init__(self, base_url: str | None = None) -> None:
        """初始化客户端
        
        Args:
            base_url: RAG 服务的基础 URL，如不提供则从环境变量读取
        """
        if base_url is None:
            gpu_host = os.getenv("GPU_HOST", "localhost")
            gpu_port = os.getenv("GPU_PORT", "8001")
            base_url = f"http://{gpu_host}:{gpu_port}"
        
        self._base_url = base_url
        self._client = httpx.AsyncClient(base_url=base_url, timeout=30.0)

    async def health(self) -> dict[str, Any]:
        """检查服务健康状态"""
        r = await self._client.get("/health")
        r.raise_for_status()
        return r.json()

    async def stats(self) -> dict[str, Any]:
        """获取服务统计信息"""
        r = await self._client.get("/stats")
        r.raise_for_status()
        return r.json()

    async def search(
        self,
        query: str,
        top_k: int = 10,
        filter_fund_code: str | None = None,
        search_type: str = "hybrid",
        use_reranker: bool = True,
        min_score: float | None = None,
    ) -> list[dict[str, Any]]:
        """检索基金报告内容

        Args:
            query: 查询文本
            top_k: 返回结果数量
            filter_fund_code: 单基金代码过滤，如 "159103"；None 表示全局检索
            search_type: 检索类型 - "dense"(仅稠密), "sparse"(仅稀疏), "hybrid"(混合RRF，推荐)
            use_reranker: 是否使用BGE-Reranker-v2-m3重排（推荐开启）
            min_score: 可选的分数阈值，低于该值的结果丢弃（避免凑数）

        Returns:
            检索结果列表，每个结果包含：
            - content: 文档内容
            - fund_code: 基金代码
            - score: 相似度分数
        """
        # 构建请求参数
        payload = {
            "query": query,
            "top_k": top_k,
            "filter_fund_code": filter_fund_code,
            "search_type": search_type,
            "use_reranker": use_reranker,
            "rerank_top_k": top_k * 2,  # 重排前获取2倍候选
        }
        if min_score is not None:
            payload["min_score"] = min_score

        r = await self._client.post("/fund_reports/search", json=payload)
        r.raise_for_status()
        return r.json().get("results", [])

    async def identify_funds(
        self,
        query: str,
        top_k: int = 5,
        min_score: float = 0.5,
    ) -> list[dict]:
        """两级RAG第一级：从用户问题中语义识别基金代码。

        通过检索 fund_index collection 实现，能处理别名、简称、模糊称呼等
        字符串匹配无法覆盖的情况。纯6位数字代码由服务端做精确匹配短路，
        无需调用方额外处理。

        Args:
            query:     用户问题原文（或已提取的描述片段）
            top_k:     最多返回几个候选基金
            min_score: 最低置信度，低于此值的结果丢弃（默认 0.5）

        Returns:
            [{"fund_code": "159103", "full_name": "...", "short_name": "...", "score": 0.92}, ...]
        """
        payload = {
            "query": query,
            "top_k": top_k,
            "min_score": min_score,
        }
        r = await self._client.post("/fund_index/search", json=payload)
        r.raise_for_status()
        return r.json().get("results", [])

    async def list_fund_index(self) -> list[dict]:
        """从 fund_index collection 获取所有基金清单（代码+名称）。

        用于替代原有的 list_funds() + FundRegistry 流程。

        Returns:
            [{"code": "159103", "full_name": "...", "short_name": "..."}, ...]
        """
        r = await self._client.get("/fund_index/list")
        r.raise_for_status()
        return r.json().get("funds", [])

    async def list_funds(self) -> list[dict[str, str]]:
        """获取所有基金清单（兼容旧接口，从年报 collection 读取）

        Returns:
            基金列表，每个基金包含 code 和 name
        """
        r = await self._client.get("/funds")
        r.raise_for_status()
        return r.json().get("funds", [])

    async def aclose(self) -> None:
        """关闭客户端连接"""
        await self._client.aclose()
