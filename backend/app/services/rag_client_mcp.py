"""RAG 客户端 MCP 适配器 - 通过 MCP 访问 GPU RAG 服务

不再直接 HTTP 访问 GPU，而是通过 rag-mcp 的 MCP 工具。
提供与原有 RagClient 相同的接口，但底层使用 MCP。
"""
import json
import logging
from typing import Any

logger = logging.getLogger(__name__)


class RagClientMCP:
    """RAG 客户端 - MCP 版本"""
    
    def __init__(self) -> None:
        """初始化 MCP 适配器"""
        self._server_name = "rag-mcp"
    
    async def _call_tool(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        """调用 MCP 工具的通用方法"""
        from app.services.mcp_client import get_mcp_client
        
        mcp_client = await get_mcp_client()
        return await mcp_client.call_tool(tool_name, arguments, server_name=self._server_name)
    
    async def health(self) -> dict[str, Any]:
        """检查服务健康状态"""
        result = await self._call_tool("rag_health", {})
        return json.loads(result)
    
    async def stats(self) -> dict[str, Any]:
        """获取服务统计信息"""
        result = await self._call_tool("rag_stats", {})
        return json.loads(result)
    
    async def search(
        self,
        query: str,
        top_k: int = 10,
        filter_fund_code: str | None = None,
        search_type: str = "hybrid",
        use_reranker: bool = True,
    ) -> list[dict[str, Any]]:
        """检索基金报告内容
        
        Args:
            query: 查询文本
            top_k: 返回结果数量
            filter_fund_code: 可选的基金代码过滤
            search_type: 检索类型
            use_reranker: 是否使用重排序
        
        Returns:
            检索结果列表（已解析为字典）
        """
        result_text = await self._call_tool("rag_search", {
            "query": query,
            "top_k": top_k,
            "filter_fund_code": filter_fund_code,
            "search_type": search_type,
            "use_reranker": use_reranker,
        })
        
        # 解析返回的文本格式结果
        # 格式: "--- 结果 1 (相似度: 0.xxxx) ---\n基金代码: ...\n..."
        return self._parse_search_results(result_text)
    
    def _parse_search_results(self, text: str) -> list[dict[str, Any]]:
        """解析 RAG 搜索结果文本"""
        if "未找到相关内容" in text:
            return []
        
        results = []
        
        # 按 "--- 结果" 分割
        sections = text.split("--- 结果")
        
        for section in sections[1:]:  # 跳过第一个空部分
            try:
                lines = section.strip().split('\n')
                
                # 解析第一行：相似度
                first_line = lines[0]  # "1 (相似度: 0.8523) ---"
                score = 0.0
                if "相似度:" in first_line:
                    score_str = first_line.split("相似度:")[1].split(")")[0].strip()
                    try:
                        score = float(score_str)
                    except:
                        pass
                
                # 解析其他字段
                result = {"score": score}
                content_lines = []
                in_content = False
                
                for line in lines[1:]:
                    line = line.strip()
                    if not line:
                        continue
                    
                    if line.startswith("基金代码:"):
                        result["fund_code"] = line.split(":", 1)[1].strip()
                    elif line.startswith("文档类型:"):
                        result["doc_type"] = line.split(":", 1)[1].strip()
                    elif line.startswith("报告时间:"):
                        result["report_date"] = line.split(":", 1)[1].strip()
                    elif line.startswith("内容:"):
                        in_content = True
                    elif in_content:
                        content_lines.append(line)
                
                result["content"] = "\n".join(content_lines)
                results.append(result)
                
            except Exception as e:
                logger.warning(f"Failed to parse search result section: {e}")
                continue
        
        return results
    
    async def list_funds(self) -> list[dict[str, str]]:
        """获取所有基金清单
        
        Returns:
            基金列表 [{"code": "...", "name": "..."}, ...]
        """
        result_text = await self._call_tool("rag_list_funds", {})
        
        # 解析返回的文本
        # 格式: "共 N 只基金:\nCODE1: NAME1\nCODE2: NAME2\n..."
        funds = []
        lines = result_text.strip().split('\n')
        
        for line in lines[1:]:  # 跳过第一行
            line = line.strip()
            if not line or ':' not in line:
                continue
            
            try:
                code, name = line.split(':', 1)
                funds.append({
                    "code": code.strip(),
                    "name": name.strip()
                })
            except:
                continue
        
        return funds
    
    async def aclose(self) -> None:
        """关闭客户端（MCP 版本不需要特殊处理）"""
        pass


# 全局单例
_rag_client_mcp: RagClientMCP | None = None


def get_rag_client() -> RagClientMCP:
    """获取 RAG 客户端（MCP 版本）"""
    global _rag_client_mcp
    if _rag_client_mcp is None:
        _rag_client_mcp = RagClientMCP()
    return _rag_client_mcp
