"""从 LLM 响应中提取 token 用量，转换为 token_usage state 字段的增量更新"""
from langchain_core.messages import BaseMessage


def record_usage(bucket: str, response: BaseMessage) -> dict[str, dict[str, int]]:
    """构造一份 token_usage 增量更新，供节点合并进返回的 state 更新字典。

    Args:
        bucket: 调用点标识，如 "supervisor"、"rag_agent:t1"
        response: llm.ainvoke() 的返回消息，usage_metadata 缺失时（如流式响应
            不透出用量，或供应商未返回）返回空字典，不写入该 bucket

    Returns:
        形如 {bucket: {"input_tokens": .., "output_tokens": .., "total_tokens": ..}}，
        与 merge_token_usage reducer 配合按 bucket 累加。usage_metadata 缺失时返回 {}。
    """
    usage = getattr(response, "usage_metadata", None)
    if not usage:
        return {}

    return {
        bucket: {
            "input_tokens": usage.get("input_tokens", 0),
            "output_tokens": usage.get("output_tokens", 0),
            "total_tokens": usage.get("total_tokens", 0),
        }
    }
