"""对话历史处理工具函数"""
from typing import List
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage


def extract_recent_history(messages: List[BaseMessage], rounds: int = 3) -> List[BaseMessage]:
    """提取最近N轮对话历史
    
    Args:
        messages: 完整的消息列表
        rounds: 保留的轮数(基于用户问题数量)
        
    Returns:
        最近N轮的消息列表(按时间顺序)
    """
    recent_history = []
    human_count = 0
    
    # 从后往前遍历
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            human_count += 1
            if human_count > rounds:
                break
        recent_history.insert(0, msg)
    
    return recent_history


def format_history_for_prompt(
    messages: List[BaseMessage],
    rounds: int = 3,
    max_response_length: int = 200,
    exclude_last: bool = True
) -> str:
    """格式化对话历史为prompt文本
    
    Args:
        messages: 完整的消息列表
        rounds: 保留的轮数
        max_response_length: AI回复的最大长度(截断)
        exclude_last: 是否排除最后一条消息(通常最后一条是当前问题)
        
    Returns:
        格式化的历史文本
    """
    recent_history = extract_recent_history(messages, rounds)
    
    if not recent_history or (exclude_last and len(recent_history) <= 1):
        return ""
    
    # 如果需要排除最后一条,去掉
    if exclude_last:
        recent_history = recent_history[:-1]
    
    if not recent_history:
        return ""
    
    history_text = "\n\n对话历史(用于理解上下文):\n"
    for msg in recent_history:
        if isinstance(msg, HumanMessage):
            history_text += f"用户: {msg.content}\n"
        elif isinstance(msg, AIMessage):
            # 截断过长的回复
            content = msg.content
            if len(content) > max_response_length:
                content = content[:max_response_length] + "..."
            history_text += f"助手: {content}\n"
    
    return history_text


def get_recent_messages_for_agent(
    messages: List[BaseMessage],
    rounds: int = 2,
    max_response_length: int = 500,
    exclude_current_human: bool = False,
) -> List[BaseMessage]:
    """获取专家Agent使用的最近消息列表(用于注入Agent的消息历史)

    Args:
        messages:               完整的消息列表
        rounds:                 保留的轮数
        max_response_length:    AI回复的最大长度(截断)
        exclude_current_human:  是否排除最末尾的 HumanMessage。
                                子任务 Agent 应传 True，避免 Agent 看到含多基金的
                                原始用户问题（如"对比A和B"），从而越权做跨基金对比。

    Returns:
        处理后的消息列表(可直接添加到Agent的messages中)
    """
    source = messages

    # 排除当前轮的用户消息（列表末尾的最后一条 HumanMessage）
    if exclude_current_human and source:
        # 从末尾找到第一条 HumanMessage，将其及之后的内容都去掉
        cutoff = len(source)
        for i in range(len(source) - 1, -1, -1):
            if isinstance(source[i], HumanMessage):
                cutoff = i
                break
        source = source[:cutoff]

    recent_history = extract_recent_history(source, rounds)

    # 保留最近的 rounds*2 条消息(每轮包含用户问题+AI回答)
    max_messages = rounds * 2
    if len(recent_history) > max_messages:
        recent_history = recent_history[-max_messages:]

    # 截断过长的AI回复
    processed_messages = []
    for msg in recent_history:
        if isinstance(msg, AIMessage) and len(msg.content) > max_response_length:
            # 创建新的AIMessage,内容被截断
            processed_messages.append(
                AIMessage(content=msg.content[:max_response_length])
            )
        else:
            processed_messages.append(msg)

    return processed_messages
