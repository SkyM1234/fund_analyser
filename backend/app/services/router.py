"""查询路由器：两层路由。

第1层：快速规则过滤（闲聊、越界、敏感）
第2层：LLM意图分类，直接传入近几轮对话历史，由 LLM 结合上下文判断意图
"""
import json
import re
from typing import TYPE_CHECKING, Literal

from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from app.core.config import get_settings
from app.core.llm_concurrency import llm_ainvoke

if TYPE_CHECKING:
    from langchain_core.messages import BaseMessage


class RouteResult(BaseModel):
    """路由结果。"""

    intent: Literal[
        "chitchat",          # 闲聊/问候
        "out_of_scope",      # 不在能力范围
        "sensitive",         # 敏感问题（投资建议）
        "fund_query",        # 基金查询
        "fund_screening",    # 基金筛选（查找符合条件的基金）
        "general_finance",   # 通用金融知识
    ]
# ===== 第1层：快速规则过滤 =====
CHITCHAT_PATTERNS = [
    r'^(你好|hi|hello|嗨|哈喽|早上好|晚上好|下午好)',
    r'^(谢谢|多谢|感谢)',
    r'^(再见|拜拜|bye)',
    r'^(你是谁|你叫什么|你会什么)',
]

OUT_OF_SCOPE_KEYWORDS = ['天气', '新闻', '娱乐', '股票代码', '个股', '期货', '外汇', '加密货币']

SENSITIVE_KEYWORDS = [
    '推荐', '建议买', '该买', '应该买', '值得买', '买哪个', '买哪只', '买什么',
    '会涨', '会跌', '能涨', '能赚', '收益率预测', '未来收益', '预期收益',
    '保证收益', '稳赚', '必涨', '帮我选', '给我推荐',
]


# ===== 第2层：LLM分类 Prompt =====
LLM_CLASSIFIER_SYSTEM = """你是基金问答系统的意图分类器。根据对话历史和当前问题，判断用户意图。

意图类型：

**sensitive** - 请求投资建议、基金推荐或收益预测
- "推荐一只基金"、"买哪个好"、"这只基金会涨吗"、"帮我选"、"预期收益多少"
- 要求推荐、选择、预测的都归此类

**fund_query** - 查询基金信息
- "科创债ETF万家的规模"、"159103的持仓"
- "人工智能板块有哪些ETF"、"新能源基金有哪些"
- "比较华夏国证港股通科技ETF和招商国证港股通科技ETF"
- 包括具体基金、多个基金及板块/主题基金的客观查询

**fund_screening** - 不涉及特定基金或者板块，而是按条件筛选基金
- "规模超过10亿且费率低于0.5%的新能源基金"、"持有宁德时代且规模超过10亿的基金"
- 重点是多个客观条件的筛选，不是按板块或主题泛查基金

**general_finance** - 通用金融知识
- "什么是ETF"、"基金经理如何选股"、"新能源行业前景"

⚠️ 注意：
- 若当前消息是追问或模糊指令（如"再试一下"、"换一个"），结合对话历史判断实际意图
- sensitive 和 fund_query/fund_screening 的区别：前者要求推荐/预测，后者是客观查询

输出 JSON（不要解释）：
{"intent": "sensitive|fund_query|fund_screening|general_finance"}"""


async def route_query(
    query: str,
    history_messages: "list[BaseMessage] | None" = None,
) -> RouteResult:
    """两层路由。

    [第1层] 快速规则过滤（闲聊、越界、敏感）
    [第2层] LLM意图分类，传入近几轮对话让 LLM 结合上下文判断
    """
    query_clean = query.strip()

    # ===== 第1层：快速规则过滤 =====
    for pattern in CHITCHAT_PATTERNS:
        if re.search(pattern, query_clean, re.IGNORECASE):
            return RouteResult(intent="chitchat")

    if any(kw in query_clean for kw in OUT_OF_SCOPE_KEYWORDS):
        return RouteResult(intent="out_of_scope")

    if any(kw in query_clean for kw in SENSITIVE_KEYWORDS):
        return RouteResult(intent="sensitive")

    # ===== 第2层：LLM意图分类 =====
    return await _llm_classify(query_clean, history_messages)


async def _llm_classify(
    query: str,
    history_messages: "list[BaseMessage] | None" = None,
) -> RouteResult:
    """LLM 意图分类，直接传入近几轮对话历史。"""
    from langchain_core.messages import HumanMessage, AIMessage

    s = get_settings()
    llm = ChatOpenAI(
        base_url=s.LLM_BASE_URL,
        api_key=s.LLM_API_KEY,
        model=s.LLM_MODEL,
        temperature=0,
    )

    # 构建消息列表：system + 近3轮历史（不含当前） + 当前用户消息
    llm_messages: list[dict] = [{"role": "system", "content": LLM_CLASSIFIER_SYSTEM}]

    if history_messages:
        # 提取最近3轮（不含当前消息，即列表末尾的最后一条 HumanMessage）
        history_window: list[dict] = []
        human_count = 0
        skip_first_human = True
        for msg in reversed(history_messages):
            if isinstance(msg, HumanMessage):
                if skip_first_human:
                    skip_first_human = False
                    continue
                human_count += 1
                if human_count > 3:
                    break
                history_window.insert(0, {"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage):
                if not skip_first_human:
                    # 截断过长的 AI 回复，避免撑爆 context
                    content = msg.content[:300] + "..." if len(msg.content) > 300 else msg.content
                    history_window.insert(0, {"role": "assistant", "content": content})

        llm_messages.extend(history_window)

    llm_messages.append({"role": "user", "content": query})

    resp = await llm_ainvoke(llm, llm_messages)
    content = resp.content.strip()

    if content.startswith("```"):
        content = re.sub(r"^```(?:json)?\s*", "", content)
        content = re.sub(r"\s*```$", "", content)

    try:
        data = json.loads(content)
        intent = data["intent"]
        if intent in (
            "sensitive",
            "fund_query",
            "fund_screening",
            "general_finance",
        ):
            return RouteResult(intent=intent)
    except Exception:
        pass

    # 解析失败保守降级
    return RouteResult(intent="general_finance")
