"""评测数据集 schema 定义。

三类数据集共用"查询"这一输入形式，但评测的环节和 ground truth 不同：

1. NameResolutionExample：评测两级RAG第一级（rag_identify_funds）
   - 能否从基金名称/别名/模糊描述正确识别出基金代码
   - 与 RetrievalExample 解耦：本类不涉及向量库内容检索

2. RetrievalExample：评测两级RAG第二级（rag_search）
   - 给定正确的 filter_fund_code 后，是否召回了目标基金的相关 chunk
   - 排序是否合理

3. AnswerExample：评测 Agent 端到端回答
   - 与参考答案是否一致
   - 是否产生幻觉
   - 引用是否正确
   - 敏感问题是否拒绝
"""
from typing import Literal, Optional

from pydantic import BaseModel, Field


# ============ 检索评测样本 ============
class RetrievalExample(BaseModel):
    """RAG 检索评测样本。

    至少需要 expected_fund_codes 或 relevant_chunk_ids 之一，作为 ground truth。
    若两者都有，相关性判定更精确；若都无，则只能跑 LLM-judge 模式。
    """

    id: str = Field(..., description="样本唯一 ID，建议 retrieval-001 这种格式")
    query: str = Field(..., description="用户查询原文")

    # ---- 输入侧（用于 target 调用 rag_search）----
    filter_fund_code: Optional[str] = Field(
        None, description="可选：调用 rag_search 时的过滤参数（单基金代码），模拟路由层注入"
    )
    top_k: int = Field(10, ge=10, description="检索返回数量，不能低于 10")

    # ---- ground truth ----
    expected_fund_codes: list[str] = Field(
        default_factory=list,
        description="期望被召回的基金代码集合（任意 chunk 命中即视为该基金被召回）",
    )
    relevant_chunk_ids: list[str] = Field(
        default_factory=list,
        description="可选：精确到 chunk 级的相关性标注（Milvus 主键 id）",
    )
    relevant_keywords: list[str] = Field(
        default_factory=list,
        description="可选：相关 chunk 应包含的关键词（用于无 chunk_id 时的弱标注）",
    )

    # ---- 元数据 ----
    category: Optional[str] = Field(
        None, description="样本类别，如 single_fund / multi_fund / global / screening"
    )
    note: Optional[str] = Field(None, description="标注说明")


# ============ 基金名称识别评测样本 ============
class NameResolutionExample(BaseModel):
    """两级RAG第一级（rag_identify_funds）评测样本：名称/别名/模糊描述 → 基金代码。

    与 RetrievalExample 解耦：RetrievalExample 的 filter_fund_code 是人工预置的
    正确答案，专测检索质量；本 schema 反过来测"能否从名称正确识别出代码"这一跳，
    不涉及向量库内容检索。
    """

    id: str = Field(..., description="样本唯一 ID，建议 name-res-001 这种格式")
    query: str = Field(..., description="包含基金名称/别名/模糊描述的查询文本")
    top_k: int = Field(5, description="最多返回几个候选基金")
    min_score: float = Field(0.5, description="最低置信度阈值")

    # ---- ground truth ----
    expected_fund_code: Optional[str] = Field(
        None, description="期望识别到的基金代码；None 表示期望不命中（如描述过于模糊/无对应基金）"
    )

    # ---- 元数据 ----
    category: Optional[str] = Field(
        None, description="样本类别，如 exact_name / alias / vague_description / no_match"
    )
    note: Optional[str] = Field(None, description="标注说明")


# ============ 回答评测样本 ============
class AnswerExample(BaseModel):
    """Agent 端到端回答评测样本。"""

    id: str = Field(..., description="样本唯一 ID，建议 answer-001")
    query: str = Field(..., description="用户查询")

    # ---- ground truth ----
    reference_answer: str = Field(
        "", description="参考答案；敏感问题样本可留空，靠 should_refuse 判定"
    )
    expected_fund_codes: list[str] = Field(
        default_factory=list,
        description="期望出现在回答引用中的基金代码（用于 citation_accuracy）",
    )
    key_facts: list[str] = Field(
        default_factory=list,
        description="必须出现在回答中的关键事实点（用于结构化 correctness）",
    )
    should_refuse: bool = Field(
        False, description="若为 True，期望 Agent 拒绝回答（敏感问题）"
    )
    relevant_keywords: list[str] = Field(
        default_factory=list,
        description="可选：检索结果应包含的关键词（用于 Agent RAG 的弱标注）",
    )
    relevant_chunk_ids: list[str] = Field(
        default_factory=list,
        description="Agent 完整链路应召回的 Milvus chunk 主键，用于与直接 RAG 对比",
    )

    # ---- 可选：工具调用准确率评测 ----
    expected_tool_calls: list[dict] = Field(
        default_factory=list,
        description=(
            "可选：期望被调用的工具列表，每项形如 {\"name\": str, \"args\": dict}"
            "（用于 tool_call_accuracy；args 只需列出关键参数，不要求列全）"
        ),
    )

    # ---- 元数据 ----
    intent: Optional[
        Literal[
            "chitchat",
            "out_of_scope",
            "sensitive",
            "fund_query",
            "cross_fund_query",
            "fund_screening",
            "general_finance",
        ]
    ] = Field(None, description="期望的路由意图，用于路由准确率统计")
    category: Optional[str] = Field(None)
    note: Optional[str] = Field(None)
