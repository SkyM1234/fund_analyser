"""Synthesizer - 汇总所有子任务结果"""
import logging
from typing import Any
from datetime import datetime

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

from app.core.config import get_settings
from app.core.llm_concurrency import llm_ainvoke
from app.agent.multi_agent_state import MultiAgentState, PlanExecution
from app.tools.token_usage import record_usage

logger = logging.getLogger(__name__)


SYNTHESIZER_SYSTEM_PROMPT = """你是基金分析系统的结果汇总专家。

你的职责：
1. 整合多个子任务的结果
2. 生成连贯、完整的最终答案
3. 确保答案结构清晰、易读

输入：
- 用户原始问题
- 各个子任务的执行结果

输出要求：
1. **完整性**：覆盖所有子任务的关键信息
2. **连贯性**：不要简单罗列，要自然组织
3. **可读性**：使用标题、列表、表格等结构化格式
4. **引用**：标注数据来源，如 [159103]
5. **简洁性**：去除冗余，突出重点

格式建议：
- 单基金查询：分段叙述（概况→策略→持仓→业绩）
- 多基金查询：使用表格或对比列表展示差异，突出关键指标
- 数据+分析：先数据后分析

⚠️ 注意：
- 不要添加投资建议或推荐
- 若子任务失败，需在答案中说明
- 保持专业、客观的语气
"""


async def synthesizer_node(state: MultiAgentState) -> dict[str, Any]:
    """Synthesizer 节点：汇总所有子任务结果生成最终答案

    若上一轮合规检查未通过（compliance_passed is False），本次会带着
    compliance_reason 重新生成答案，要求换一种表达方式规避违规措辞，
    而不是重新检索/重新执行子任务。
    """

    # 获取用户问题
    messages = state["messages"]
    user_query = None
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            user_query = msg.content
            break

    if not user_query:
        logger.warning("[Synthesizer] No user query found")
        return {"final_answer": "无法生成答案：未找到用户问题"}

    # 合规重试反馈：若是因合规不通过被打回，附加改写指导
    compliance_feedback = ""
    if state.get("compliance_passed") is False and state.get("compliance_reason"):
        compliance_feedback = f"""

⚠️ 上一版答案未通过合规审查，原因：{state['compliance_reason']}
请修改表达方式，重新生成答案。"""
    
    # 获取所有子任务结果
    plan = state.get("plan", [])
    sub_results = state.get("sub_results", {})
    plan_history = state.get("plan_history", [])  # 获取历史执行记录

    # 如果当前没有plan，检查是否有历史可以参考
    if not plan:
        logger.warning("[Synthesizer] No plan found for current round")

        # 如果有历史，基于历史回答
        if plan_history:
            logger.info("[Synthesizer] Using plan_history to answer")
            # 构造基于历史的上下文
            history_context = "\n\n历史对话记录：\n"
            recent_history = plan_history[-3:]  # 最近3轮
            for i, record in enumerate(recent_history, 1):
                history_context += f"""
第 {i} 轮:
用户问题: {record.get('user_query', '')}
执行了 {len(record.get('plan', []))} 个任务
结果摘要: {record.get('final_answer', '')[:300]}
---
"""

            # 调用LLM基于历史回答
            settings = get_settings()
            llm = ChatOpenAI(
                base_url=settings.LLM_BASE_URL,
                api_key=settings.LLM_API_KEY,
                model=settings.LLM_MODEL,
                temperature=0.7,
            )

            prompt = f"""历史对话上下文：
{history_context}

当前用户问题：
{user_query}

请基于上述历史对话，回答当前问题。如果当前问题与历史相关，请利用历史信息；如果完全无关，请直接回答。{compliance_feedback}"""

            try:
                response = await llm_ainvoke(llm, [
                    {"role": "system", "content": SYNTHESIZER_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ])

                final_answer = response.content
                token_usage = record_usage("synthesizer:history_fallback", response)

                # 记录到历史
                updated_plan_history = plan_history + [PlanExecution(
                    round_id=f"round_{len(plan_history) + 1}",
                    user_query=user_query,
                    plan=[],
                    results={},
                    final_answer=final_answer,
                    timestamp=datetime.now().isoformat(),
                )]

                return {
                    "final_answer": final_answer,
                    "synthesis_complete": True,
                    "plan_history": updated_plan_history,
                    "compliance_passed": True,
                    "compliance_reason": None,
                    "token_usage": token_usage,
                }
            except Exception as e:
                logger.error(f"[Synthesizer] Failed with history: {e}")
                return {
                    "final_answer": f"抱歉，生成答案时出错：{str(e)}",
                    "synthesis_complete": True,
                    "compliance_passed": True,
                    "compliance_reason": None,
                }

        else:
            # 完全没有历史，只能直接回答
            logger.warning("[Synthesizer] No plan_history, direct LLM fallback")
            settings = get_settings()
            llm = ChatOpenAI(
                base_url=settings.LLM_BASE_URL,
                api_key=settings.LLM_API_KEY,
                model=settings.LLM_MODEL,
                temperature=0.7,
            )

            try:
                response = await llm_ainvoke(llm, [
                    {"role": "user", "content": user_query + compliance_feedback}
                ])

                final_answer = response.content
                token_usage = record_usage("synthesizer:no_history_fallback", response)

                # 记录到历史
                updated_plan_history = [PlanExecution(
                    round_id="round_1",
                    user_query=user_query,
                    plan=[],
                    results={},
                    final_answer=final_answer,
                    timestamp=datetime.now().isoformat(),
                )]

                return {
                    "final_answer": final_answer,
                    "synthesis_complete": True,
                    "plan_history": updated_plan_history,
                    "compliance_passed": True,
                    "compliance_reason": None,
                    "token_usage": token_usage,
                }
            except Exception as e:
                logger.error(f"[Synthesizer] Fallback failed: {e}")
                return {
                    "final_answer": f"抱歉，生成答案时出错：{str(e)}",
                    "synthesis_complete": True,
                }
    
    # 构造汇总上下文
    # 注意：只暴露任务描述和自然语言状态说明给 LLM，不透传 task_id / 英文状态字面量，
    # 避免 LLM 在最终答案里直译出"任务 t2""状态为 failed"这类内部实现细节。
    status_labels = {
        "completed": "已成功获取",
        "failed": "未能成功获取（以下为部分或兜底信息，可能不完整或不准确）",
        "pending": "未执行",
        "running": "执行中",
    }
    context_parts = []
    for task in plan:
        task_id = task["task_id"]
        description = task["description"]
        status_label = status_labels.get(task["status"], task["status"])
        result = sub_results.get(task_id, "未执行")

        context_parts.append(f"""
子任务：{description}
获取情况：{status_label}
结果：
{result}
""")

    context = "\n---\n".join(context_parts)

    # 构造冲突标注上下文（供 Synthesizer 按标记披露）；不透传内部 conflict_id/task_ids，
    # 只给字段名和详情文本，避免直译成用户可见的内部编号
    conflict_annotations = state.get("conflict_annotations", [])
    conflict_context = ""
    if conflict_annotations:
        conflict_lines = []
        for ann in conflict_annotations:
            risk_label = "⚠️ 高风险" if ann.get("risk") == "high" else "ℹ️ 低风险"
            resolved = ann.get("resolved", False)
            status_label = "（已通过仲裁消解）" if resolved else "（未消解，需在答案中向用户说明差异）"
            conflict_lines.append(
                f"字段「{ann.get('field', '?')}」{risk_label}{status_label}\n"
                f"  详情：{ann.get('description', '')}"
            )
        conflict_context = "\n\n⚠️ 全局反思标注的数据冲突（必须在答案中说明，不得隐藏）：\n" + "\n".join(conflict_lines)

    # 构造历史上下文（如果有）
    history_context = ""
    if plan_history:
        history_context = "\n\n历史对话执行记录（可参考上下文）：\n"
        # 只保留最近2轮历史，避免过长
        recent_history = plan_history[-2:]
        for i, record in enumerate(recent_history, 1):
            history_context += f"""
第 {i} 轮 ({record.get('round_id', 'unknown')}):
用户问题: {record.get('user_query', '')[:100]}
执行了 {len(record.get('plan', []))} 个任务
回答: {record.get('final_answer', '')[:200]}...
"""

    # 调用 LLM 汇总
    settings = get_settings()
    llm = ChatOpenAI(
        base_url=settings.LLM_BASE_URL,
        api_key=settings.LLM_API_KEY,
        model=settings.LLM_MODEL,
        temperature=0.5,
    )

    prompt = f"""用户问题：
{user_query}
{history_context}
当前轮次各子任务执行结果：
{context}
{conflict_context}

请整合上述结果，生成一个完整、连贯、易读的答案。如果有历史对话，请利用历史信息提供更连贯的回答。
如果有冲突标注，须在答案对应位置明确说明数据存在差异（如"两个来源数据不一致，数据 A 为 X，数据 B 为 Y，建议以官方公告为准"），不得选择性忽略。{compliance_feedback}"""

    try:
        response = await llm_ainvoke(llm, [
            {"role": "system", "content": SYNTHESIZER_SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ])
        
        final_answer = response.content
        token_usage = record_usage("synthesizer", response)
        logger.info(f"[Synthesizer] Generated answer ({len(final_answer)} chars)")

        # 保存本轮执行记录到历史
        plan_history = state.get("plan_history", [])
        current_round = PlanExecution(
            round_id=f"round_{len(plan_history) + 1}",
            user_query=user_query,
            plan=plan,
            results=sub_results.copy(),
            final_answer=final_answer,
            timestamp=datetime.now().isoformat(),
        )
        updated_plan_history = plan_history + [current_round]

        return {
            "final_answer": final_answer,
            "synthesis_complete": True,
            "plan_history": updated_plan_history,  # 更新历史
            "compliance_passed": True,
            "compliance_reason": None,
            "token_usage": token_usage,
        }

    except Exception as e:
        logger.error(f"[Synthesizer] Failed to synthesize: {e}")
        # 降级：拼接所有结果
        fallback_parts = [f"针对您的问题：{user_query}\n"]
        for task in plan:
            task_id = task["task_id"]
            result = sub_results.get(task_id, "未执行")
            fallback_parts.append(f"\n**{task['description']}**\n{result}\n")
        
        fallback_answer = "\n".join(fallback_parts)

        # 保存历史（即使失败也记录）
        plan_history = state.get("plan_history", [])
        current_round = PlanExecution(
            round_id=f"round_{len(plan_history) + 1}",
            user_query=user_query,
            plan=plan,
            results=sub_results.copy(),
            final_answer=fallback_answer,
            timestamp=datetime.now().isoformat(),
        )
        updated_plan_history = plan_history + [current_round]

        return {
            "final_answer": fallback_answer,
            "synthesis_complete": True,
            "plan_history": updated_plan_history,
            "compliance_passed": True,
            "compliance_reason": None,
        }
