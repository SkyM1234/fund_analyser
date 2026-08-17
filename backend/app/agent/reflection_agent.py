"""Reflection Agent - 反思与质量控制"""
import logging
import json
import re
import uuid
from typing import Any

from langchain_openai import ChatOpenAI

from app.core.config import get_settings
from app.core.llm_concurrency import llm_ainvoke
from app.agent.multi_agent_state import (
    MultiAgentState,
    SubTask,
    ConflictAnnotation,
    is_plan_complete,
)
from app.tools.llm_json import extract_json_block
from app.tools.token_usage import record_usage

logger = logging.getLogger(__name__)


# 反思配置
MAX_REFLECTION_RETRIES = 2  # 单个任务最多图层重试次数
MAX_CLARIFICATION_ROUNDS = 1  # 全局反思最多触发一次定向澄清


GLOBAL_REFLECTION_SYSTEM_PROMPT = """你是基金分析系统的全局质量裁判，负责跨 Agent 结果的冲突识别与定级。

你的职责（严格限制）：
1. 识别不同 Agent 结果之间的数据冲突、逻辑矛盾、口径不一致
2. 对冲突进行风险定级（low / high）
3. 对高风险冲突决策是否需要定向澄清任务

绝对禁止：
- 修改任何 Agent 的原始数据或结论
- 凭空补充未经检索的数据
- 对低风险冲突触发额外调度

风险定级标准：
- **high**（需要澄清）：核心结论级冲突，如同一基金净值相差 >5%、业绩正负相反、同一指标数值矛盾
- **low**（标注披露）：非核心冲突，如数据日期不同、口径说明不同、描述风格差异

数据源判断依据：
- 每条 Agent 结果末尾应有 [基金代码] 或数据时间标注
- 若无标注，默认来源不明，降级为 low

失败任务处理：
- 若某条结果标注为"执行失败"，其内容可能只是部分数据或兜底文案，不能直接当作与成功任务对等的事实来源
- 若失败一方与成功一方的核心结论不一致：优先判定为 low（标注"该来源执行失败，以成功来源为准"），除非成功一方本身也标注了明显的不确定性或数据缺口
- 若两方都失败，或失败方缺口正是用户提问的关键信息（如完全没有覆盖到历史数据），应在 description 中明确指出信息缺失，不要虚构对比结论

输出格式（JSON）：
```json
{
  "conflicts": [
    {
      "conflict_id": "c1",
      "risk": "high",
      "task_ids": ["t1", "t2"],
      "field": "净值",
      "description": "t1 报告净值 1.23，t2 报告净值 1.45，相差 18%",
      "clarification_query": "请重新确认基金 159103 截至最新日期的准确净值，来源须标注数据日期"
    }
  ],
  "summary": "发现 N 个冲突，其中 M 个高风险"
}
```

clarification_query 只在 risk=high 时填写，low 时留空字符串。
若无冲突，conflicts 返回空列表。
"""


async def global_reflection_node(state: MultiAgentState) -> dict[str, Any]:
    """第二层全局反思节点。

    职责：冲突识别 + 定级 + 策略决策（调度权 + 标注权）。
    严格禁止：直接修改 sub_results 中的任何 Agent 原始数据。

    流程：
    1. 计划未全部进入终态时，不做反思，直接放行下一调度批次
    2. 原始任务全部终态后，统一比较所有尚未反思的原始任务结果
    3. low 冲突 → 写入 conflict_annotations（resolved=False），交 Synthesizer 披露
    4. high 冲突 → 写入 conflict_annotations + 生成定向澄清 SubTask（clarify_ 前缀）
    5. 澄清任务完成后仅回写既有冲突的 resolved 状态，不再次全量比对
    6. 已达 clarification_round 上限时，high 冲突强制降级为 low（标注披露）
    """
    plan = list(state.get("plan", []))
    sub_results = state.get("sub_results", {})
    clarification_round = state.get("clarification_round", 0)

    # batch_reflection 是每个调度批次后的汇合点，但跨批次依赖任务只能在整个
    # DAG 结束后才能做完整一致性比较。中间批次绝不能提前写 reflected=True，
    # 否则后续批次的结果不会再与上游结果进入同一次比较。
    if not is_plan_complete(state):
        pending_or_running = [
            t["task_id"]
            for t in plan
            if t["status"] in ("pending", "running")
        ]
        logger.info(
            "[GlobalReflection] Plan not complete; deferring full consistency check. "
            "Outstanding tasks: %s",
            pending_or_running,
        )
        return {}

    # 仅原始任务参与一次完整一致性检查。澄清任务负责裁决既有冲突，不应成为
    # 新一轮冲突识别的输入，避免澄清结果与原始结果被重复比较。
    new_tasks = [
        t for t in plan
        if (
            t["status"] in ("completed", "failed")
            and not t.get("reflected", False)
            and not t["task_id"].startswith("clarify_")
        )
    ]

    # 只有真正进入最终检查的原始任务才标记为已反思。
    new_task_ids = {task["task_id"] for task in new_tasks}
    for t in plan:
        if t["task_id"] in new_task_ids:
            t["reflected"] = True

    total_token_usage: dict[str, dict[str, int]] = {}

    # 澄清任务通常会在原始任务的一次完整检查之后单独完成。它们不参与二次
    # 冲突识别，只用于回写既有 conflict_annotations 的 resolved 状态。
    resolved_updates: list[ConflictAnnotation] = []
    newly_completed_clarify_ids = {
        t["task_id"]
        for t in plan
        if (
            t["task_id"].startswith("clarify_")
            and t["status"] in ("completed", "failed")
            and not t.get("reflected", False)
        )
    }
    if newly_completed_clarify_ids:
        for ann in state.get("conflict_annotations", []):
            clarify_id = ann.get("clarification_task_id")
            if clarify_id and clarify_id in newly_completed_clarify_ids and not ann.get("resolved", False):
                clarify_task = next((t for t in plan if t["task_id"] == clarify_id), None)
                clarify_result = sub_results.get(clarify_id, "")

                # arbiter_agent 的输出以 "[裁决:verdict] ..." 为前缀，verdict 为
                # adopt_a / adopt_b / not_conflicting 时视为已消解；unresolved 或执行失败则保留未消解，
                # 交由 Synthesizer 向用户披露双方差异。
                verdict_match = re.match(r"\[裁决:(\w+)\]", clarify_result)
                verdict = verdict_match.group(1) if verdict_match else None
                is_resolved = bool(
                    clarify_task
                    and clarify_task.get("status") == "completed"
                    and verdict in ("adopt_a", "adopt_b", "not_conflicting")
                )

                new_ann = dict(ann)
                new_ann["resolved"] = is_resolved
                if clarify_result:
                    new_ann["description"] = f"{ann.get('description', '')}\n仲裁结论：{clarify_result}"
                resolved_updates.append(new_ann)
                logger.info(f"[GlobalReflection] Clarification {clarify_id} verdict={verdict} resolved={is_resolved} for conflict {ann.get('conflict_id')}")

    for t in plan:
        if t["task_id"] in newly_completed_clarify_ids:
            t["reflected"] = True

    # 澄清完成后的终态批次，或只有一个原始任务时，无需调用 LLM 做跨任务比较。
    if len(new_tasks) <= 1:
        logger.info(
            "[GlobalReflection] %s original task(s) awaiting final check; "
            "skipping cross-task comparison",
            len(new_tasks),
        )
        return {
            "plan": plan,
            "reflection_count": state.get("reflection_count", 0) + 1,
            **({"conflict_annotations": resolved_updates} if resolved_updates else {}),
        }

    # 构造跨任务结果上下文，供 LLM 分析；标明任务状态，避免 failed 任务的兜底文案
    # 被误当作正常事实数据参与冲突比对
    context_parts = []
    for t in new_tasks:
        tid = t["task_id"]
        result = sub_results.get(tid, "（无结果）")
        status_note = "（正常完成）" if t["status"] == "completed" else "（⚠️ 执行失败，以下可能是部分/兜底结果，不可直接当作事实）"
        context_parts.append(f"[{tid}] 描述：{t.get('description', '')}  基金：{t.get('fund_codes', [])}  状态：{t['status']}{status_note}\n结果：\n{result}")
    cross_context = "\n\n---\n\n".join(context_parts)

    user_query = ""
    from langchain_core.messages import HumanMessage
    for msg in reversed(state.get("messages", [])):
        if isinstance(msg, HumanMessage):
            user_query = msg.content
            break

    prompt = f"""用户原始问题：{user_query}

各 Agent 执行结果（共 {len(new_tasks)} 条）：

{cross_context}

请识别上述结果之间的冲突并定级。"""

    settings = get_settings()
    llm = ChatOpenAI(
        base_url=settings.LLM_BASE_URL,
        api_key=settings.LLM_API_KEY,
        model=settings.LLM_MODEL,
        temperature=0.1,
    )

    new_annotations: list[ConflictAnnotation] = []
    clarification_tasks: list[SubTask] = []

    try:
        response = await llm_ainvoke(llm, [
            {"role": "system", "content": GLOBAL_REFLECTION_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ])
        usage = record_usage("global_reflection", response)
        for k, v in usage.items():
            existing = total_token_usage.setdefault(k, {})
            for field, val in v.items():
                existing[field] = existing.get(field, 0) + val

        content = extract_json_block(response.content)
        result_data = json.loads(content)
        conflicts = result_data.get("conflicts", [])
        logger.info(f"[GlobalReflection] Found {len(conflicts)} conflict(s): {result_data.get('summary', '')}")

        for c in conflicts:
            risk = c.get("risk", "low")
            cid = c.get("conflict_id") or f"c_{uuid.uuid4().hex[:6]}"

            # 已达澄清轮次上限，high 强制降级为 low
            if risk == "high" and clarification_round >= MAX_CLARIFICATION_ROUNDS:
                logger.warning(f"[GlobalReflection] Conflict {cid} downgraded to low (clarification_round={clarification_round})")
                risk = "low"

            annotation = ConflictAnnotation(
                conflict_id=cid,
                risk=risk,
                task_ids=c.get("task_ids", []),
                field=c.get("field", ""),
                description=c.get("description", ""),
                resolved=False,
                clarification_task_id=None,
            )

            if risk == "high":
                # 生成定向澄清任务，统一交由中立的 arbiter_agent 裁决
                clarify_query = c.get("clarification_query", "")
                if clarify_query:
                    ref_task = next((t for t in new_tasks if t["task_id"] in c.get("task_ids", [])), new_tasks[0])
                    clarify_id = f"clarify_{cid}"

                    # 拼接冲突双方的原始结果，供仲裁 agent 交叉核实，而非交给某一方孤立重新查询
                    conflict_task_ids = c.get("task_ids", [])
                    source_blocks = []
                    merged_fund_codes: list[str] = []
                    for tid in conflict_task_ids:
                        src_task = next((t for t in plan if t["task_id"] == tid), None)
                        agent_name = src_task.get("assigned_agent", "?") if src_task else "?"
                        task_status = src_task.get("status", "?") if src_task else "?"
                        original_result = sub_results.get(tid, "（无结果）")
                        source_blocks.append(f"[原始结果 {tid} | 来源agent: {agent_name} | 任务状态: {task_status}]\n{original_result}")
                        if src_task:
                            for fc in src_task.get("fund_codes", []):
                                if fc not in merged_fund_codes:
                                    merged_fund_codes.append(fc)
                    sources_context = "\n\n".join(source_blocks)
                    logger.info(
                        f"[GlobalReflection] Built clarify context for {cid}: "
                        f"{len(conflict_task_ids)} source task(s) {conflict_task_ids}, "
                        f"{len(sources_context)} chars total"
                    )

                    enriched_query = (
                        f"以下是关于「{c.get('field', '')}」的冲突背景，请交叉核实并给出准确结论：\n\n"
                        f"{sources_context}\n\n"
                        f"冲突描述：{c.get('description', '')}\n\n"
                        f"澄清任务：{clarify_query}"
                    )

                    clarify_task = SubTask(
                        task_id=clarify_id,
                        task_type=ref_task.get("task_type", "rag_search"),
                        description=f"[定向澄清] {c.get('field', '')} 冲突消解",
                        assigned_agent="arbiter_agent",
                        fund_codes=merged_fund_codes or ref_task.get("fund_codes", []),
                        query=enriched_query,
                        depends_on=[],
                        status="pending",
                        result=None,
                        error=None,
                        retry_count=0,
                        reflected=False,
                    )
                    clarification_tasks.append(clarify_task)
                    annotation["clarification_task_id"] = clarify_id
                    logger.info(f"[GlobalReflection] Scheduling clarification task {clarify_id} for conflict {cid}")

            new_annotations.append(annotation)

    except Exception as e:
        logger.error(f"[GlobalReflection] LLM conflict detection failed: {e}")
        # 降级：无冲突处理，直接放行
        return {
            "plan": plan,
            "reflection_count": state.get("reflection_count", 0) + 1,
            "token_usage": total_token_usage,
            **({"conflict_annotations": resolved_updates} if resolved_updates else {}),
        }

    # 将澄清任务追加到 plan
    if clarification_tasks:
        plan.extend(clarification_tasks)
        clarification_round += 1

    return {
        "plan": plan,
        "conflict_annotations": new_annotations + resolved_updates,
        "clarification_round": clarification_round,
        "reflection_count": state.get("reflection_count", 0) + 1,
        "token_usage": total_token_usage,
    }


async def _improve_query(task: SubTask, reason: str) -> tuple[str, dict[str, dict[str, int]]]:
    """基于反思结果改进查询

    返回 (改进后的查询, token 用量增量)。
    """

    original_query = task.get("query", "")
    description = task["description"]

    settings = get_settings()
    llm = ChatOpenAI(
        base_url=settings.LLM_BASE_URL,
        api_key=settings.LLM_API_KEY,
        model=settings.LLM_MODEL,
        temperature=0.3,
    )

    prompt = f"""原始查询失败了，需要改进查询策略。

任务描述：{description}
原始查询：{original_query}
失败原因：{reason}

请生成一个改进的查询，使其更有可能获得充分的结果。改进策略：
1. 更具体的关键词
2. 补充相关术语
3. 调整查询角度

只输出改进后的查询文本，不要解释。"""

    try:
        response = await llm_ainvoke(llm, [
            {"role": "user", "content": prompt}
        ])

        improved = response.content.strip()
        token_usage = record_usage(f"reflection_rewrite:{task['task_id']}", response)
        return (improved if improved else original_query, token_usage)

    except Exception as e:
        logger.error(f"[Reflection] Query improvement failed: {e}")
        # 降级：添加更多关键词
        return (f"{original_query} 详细信息", {})


# ===== 第一层：Agent 内置自检（供 rag_agent / market_agent 调用）=====

AGENT_SELF_CHECK_PROMPT = """你是输出质量自检专家，负责检查单个 Agent 的输出是否合格。

范围控制：
- 若输出加入了与子任务无直接关系的背景、指标、比较、建议或延伸分析，应删除这些内容。
- 信息已经足以回答子任务时，不得因“更全面”而要求继续检索。

检查维度（逐项评估）：
1. **格式合规**：输出是否结构清晰、有数据来源标注（如 [基金代码] 或数据时间）
2. **字数合理**：不少于 50 字，不超过 800 字
3. **子任务覆盖**：是否覆盖了分配的子任务全部要求，无明显遗漏
4. **无明显错误**：无语法错误、无自相矛盾、无明显计算错误
5. **事实一致性**：数据之间是否存在明显矛盾（如同一指标两个数值不一致、涨跌幅正负号相反、不同时间口径混用）
6. **无敏感内容**：不包含投资建议、收益预测、买卖推荐

输出格式（JSON）：
```json
{
  "passed": true/false,
  "score": 0.0-1.0,
  "issues": ["问题描述1"],
  "improved_output": "改写后的输出（仅当 passed=false 且 needs_recheck=false 时填写）",
  "needs_recheck": true/false,
  "recheck_query": "需要重新查证的具体查询（仅 needs_recheck=true 时填写）"
}
```

规则：
- **事实矛盾**（needs_recheck=true）：数据间存在无法通过文本改写修复的数值冲突，必须由 Agent 重新调用工具查证
  - recheck_query 必须包含具体的查证指令，如"请重新查询 005827 的近1年收益率数据，与已获取的 -18.92% 交叉验证，确认准确值"
  - 此时 passed 必须为 false，improved_output 留空字符串
- **零工具调用/未检索直接拒答**（needs_recheck=true，优先于"格式/遗漏问题"判定）：若输出内容是反问用户要基金代码、声称"未能识别到基金代码"、"无法获取信息"等，但 Agent 本次全程未调用任何工具（未见工具调用记录/未见检索结果），一律视为事实矛盾类问题，**禁止**当作格式/遗漏问题用 improved_output 改写措辞了事——因为改写只能让拒答的话术更委婉，无法凭空补上缺失的检索结果
  - recheck_query 必须明确指示"必须先调用识别/搜索工具确认基金代码，再基于结果回答"，如"请先调用基金识别工具确认'万家科创债ETF'对应的基金代码，再回答基金合同生效日"
  - 此时 passed 必须为 false，improved_output 留空字符串
- **格式/遗漏问题**（needs_recheck=false, passed=false）：Agent 已调用过工具、只是输出的文本存在格式/措辞/遗漏问题时，在 improved_output 里改写，只改格式/遗漏/敏感措辞，不能增加未经检索的新信息
- **通过**（needs_recheck=false, passed=true）：直接输出原文，improved_output 留空字符串
- 改写次数已达上限时，passed 强制为 true 以避免无限循环（但"零工具调用"类问题达到改写上限时也不应放行，应保持 needs_recheck 语义，由调用方的重试预算控制）
"""


async def agent_self_check(
    task: "SubTask",
    result: str,
    agent_label: str,
    *,
    max_self_retries: int = 1,
) -> tuple[str, dict[str, dict[str, int]], str | None]:
    """对单个 Agent 的输出做一次轻量自检。

    返回 (最终输出文本, token_usage 增量, recheck_query | None)。
    - recheck_query 非空：存在事实性数据矛盾，调用方应追加工具查证循环
    - recheck_query 为 None：文本级自检完成（通过或改写）
    若 LLM 调用失败则直接返回原始 result。
    max_self_retries 控制最多改写次数（默认 1），超过则不再改写。
    """
    import json as _json

    settings = get_settings()
    llm = ChatOpenAI(
        base_url=settings.LLM_BASE_URL,
        api_key=settings.LLM_API_KEY,
        model=settings.LLM_MODEL,
        temperature=0.1,
    )

    current = result
    total_usage: dict[str, dict[str, int]] = {}

    for attempt in range(max_self_retries + 1):
        prompt = f"""子任务描述：{task.get('description', '')}
子任务查询：{task.get('query', '')}
已尝试改写次数：{attempt}（上限 {max_self_retries}）

Agent 输出：
{current}

请对以上输出进行自检。"""

        try:
            response = await llm_ainvoke(llm, [
                {"role": "system", "content": AGENT_SELF_CHECK_PROMPT},
                {"role": "user", "content": prompt},
            ])
            usage = record_usage(f"{agent_label}_self_check:{task['task_id']}:{attempt}", response)
            for k, v in usage.items():
                existing = total_usage.setdefault(k, {})
                for field, val in v.items():
                    existing[field] = existing.get(field, 0) + val

            content = extract_json_block(response.content)
            assessment = _json.loads(content)

            # 事实矛盾 → 返回 recheck_query，由调用方追加工具查证
            if assessment.get("needs_recheck"):
                recheck_query = assessment.get("recheck_query", "")
                if recheck_query:
                    logger.info(
                        f"[{agent_label}] Self-check requested recheck (attempt={attempt}), "
                        f"query: {recheck_query[:100]}, "
                        f"issues: {assessment.get('issues', [])}"
                    )
                    return (current, total_usage, recheck_query)
                logger.warning(f"[{agent_label}] needs_recheck=true but no recheck_query, falling through")

            if assessment.get("passed", True):
                logger.info(f"[{agent_label}] Self-check passed (attempt={attempt}, score={assessment.get('score', '?')})")
                return (current, total_usage, None)

            improved = assessment.get("improved_output", "").strip()
            if not improved:
                logger.warning(f"[{agent_label}] Self-check failed but no improved_output, keeping original")
                return (current, total_usage, None)

            logger.info(f"[{agent_label}] Self-check rewrote output (attempt={attempt}), issues: {assessment.get('issues', [])}")
            current = improved

        except Exception as e:
            logger.warning(f"[{agent_label}] Self-check error (attempt={attempt}): {e}")
            return (current, total_usage, None)

    return (current, total_usage, None)
