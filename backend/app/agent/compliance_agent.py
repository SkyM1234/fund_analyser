"""Compliance Agent - 合规检查"""
import logging
from typing import Any
import re

from langchain_openai import ChatOpenAI

from app.core.config import get_settings
from app.core.llm_concurrency import llm_ainvoke
from app.agent.multi_agent_state import MultiAgentState
from app.tools.llm_json import extract_json_block
from app.tools.token_usage import record_usage

logger = logging.getLogger(__name__)


COMPLIANCE_SYSTEM_PROMPT = """你是基金分析系统的合规审查专家。

你的职责：
检查最终答案是否触碰投资建议红线。

禁止内容：
1. **具体投资建议** - "建议买入"、"推荐配置"、"现在是入场时机"
2. **收益预测** - "未来会涨"、"预期收益率X%"、"有上涨空间"
3. **基金推荐** - "这只基金更好"、"适合你"、"值得投资"
4. **买卖指导** - "应该卖出"、"可以加仓"、"止盈止损建议"

允许内容：
- 客观数据展示（规模、费率、持仓、历史净值）
- 基金对比分析（不带倾向性）
- 专业概念解释
- 风险提示

判断标准：
- 若答案包含上述禁止内容 → 不通过
- 若仅提供客观信息和数据 → 通过
- 边界模糊时 → 倾向于不通过（宁可误拒）

输出格式（JSON）：
```json
{
  "passed": true/false,
  "reason": "通过/不通过的原因",
  "risk_level": "none/low/high"
}
```
"""

async def compliance_agent_node(state: MultiAgentState) -> dict[str, Any]:
    """Compliance Agent 节点：合规检查"""
    
    final_answer = state.get("final_answer")
    
    if not final_answer:
        logger.info("[Compliance] No final_answer to check")
        return {"compliance_passed": True}
    
    logger.info("[Compliance] Starting compliance check")
    retry_count = state.get("compliance_retry_count", 0)

    # LLM 深度检查（更准确但成本高）
    settings = get_settings()

    # 可选：使用更强的模型做合规判断
    llm = ChatOpenAI(
        base_url=settings.LLM_BASE_URL,
        api_key=settings.LLM_API_KEY,
        model=settings.LLM_MODEL,
        temperature=0.0,  # 确定性输出
    )

    prompt = f"""请审查以下答案是否符合合规要求：

答案内容：
{final_answer}

请判断是否包含投资建议、收益预测、基金推荐或买卖指导。"""

    try:
        response = await llm_ainvoke(llm, [
            {"role": "system", "content": COMPLIANCE_SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ])
        
        content = response.content
        token_usage = record_usage("compliance", response)

        # 解析 JSON
        import json
        content = extract_json_block(content)

        result = json.loads(content)

        passed = result.get("passed", True)
        reason = result.get("reason", "LLM审查完成")
        risk_level = result.get("risk_level", "none")

        logger.info(f"[Compliance] LLM check - passed={passed}, risk={risk_level}")

        if passed:
            return {
                "compliance_passed": True,
                "compliance_reason": reason,
                "token_usage": token_usage,
            }
        return {
            "compliance_passed": False,
            "compliance_reason": reason,
            "compliance_retry_count": retry_count + 1,
            "token_usage": token_usage,
        }

    except Exception as e:
        logger.error(f"[Compliance] LLM check failed: {e}")
        # 降级：严格模式，检查失败视为不通过
        return {
            "compliance_passed": False,
            "compliance_reason": f"合规检查失败（系统错误）：{str(e)}",
            "compliance_retry_count": retry_count + 1,
        }
