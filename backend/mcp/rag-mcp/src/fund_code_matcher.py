"""基金代码字符串匹配兜底

为 rag_match_fund_codes 工具提供纯规则匹配（6位代码 + 精确别名子串），
作为 rag_identify_funds 语义识别失败时的兜底方案。
"""
import re
from dataclasses import dataclass


@dataclass
class FundInfo:
    """基金信息"""
    code: str       # 基金代码，如 "161725"
    name: str       # 完整名称
    short_name: str # 简短名称


class FundCodeMatcher:
    """基金代码字符串匹配器：code ↔ name 双向索引 + 别名匹配"""

    def __init__(self, funds: list[FundInfo]) -> None:
        self.funds = funds
        self.by_code = {f.code: f for f in funds}

        # 构建别名索引：名称 → code（小写+去空格，按长度降序避免短名误匹配）
        self._alias_map: dict[str, str] = {}
        for f in funds:
            for alias in [f.name, f.short_name]:
                norm = self._normalize(alias)
                if norm and norm not in self._alias_map:
                    self._alias_map[norm] = f.code

        # 按长度降序（长名优先匹配）
        self._alias_sorted = sorted(self._alias_map.keys(), key=len, reverse=True)

    @staticmethod
    def _normalize(s: str) -> str:
        """规范化：小写 + 去除空格/下划线/括号差异"""
        return (
            s.replace(" ", "")
            .replace("_", "")
            .replace("（", "(")
            .replace("）", ")")
            .lower()
        )

    def match_codes(self, query: str) -> list[str]:
        """从用户问题中抽取基金代码，规则优先，按顺序去重

        匹配规则：
        1. 正则匹配 6 位数字（\\b\\d{6}\\b）
        2. 别名匹配（最长优先，精确边界）

        Args:
            query: 用户查询文本

        Returns:
            匹配到的基金代码列表（按出现顺序）
        """
        codes = []
        seen = set()

        # 规则 1: 6 位数字
        for m in re.finditer(r"\b(\d{6})\b", query):
            code = m.group(1)
            if code in self.by_code and code not in seen:
                codes.append(code)
                seen.add(code)

        # 规则 2: 别名匹配（最长优先，避免子串误匹配）
        norm_q = self._normalize(query)
        matched_positions = []  # 记录已匹配的位置范围，避免重叠

        for alias in self._alias_sorted:
            # 查找所有出现位置
            start = 0
            while True:
                pos = norm_q.find(alias, start)
                if pos == -1:
                    break

                end = pos + len(alias)

                # 检查是否与已匹配区域重叠
                overlaps = any(
                    not (end <= m_start or pos >= m_end)
                    for m_start, m_end in matched_positions
                )

                if not overlaps:
                    code = self._alias_map[alias]
                    if code not in seen:
                        codes.append(code)
                        seen.add(code)
                        matched_positions.append((pos, end))
                        break  # 找到一个就够了，继续下一个 alias

                start = pos + 1

        return codes

    def get_fund(self, code: str) -> FundInfo | None:
        """根据代码获取基金信息"""
        return self.by_code.get(code)


async def load_fund_code_matcher(rag_client) -> FundCodeMatcher:
    """从 RAG 服务加载基金清单，构建字符串匹配器

    Args:
        rag_client: RagClient 实例

    Returns:
        FundCodeMatcher 实例
    """
    raw = await rag_client.list_funds()
    funds = []

    for item in raw:
        code = item.get("code", "").strip()
        name = item.get("name", "").strip()
        if not code:
            continue

        # 简化名称：取 "_" 前的部分作为 short_name
        short_name = name.split("_")[0] if "_" in name else name
        funds.append(FundInfo(code=code, name=name, short_name=short_name))

    return FundCodeMatcher(funds)
