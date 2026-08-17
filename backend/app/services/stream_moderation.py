"""Low-latency moderation for text chunks sent to the chat client."""
from __future__ import annotations

import re
from dataclasses import dataclass


STREAM_MODERATION_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"(?:建议|推荐).{0,12}(?:买入|卖出|加仓|减仓|配置|投资)", "检测到直接投资建议"),
    (r"(?:买入|卖出|加仓|减仓|止盈|止损).{0,12}(?:基金|ETF|股票|仓位)", "检测到买卖操作指导"),
    (r"(?:高抛低吸|值得投资|适合投资|推荐购买)", "检测到投资引导措辞"),
    (r"(?:预期|预计|有望|将会).{0,12}(?:上涨|下跌|收益|回报)", "检测到收益或走势预期"),
    (r"(?:收益率|回报率).{0,8}\d+(?:\.\d+)?\s*%", "检测到具体收益率预测"),
)


@dataclass(frozen=True)
class ModerationHit:
    reason: str
    matched_text: str


class StreamModerationBuffer:
    def __init__(self, window_chars: int = 80, overlap_chars: int = 24) -> None:
        if window_chars <= 0:
            raise ValueError("window_chars must be positive")
        if overlap_chars < 0 or overlap_chars >= window_chars:
            raise ValueError("overlap_chars must be in [0, window_chars)")
        self.window_chars = window_chars
        self.overlap_chars = overlap_chars
        self._buffer = ""
        self._patterns = tuple(
            (re.compile(pattern, re.IGNORECASE), reason)
            for pattern, reason in STREAM_MODERATION_PATTERNS
        )

    def _find_hit(self) -> ModerationHit | None:
        for pattern, reason in self._patterns:
            match = pattern.search(self._buffer)
            if match:
                return ModerationHit(reason=reason, matched_text=match.group(0))
        return None

    def feed(self, text: str) -> tuple[str, ModerationHit | None]:
        if not text:
            return "", None
        self._buffer += text
        hit = self._find_hit()
        if hit:
            return "", hit

        if len(self._buffer) <= self.window_chars:
            return "", None

        release_len = len(self._buffer) - self.overlap_chars
        released = self._buffer[:release_len]
        self._buffer = self._buffer[release_len:]
        return released, None

    def flush(self) -> tuple[str, ModerationHit | None]:
        hit = self._find_hit()
        if hit:
            return "", hit
        released = self._buffer
        self._buffer = ""
        return released, None
