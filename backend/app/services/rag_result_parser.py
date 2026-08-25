"""从 rag_search 文本结果中解析结构化的 chunk 元数据。"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class RagSearchChunkSection:
    """rag-mcp 文本响应中的单个 chunk 块。"""

    chunk_id: str
    score: float | None
    text: str

_RESULT_SPLIT_RE = re.compile(r"^--- 结果\s+\d+\s+\(相似度:\s*([^)]+)\)\s+---$", re.MULTILINE)


def parse_rag_search_result(text: str) -> list[dict]:
    """返回 rag-mcp 文本响应中嵌入的、带排名的 chunk 元数据。"""
    matches = list(_RESULT_SPLIT_RE.finditer(text or ""))
    chunks: list[dict] = []

    for index, match in enumerate(matches):
        section_start = match.end()
        section_end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        section = text[section_start:section_end]

        chunk_id = _field(section, "Chunk ID")
        if not chunk_id:
            continue

        chunk: dict = {
            "id": chunk_id,
            "fund_code": _field(section, "基金代码"),
        }
        chunk_index = _field(section, "Chunk Index")
        if chunk_index:
            try:
                chunk["chunk_index"] = int(chunk_index)
            except ValueError:
                pass
        try:
            chunk["score"] = float(match.group(1).strip())
        except ValueError:
            pass
        chunks.append(chunk)

    return chunks


def parse_rag_search_sections(text: str) -> list[RagSearchChunkSection]:
    """返回带原始文本的 chunk 块，用于 prompt 上下文的去重。"""
    matches = list(_RESULT_SPLIT_RE.finditer(text or ""))
    sections: list[RagSearchChunkSection] = []

    for index, match in enumerate(matches):
        section_end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        block = text[match.start():section_end].strip()
        chunk_id = _field(block, "Chunk ID")
        if not chunk_id:
            continue

        try:
            score = float(match.group(1).strip())
        except ValueError:
            score = None

        sections.append(
            RagSearchChunkSection(
                chunk_id=chunk_id,
                score=score,
                text=block,
            )
        )

    return sections


def tool_output_to_text(output: Any) -> str:
    """把 LangChain 工具输出内容规范化为 rag-mcp 输出的文本。"""
    content = getattr(output, "content", output)
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return str(content)

    text_parts: list[str] = []
    for block in content:
        if isinstance(block, str):
            text_parts.append(block)
        elif isinstance(block, dict):
            text = block.get("text")
            if isinstance(text, str):
                text_parts.append(text)
    return "\n".join(text_parts)


def _field(section: str, label: str) -> str:
    match = re.search(rf"^{re.escape(label)}:\s*(.+?)\s*$", section, re.MULTILINE)
    return match.group(1).strip() if match else ""
