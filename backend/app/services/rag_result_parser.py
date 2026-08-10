"""Parse structured chunk metadata from rag_search text results."""
from __future__ import annotations

import re
from typing import Any

_RESULT_SPLIT_RE = re.compile(r"^--- 结果\s+\d+\s+\(相似度:\s*([^)]+)\)\s+---$", re.MULTILINE)


def parse_rag_search_result(text: str) -> list[dict]:
    """Return ranked chunk metadata embedded in rag-mcp's text response."""
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


def tool_output_to_text(output: Any) -> str:
    """Normalize LangChain tool output content to the text emitted by rag-mcp."""
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
