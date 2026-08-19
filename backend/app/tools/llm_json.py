"""LLM 输出中 JSON 代码块提取工具"""
import json
import re


def extract_json_block(content: str) -> str:
    """从 LLM 输出中提取 JSON 文本，去除可能包裹的 markdown 代码块标记

    LLM 输出通常形如：
        ```json
        { ... }
        ```
    或直接输出裸 JSON。部分模型会在 JSON 前后增加说明文字，因此依次尝试：
    完整代码块、任意位置的 JSON 对象、原始文本。
    """
    text = content.strip()

    fenced_blocks = re.findall(
        r"```(?:json)?\s*(.*?)```",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    for block in fenced_blocks:
        candidate = block.strip()
        if candidate.startswith(("{", "[")):
            return candidate

    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char not in "[{":
            continue
        try:
            _, end = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        return text[index:index + end]

    return text
