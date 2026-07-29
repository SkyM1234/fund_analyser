"""LLM 输出中 JSON 代码块提取工具"""


def extract_json_block(content: str) -> str:
    """从 LLM 输出中提取 JSON 文本，去除可能包裹的 markdown 代码块标记

    LLM 输出通常形如：
        ```json
        { ... }
        ```
    或直接输出裸 JSON。使用 startswith/endswith 判断首尾标记，
    而非 str.split("```")，避免当 JSON 内容本身包含反引号，
    或输出中出现多个代码块时提取到错误片段。
    """
    text = content.strip()

    if text.startswith("```json"):
        text = text[len("```json"):]
    elif text.startswith("```"):
        text = text[len("```"):]

    if text.endswith("```"):
        text = text[:-len("```")]

    return text.strip()
