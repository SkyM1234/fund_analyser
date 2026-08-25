"""基于 MinerU 报告构建基金名称解析评测数据集。

每份报告贡献五种标题形式：
1. 报告目录中的基金简称
2. 报告目录中的基金全称
3. 带年度报告后缀的基金简称
4. 分析后 Markdown 中的原始 H1 标题
5. 原始报告目录标题（含基金代码）
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MARKDOWN_DIR = REPO_ROOT / "markdown_mineru"
DEFAULT_OUTPUT = Path(__file__).resolve().parent.parent / "datasets" / "fund_name_resolution.jsonl"

DIRECTORY_RE = re.compile(
    r"^(?P<fund_code>\d{6})_(?P<short_name>[^_]+)_(?P<report_title>.+)$"
)
ANNUAL_REPORT_SUFFIX_RE = re.compile(r"\s*2025\s*年年度报告\s*$")
H1_RE = re.compile(r"^#\s+(.+?)\s*$")


def _read_markdown_title(path: Path) -> str:
    with path.open("r", encoding="utf-8") as markdown:
        for line in markdown:
            match = H1_RE.match(line.rstrip("\r\n"))
            if match:
                return match.group(1).strip()
    raise ValueError(f"未找到 Markdown 一级主标题: {path}")


def _parse_report(directory: Path) -> dict[str, str]:
    match = DIRECTORY_RE.fullmatch(directory.name)
    if not match:
        raise ValueError(f"报告目录名格式不符合约定: {directory.name}")

    fund_code = match.group("fund_code")
    short_name = match.group("short_name").strip()
    report_title = match.group("report_title").strip()

    # 某个历史目录在完整报告标题前重复了基金代码。
    report_title = re.sub(rf"^{re.escape(fund_code)}_", "", report_title, count=1)
    full_name = ANNUAL_REPORT_SUFFIX_RE.sub("", report_title).strip()
    if not full_name or full_name == report_title:
        raise ValueError(f"目录报告标题缺少 2025 年年度报告后缀: {directory.name}")

    analyzed_files = sorted(directory.glob("*_analyzed.md"))
    if len(analyzed_files) != 1:
        raise ValueError(
            f"每个报告目录必须恰好包含一个 _analyzed.md 文件: "
            f"{directory.name}（实际 {len(analyzed_files)} 个）"
        )

    return {
        "fund_code": fund_code,
        "short_name": short_name,
        "full_name": full_name,
        "markdown_title": _read_markdown_title(analyzed_files[0]),
        "source_title": directory.name,
    }


def build_examples(markdown_dir: Path) -> list[dict]:
    directories = sorted(path for path in markdown_dir.iterdir() if path.is_dir())
    if not directories:
        raise ValueError(f"未找到报告子目录: {markdown_dir}")

    examples: list[dict] = []
    next_id = 1

    for directory in directories:
        report = _parse_report(directory)
        code = report["fund_code"]
        short_name = report["short_name"]

        variants = [
            ("short_name", short_name, "目录中的基金短名"),
            ("full_name", report["full_name"], "目录报告标题去除年度报告后缀"),
            (
                "short_report_title",
                f"{short_name}2025年年度报告",
                "基金短名加年度报告后缀",
            ),
            (
                "markdown_title",
                report["markdown_title"],
                "Markdown _analyzed.md 的一级主标题",
            ),
            (
                "source_title",
                report["source_title"],
                "markdown_mineru 中的报告子目录标题",
            ),
        ]

        queries = [query for _, query, _ in variants]
        if len(set(queries)) != len(queries):
            raise ValueError(f"同一报告的五个标题存在重复: {directory.name}")

        for category, query, note in variants:
            examples.append(
                {
                    "id": f"name-res-{next_id:03d}",
                    "query": query,
                    "expected_fund_code": code,
                    "category": category,
                    "note": note,
                }
            )
            next_id += 1

    return examples


def validate_examples(examples: list[dict], report_count: int) -> None:
    expected_count = report_count * 5
    if len(examples) != expected_count:
        raise ValueError(f"样本数错误: 期望 {expected_count}，实际 {len(examples)}")

    ids = [example["id"] for example in examples]
    queries = [example["query"] for example in examples]
    if len(ids) != len(set(ids)):
        raise ValueError("样本 ID 存在重复")
    if len(queries) != len(set(queries)):
        raise ValueError("全量标题 query 存在重复")

    categories = {example["category"] for example in examples}
    expected_categories = {
        "short_name",
        "full_name",
        "short_report_title",
        "markdown_title",
        "source_title",
    }
    if categories != expected_categories:
        raise ValueError(f"标题类别不完整: {sorted(categories)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--markdown-dir", type=Path, default=DEFAULT_MARKDOWN_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    examples = build_examples(args.markdown_dir)
    report_count = len([path for path in args.markdown_dir.iterdir() if path.is_dir()])
    validate_examples(examples, report_count)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    content = "\n".join(
        json.dumps(example, ensure_ascii=False) for example in examples
    )
    args.output.write_text(content + "\n", encoding="utf-8")

    print(
        f"[OK] 生成 {len(examples)} 条样本，覆盖 {report_count} 份报告: "
        f"{args.output}"
    )


if __name__ == "__main__":
    main()
