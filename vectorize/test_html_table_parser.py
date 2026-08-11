"""
Standalone HTML table normalization tester.

Examples:
    python test_html_table_parser.py --file table.html
    Get-Content -Raw table.html | python test_html_table_parser.py
"""
import argparse
import re
import sys
from html import unescape
from io import StringIO
from pathlib import Path

import pandas as pd


RE_TABLE_BLOCK = re.compile(
    r'<!--\s*TABLE_START[^>]*?-->.*?<!--\s*TABLE_END[^>]*?-->',
    re.DOTALL | re.IGNORECASE,
)
RE_HTML_TABLE = re.compile(r'<table\b[^>]*>.*?</table>', re.DOTALL | re.IGNORECASE)


def normalize_cell(value) -> str:
    if value is None:
        return ""

    text = str(value)
    if text.lower() == "nan":
        return ""

    return re.sub(r'\s+', ' ', text).strip()


def is_vertical_key_value_table(rows: list[list[str]]) -> bool:
    if len(rows) < 2 or any(len(row) != 2 for row in rows):
        return False

    keys = [row[0] for row in rows]
    values = [row[1] for row in rows]
    if not all(keys) or not all(values):
        return False

    unique_key_ratio = len(set(keys)) / len(keys)
    average_key_length = sum(len(key) for key in keys) / len(keys)
    return unique_key_ratio >= 0.8 and average_key_length <= 30


def looks_like_header_row(row: list[str]) -> bool:
    if not row or not all(row):
        return False

    average_length = sum(len(cell) for cell in row) / len(row)
    has_number = any(re.search(r'\d', cell) for cell in row)
    return average_length <= 30 and not has_number


def infer_header_row_count(rows: list[list[str]]) -> int:
    if len(rows) < 2 or not looks_like_header_row(rows[0]):
        return 0

    header_rows = 1
    previous_row = rows[0]
    max_header_rows = min(3, len(rows) - 1)

    while header_rows < max_header_rows:
        candidate = rows[header_rows]
        if not looks_like_header_row(candidate):
            break

        repeated_columns = sum(
            previous == current
            for previous, current in zip(previous_row, candidate)
            if previous
        )
        if repeated_columns < max(1, int(len(candidate) * 0.4)):
            break

        header_rows += 1
        previous_row = candidate

    return header_rows


def format_row_fallback(row: list[str]) -> str:
    return "表格行: " + " | ".join(cell for cell in row if cell)


def dataframe_to_text(dataframe: pd.DataFrame) -> str:
    rows = [
        [normalize_cell(value) for value in row]
        for row in dataframe.fillna("").values.tolist()
    ]
    rows = [row for row in rows if any(row)]
    if not rows:
        return ""

    if is_vertical_key_value_table(rows):
        return "\n".join(f"{key}: {value}" for key, value in rows)

    header_row_count = infer_header_row_count(rows)
    if not header_row_count or header_row_count >= len(rows):
        return "\n".join(format_row_fallback(row) for row in rows)

    headers = []
    for column_index in range(len(rows[0])):
        header_parts = []
        for header_row in rows[:header_row_count]:
            value = header_row[column_index]
            if value and value not in header_parts:
                header_parts.append(value)
        headers.append(" > ".join(header_parts) or f"列{column_index + 1}")

    normalized_rows = []
    for row in rows[header_row_count:]:
        fields = [
            f"{header}: {value}"
            for header, value in zip(headers, row)
            if value
        ]
        if fields:
            normalized_rows.append("；".join(fields))

    return "\n".join(normalized_rows) or "\n".join(
        format_row_fallback(row) for row in rows
    )


def fallback_table_to_text(table_html: str) -> str:
    row_matches = re.findall(r'<tr[^>]*>(.*?)</tr>', table_html, re.DOTALL | re.IGNORECASE)
    formatted_rows = []

    for row_html in row_matches:
        cell_matches = re.findall(
            r'<t[dh][^>]*>(.*?)</t[dh]>',
            row_html,
            re.DOTALL | re.IGNORECASE,
        )
        cells = [
            re.sub(r'\s+', ' ', unescape(re.sub(r'<[^>]+>', ' ', cell))).strip()
            for cell in cell_matches
        ]
        if any(cells):
            formatted_rows.append(format_row_fallback(cells))

    if formatted_rows:
        return "\n".join(formatted_rows)

    plain_text = unescape(re.sub(r'<[^>]+>', ' ', table_html))
    return re.sub(r'\s+', ' ', plain_text).strip()


def table_to_text(table_html: str) -> str:
    try:
        dataframes = pd.read_html(
            StringIO(table_html),
            header=None,
            keep_default_na=False,
        )
    except Exception:
        return fallback_table_to_text(table_html)

    return "\n".join(
        table_text
        for table_text in (dataframe_to_text(dataframe) for dataframe in dataframes)
        if table_text
    )


def extract_tables(content: str) -> list[str]:
    marked_tables = [match.group() for match in RE_TABLE_BLOCK.finditer(content)]
    if marked_tables:
        return marked_tables
    return [match.group() for match in RE_HTML_TABLE.finditer(content)]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert HTML tables into embedding-friendly text."
    )
    parser.add_argument(
        "--file",
        type=Path,
        help="UTF-8 file containing one or more HTML tables. Reads stdin if omitted.",
    )
    args = parser.parse_args()

    content = args.file.read_text(encoding="utf-8") if args.file else sys.stdin.read()
    if not content.strip():
        parser.error("No HTML input received.")

    tables = extract_tables(content)
    if not tables:
        parser.error("No <table>...</table> block found.")

    for index, table_html in enumerate(tables, start=1):
        print(f"=== Table {index} ===")
        print(table_to_text(table_html))
        if index < len(tables):
            print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
