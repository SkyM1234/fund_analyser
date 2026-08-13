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


def format_row_fallback(row: list[str]) -> str:
    return " | ".join(cell for cell in row if cell)


def dataframe_to_text(dataframe: pd.DataFrame) -> str:
    rows = [
        [normalize_cell(value) for value in row]
        for row in dataframe.fillna("").values.tolist()
    ]
    rows = [row for row in rows if any(row)]
    if not rows:
        return ""

    return "\n".join(format_row_fallback(row) for row in rows)


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
