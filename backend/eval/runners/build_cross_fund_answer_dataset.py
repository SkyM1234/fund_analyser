"""Build the dedicated cross-fund answer dataset.

The questions are grounded in Markdown-derived fund/report positions and are
resolved against the live Milvus collection before writing the JSONL file.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from pymilvus import MilvusClient

try:
    from .build_single_fund_answer_dataset import (
        CROSS_FUND_SPECS,
        build_cross_fund_examples,
        write_jsonl,
    )
except ImportError:
    from build_single_fund_answer_dataset import (
        CROSS_FUND_SPECS,
        build_cross_fund_examples,
        write_jsonl,
    )

EVAL_DIR = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = EVAL_DIR / "datasets" / "answer_cross_fund.jsonl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--milvus-uri", default="http://localhost:19595")
    parser.add_argument("--collection", default="fund_reports_mineru")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    client = MilvusClient(uri=args.milvus_uri)
    rows = build_cross_fund_examples(client, args.collection)
    write_jsonl(args.output, rows)
    print(f"Wrote {len(rows)} cross-fund examples to {args.output}")


if __name__ == "__main__":
    main()
