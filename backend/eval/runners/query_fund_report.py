"""

用法:
    # 单个基金
    python scripts/query_fund_report.py 159103                  # 全部 chunk
    python scripts/query_fund_report.py 159103 --chunk 5        # 第 5 个 chunk
    python scripts/query_fund_report.py 159103 --chunks 3,6,10  # 多个指定 chunk
    python scripts/query_fund_report.py 159103 --from 0 --to 9  # chunk 0~9
    python scripts/query_fund_report.py 159103 --headers        # 仅显示 chunk 目录结构

    # 批量查询多个基金
    python scripts/query_fund_report.py --funds 159101,159102,159103
    python scripts/query_fund_report.py --funds 159101,159102 --chunks 3,6
    python scripts/query_fund_report.py --funds 159101,159102 --headers

    # 全局操作
    python backend/eval/runners/query_fund_report.py --list     # 列出所有已索引基金
"""

from __future__ import annotations

import io
import os
import sys
from pymilvus import MilvusClient

# ── 配置 ──────────────────────────────────────────────
MILVUS_HOST = os.getenv("MILVUS_HOST", "localhost")
MILVUS_PORT = int(os.getenv("MILVUS_PORT", "19595"))
COLLECTION = os.getenv("MILVUS_COLLECTION", "fund_reports_mineru")

OUTPUT_FIELDS = [
    "id", "fund_code", "fund_name", "content", "chunk_index",
    "header_1", "header_2", "header_3",
    "header_4", "header_5", "header_6",
    "file_path",
]


def connect():
    uri = f"http://{MILVUS_HOST}:{MILVUS_PORT}"
    return MilvusClient(uri=uri)


def list_funds(client: MilvusClient, collection: str = COLLECTION) -> list[dict]:
    """列出所有已索引的基金代码和名称"""
    results = client.query(
        collection_name=collection,
        filter="fund_code != ''",
        output_fields=["fund_code", "fund_name"],
        limit=10000,
    )
    seen = {}
    for r in results:
        code = r.get("fund_code", "").strip()
        name = r.get("fund_name", "").strip()
        if code and code not in seen:
            seen[code] = name
    return [{"code": k, "name": v} for k, v in sorted(seen.items())]


def query_by_fund(
    client: MilvusClient,
    fund_code: str,
    chunk_indexes: list[int] | None = None,
    chunk_start: int | None = None,
    chunk_end: int | None = None,
    collection: str = COLLECTION,
) -> list[dict]:
    """
    按基金代码拉取报告 chunk，支持多种索引过滤方式。

    Args:
        fund_code: 6 位基金代码
        chunk_indexes: 精确指定的 chunk_index 列表 (优先级最高)
        chunk_start:  起始 chunk_index（含），None 表示从头
        chunk_end:    结束 chunk_index（含），None 表示到末尾
    """
    filters = [f'fund_code == "{fund_code}"']

    if chunk_indexes is not None:
        # 多个精确 chunk: chunk_index in [3, 6, 10]
        ids_str = ", ".join(str(i) for i in chunk_indexes)
        filters.append(f"chunk_index in [{ids_str}]")
    else:
        if chunk_start is not None:
            filters.append(f"chunk_index >= {chunk_start}")
        if chunk_end is not None:
            filters.append(f"chunk_index <= {chunk_end}")

    filter_expr = " and ".join(filters)

    results = client.query(
        collection_name=collection,
        filter=filter_expr,
        output_fields=OUTPUT_FIELDS,
        limit=10000,
    )
    results.sort(key=lambda r: r.get("chunk_index", 0))
    return results


def query_multiple_funds(
    client: MilvusClient,
    fund_codes: list[str],
    chunk_indexes: list[int] | None = None,
    chunk_start: int | None = None,
    chunk_end: int | None = None,
    collection: str = COLLECTION,
) -> dict[str, list[dict]]:
    """批量查询多个基金，返回 {fund_code: [chunks]}"""
    results = {}
    for code in fund_codes:
        chunks = query_by_fund(
            client, code,
            chunk_indexes=chunk_indexes,
            chunk_start=chunk_start,
            chunk_end=chunk_end,
            collection=collection,
        )
        results[code] = chunks
    return results


def build_header_path(r: dict) -> str:
    """拼接 header 路径"""
    parts = []
    for i in range(1, 7):
        h = r.get(f"header_{i}", "").strip()
        if h:
            parts.append(h)
    return " > ".join(parts)


def print_headers(results: list[dict], fund_label: str, show_id: bool = True):
    """打印 chunk 目录结构"""
    print(f"\n{'='*70}")
    print(f"  {fund_label}  —  {len(results)} chunks")
    print(f"{'='*70}")
    for r in results:
        ci = r["chunk_index"]
        path = build_header_path(r)
        line = f"  chunk[{ci:03d}]  {path}"
        if show_id:
            line += f"  | id={r['id']}"
        print(line)


def print_content(results: list[dict], fund_label: str, max_content_len: int = 0):
    """打印完整 chunk 内容"""
    print(f"\n{'='*70}")
    print(f"  {fund_label}  —  {len(results)} chunks")
    print(f"{'='*70}")
    for r in results:
        path = build_header_path(r)
        content = r["content"]
        if max_content_len > 0 and len(content) > max_content_len:
            content = content[:max_content_len] + f"\n... [truncated, {len(r['content'])} total chars]"

        print(f"\n── Chunk {r['chunk_index']} ──")
        if path:
            print(f"   [{path}]")
        print(f"  {content}")


def parse_args(argv: list[str]) -> dict:
    """解析命令行参数"""
    args = {
        "fund_code": None,
        "fund_codes": None,
        "chunk_indexes": None,
        "chunk_start": None,
        "chunk_end": None,
        "list_mode": False,
        "headers_only": False,
        "max_content_len": 0,
    }
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == "--list":
            args["list_mode"] = True
            i += 1
        elif a == "--headers":
            args["headers_only"] = True
            i += 1
        elif a == "--funds":
            args["fund_codes"] = [c.strip() for c in argv[i + 1].split(",") if c.strip()]
            i += 2
        elif a == "--chunk":
            args["chunk_indexes"] = [int(argv[i + 1])]
            i += 2
        elif a == "--chunks":
            args["chunk_indexes"] = [int(c.strip()) for c in argv[i + 1].split(",") if c.strip()]
            i += 2
        elif a == "--from":
            args["chunk_start"] = int(argv[i + 1])
            i += 2
        elif a == "--to":
            args["chunk_end"] = int(argv[i + 1])
            i += 2
        elif a == "--content-len":
            args["max_content_len"] = int(argv[i + 1])
            i += 2
        elif not a.startswith("--") and args["fund_code"] is None:
            args["fund_code"] = a
            i += 1
        else:
            i += 1
    return args


def print_usage():
    print("用法:")
    print("  # 单个基金")
    print("  python scripts/query_fund_report.py 159103                  # 全部 chunk")
    print("  python scripts/query_fund_report.py 159103 --chunk 5        # 第 5 个 chunk")
    print("  python scripts/query_fund_report.py 159103 --chunks 3,6,10  # 多个指定 chunk")
    print("  python scripts/query_fund_report.py 159103 --from 0 --to 9  # chunk 0~9")
    print("  python scripts/query_fund_report.py 159103 --headers        # 仅显示目录结构")
    print()
    print("  # 批量查询")
    print("  python scripts/query_fund_report.py --funds 159101,159102,159103")
    print("  python scripts/query_fund_report.py --funds 159101,159102 --chunks 3,6")
    print("  python scripts/query_fund_report.py --funds 159101,159102 --headers")
    print()
    print("  # 全局")
    print("  python scripts/query_fund_report.py --list")


def main():
    # 修复 Windows GBK 编码问题
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

    if len(sys.argv) < 2:
        print_usage()
        sys.exit(1)

    args = parse_args(sys.argv[1:])
    client = connect()

    # ── --list ──
    if args["list_mode"]:
        funds = list_funds(client)
        print(f"\n已索引基金: {len(funds)} 只\n")
        for f in funds:
            print(f"  {f['code']}  {f['name']}")
        return

    # ── 批量查询 --funds ──
    if args["fund_codes"]:
        all_results = query_multiple_funds(
            client,
            args["fund_codes"],
            chunk_indexes=args["chunk_indexes"],
            chunk_start=args["chunk_start"],
            chunk_end=args["chunk_end"],
        )
        for code in args["fund_codes"]:
            results = all_results.get(code, [])
            if not results:
                print(f"\n{code}: 未找到内容")
                continue
            name = results[0].get("fund_name", code)
            label = f"{code}  {name}"
            if args["headers_only"]:
                print_headers(results, label)
            else:
                print_content(results, label, max_content_len=args["max_content_len"])
        return

    # ── 单个基金 ──
    if not args["fund_code"]:
        print("请提供基金代码，或使用 --funds 批量查询")
        sys.exit(1)

    results = query_by_fund(
        client,
        args["fund_code"],
        chunk_indexes=args["chunk_indexes"],
        chunk_start=args["chunk_start"],
        chunk_end=args["chunk_end"],
    )

    if not results:
        desc = f"基金代码 {args['fund_code']}"
        if args["chunk_indexes"]:
            desc += f" chunks {args['chunk_indexes']}"
        elif args["chunk_start"] is not None or args["chunk_end"] is not None:
            s = args["chunk_start"] if args["chunk_start"] is not None else "?"
            e = args["chunk_end"] if args["chunk_end"] is not None else "?"
            desc += f" chunk {s}~{e}"
        print(f"未找到 {desc} 的内容")
        return

    name = results[0].get("fund_name", args["fund_code"])
    label = f"{args['fund_code']}  {name}"

    if args["headers_only"]:
        print_headers(results, label)
    else:
        print_content(results, label, max_content_len=args["max_content_len"])


if __name__ == "__main__":
    main()
