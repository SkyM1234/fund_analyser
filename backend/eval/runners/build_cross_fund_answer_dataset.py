"""Build the dedicated cross-fund answer dataset.

The questions are grounded in Markdown-derived fund/report positions and are
resolved against the live Milvus collection before writing the JSONL file.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from pymilvus import MilvusClient

EVAL_DIR = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = EVAL_DIR / "datasets" / "answer_cross_fund.jsonl"

def named_fund_calls(codes: list[str]) -> list[dict]:
    calls = []
    for code in codes:
        calls.append({"name": "rag_identify_funds"})
        calls.append({"name": "rag_search", "args": {"filter_fund_code": code}})
    return calls

CROSS_FUND_SPECS = [
    {
        "query": "截至2025年末，国证港股通科技ETF板块中持有腾讯控股的基金有哪些？各自持仓占基金资产净值的比例是多少？",
        "fund_chunks": {
            "159101": 128,
            "159125": 148,
            "159128": 143,
            "159251": 146,
            "159636": 164,
        },
        "reference_answer": (
            "截至2025年末，国证港股通科技ETF板块中，159101、159125、159128、"
            "159251和159636均持有腾讯控股，占基金资产净值的比例分别为15.32%、"
            "15.31%、15.40%、15.32%和15.38%。"
        ),
        "expected_fund_codes": ["159101", "159125", "159128", "159251", "159636"],
        "key_facts": ["腾讯控股", "15.32%", "15.31%", "15.40%", "15.38%"],
        "relevant_keywords": ["腾讯控股", "15.32", "15.31", "15.40", "15.38"],
        "intent": "cross_fund_query",
        "category": "cross_fund_strategy",
        "note": "按国证港股通科技ETF板块的年度报告逐基金核验腾讯控股持仓。",
    },
    {
        "query": "国证港股通科技ETF板块中，2025年末非日常生活消费品或可选消费品配置占净值超过36%的基金有哪些？",
        "fund_chunks": {
            "159101": 126,
            "159125": 146,
            "159128": 141,
            "159251": 144,
            "159636": 161,
        },
        "reference_answer": (
            "截至2025年末，159101、159125、159128、159251和159636的相关消费品行业"
            "配置占基金资产净值均超过36%，比例分别为36.43%、36.41%、36.62%、"
            "36.48%和36.59%。报告中的行业名称分别为非必需消费品、非日常生活消费品、"
            "非日常生活消费品、可选消费品和非必需消费品。"
        ),
        "expected_fund_codes": ["159101", "159125", "159128", "159251", "159636"],
        "key_facts": ["超过36%", "36.43%", "36.41%", "36.62%", "36.48%", "36.59%"],
        "relevant_keywords": [
            "36.43",
            "36.41",
            "36.62",
            "36.48",
            "36.59",
        ],
        "intent": "cross_fund_query",
        "category": "cross_fund_strategy",
        "note": "按年度报告期末行业配置表逐基金筛选超过36%的消费品配置。",
    },
    {
        "query": "华夏、招商、天弘和万家的国证港股通科技ETF在2025年报告期内的份额净值增长率和期末份额净值分别是多少？",
        "fund_chunks": {
            "159101": 9,
            "159125": 10,
            "159128": 10,
            "159251": 12,
        },
        "reference_answer": (
            "2025年报告期内，华夏159101、招商159125、天弘159128和万家159251的"
            "份额净值增长率分别为-7.10%、-10.16%、-8.75%和-3.56%；期末份额净值"
            "分别为0.9290元、0.8984元、0.9125元和0.9644元。"
        ),
        "expected_fund_codes": ["159101", "159125", "159128", "159251"],
        "key_facts": ["-7.10%", "-10.16%", "-8.75%", "-3.56%", "0.9290", "0.8984", "0.9125", "0.9644"],
        "relevant_keywords": ["-7.10%", "-10.16%", "-8.75%", "-3.56%", "0.9290", "0.8984", "0.9125", "0.9644"],
        "intent": "cross_fund_query",
        "category": "cross_fund_strategy",
        "note": "仅比较 query 明确列出的华夏、招商、天弘和万家四只产品。",
    },
    {
        "query": "国证港股通科技ETF板块中的华夏、招商、天弘和万家产品分别采用什么指数复制或替代策略跟踪标的指数？",
        "fund_chunks": {
            "159101": 3,
            "159125": 3,
            "159128": 3,
            "159251": 4,
        },
        "reference_answer": (
            "华夏159101主要采用组合复制策略和适当的替代性策略；招商159125采用完全复制法，"
            "流动性不足等特殊情况下可使用成份券替代；天弘159128采用完全复制策略；"
            "万家159251采用完全复制法。四只产品均以跟踪国证港股通科技指数为目标。"
        ),
        "expected_fund_codes": ["159101", "159125", "159128", "159251"],
        "key_facts": ["组合复制策略", "替代性策略", "完全复制法", "完全复制策略", "国证港股通科技指数"],
        "relevant_keywords": ["完全复制", "替代", "国证港股通科技指数"],
        "intent": "cross_fund_query",
        "category": "cross_fund_strategy",
        "note": "按四只产品的基金产品说明和投资策略章节逐基金比较复制方式。",
    },
    {
        "query": "截至2025年末，国证港股通科技ETF板块中四只基金的个人投资者持有比例分别是多少，哪些产品还设有联接基金持有人？",
        "fund_chunks": {
            "159101": 151,
            "159125": 172,
            "159128": 164,
            "159251": 165,
        },
        "reference_answer": (
            "截至2025年末，华夏159101、招商159125、天弘159128和万家159251的个人投资者"
            "持有比例分别为75.82%、86.95%、31.08%和78.18%。设有联接基金持有人的产品为"
            "159101、159128和159251，其联接基金持有比例分别为14.56%、64.18%和17.89%；"
            "159125未列示联接基金持有人。"
        ),
        "expected_fund_codes": ["159101", "159125", "159128", "159251"],
        "key_facts": ["75.82%", "86.95%", "31.08%", "78.18%", "联接基金", "14.56%", "64.18%", "17.89%"],
        "relevant_keywords": ["个人投资者", "75.82%", "86.95%", "31.08%", "78.18", "联接基金"],
        "intent": "cross_fund_query",
        "category": "cross_fund_strategy",
        "note": "仅比较 query 明确列出的四只产品，并区分个人投资者与联接基金持有人。",
    },
    {
        "query": "万家、天弘、银华和华安四只科创债ETF在2025年报告期内的份额净值增长率和期末份额净值分别是多少？",
        "fund_chunks": {
            "159110": 14,
            "159111": 10,
            "159112": 15,
            "159115": 10,
        },
        "reference_answer": (
            "2025年报告期内，万家159110、天弘159111、银华159112和华安159115的份额净值"
            "增长率分别为0.45%、0.39%、0.26%和0.25%；期末份额净值分别为100.4468元、"
            "100.3876元、100.2649元和100.2548元。"
        ),
        "expected_fund_codes": ["159110", "159111", "159112", "159115"],
        "key_facts": ["0.45%", "0.39%", "0.26%", "0.25%", "100.4468", "100.3876", "100.2649", "100.2548"],
        "relevant_keywords": ["0.45%", "0.39%", "0.26%", "0.25%", "100.4468", "100.3876", "100.2649", "100.2548"],
        "intent": "cross_fund_query",
        "category": "cross_fund_strategy",
        "note": "按四只科创债ETF的2025年年度报告主要会计数据和财务指标逐基金比较。",
    },
    {
        "query": "截至2025年末，万家、天弘、银华和华安科创债ETF各自的第一大债券持仓是什么，占基金资产净值比例是多少？",
        "fund_chunks": {
            "159110": 152,
            "159111": 140,
            "159112": 151,
            "159115": 143,
        },
        "reference_answer": (
            "截至2025年末，四只产品的第一大债券持仓分别为：万家159110的23广金K1，"
            "占基金资产净值2.56%；天弘159111的25管网SK，占2.32%；银华159112的25GTHTK1，"
            "占0.87%；华安159115的25CMGK01，占2.17%。"
        ),
        "expected_fund_codes": ["159110", "159111", "159112", "159115"],
        "key_facts": ["23广金K1", "2.56%", "25管网SK", "2.32%", "25GTHTK1", "0.87%", "25CMGK01", "2.17%"],
        "relevant_keywords": ["23广金K1", "25管网SK", "25GTHTK1", "25CMGK01"],
        "intent": "cross_fund_query",
        "category": "cross_fund_strategy",
        "note": "按四只科创债ETF期末债券投资明细的第一行逐基金核验。",
    },
    {
        "query": "科创债ETF板块中，万家、天弘、银华和华安产品截至2025年末的机构投资者持有份额占总份额比例分别是多少？",
        "fund_chunks": {
            "159110": 160,
            "159111": 146,
            "159112": 159,
            "159115": 151,
        },
        "reference_answer": (
            "截至2025年末，万家159110、天弘159111、银华159112和华安159115的机构投资者"
            "持有份额占总份额比例分别为99.53%、99.98%、99.98%和99.73%。"
        ),
        "expected_fund_codes": ["159110", "159111", "159112", "159115"],
        "key_facts": ["99.53%", "99.98%", "99.73%"],
        "relevant_keywords": ["机构投资者", "99.53", "99.98", "99.73"],
        "intent": "cross_fund_query",
        "category": "cross_fund_strategy",
        "note": "按四只科创债ETF期末基金份额持有人结构逐基金比较机构占比。",
    },
    {
        "query": "万家、天弘、银华和华安四只科创债ETF分别采用哪些债券指数化、抽样复制或债券投资策略？",
        "fund_chunks": {
            "159110": 5,
            "159111": 3,
            "159112": 5,
            "159115": 4,
        },
        "reference_answer": (
            "万家159110主要采用债券指数化投资和抽样复制法；天弘159111采用分层抽样复制和动态最优化方法；"
            "银华159112采用抽样复制并动态最优化；华安159115可采用债券投资、国债期货投资和资产支持证券投资策略。"
        ),
        "expected_fund_codes": ["159110", "159111", "159112", "159115"],
        "key_facts": ["债券指数化投资", "抽样复制", "分层抽样复制", "动态最优化", "国债期货投资", "资产支持证券投资"],
        "relevant_keywords": ["抽样复制", "动态最优化", "债券投资", "国债期货", "资产支持证券"],
        "intent": "cross_fund_query",
        "category": "cross_fund_strategy",
        "note": "按四只产品的基金产品说明或基金简介章节逐基金归纳投资策略。",
    },
    {
        "query": "2025年9月17日成立、9月24日上市的科创债ETF有哪些？请列出万家、天弘、银华和华安产品的基金代码与期末基金份额总额。",
        "fund_chunks": {
            "159110": 3,
            "159111": 2,
            "159112": 3,
            "159115": 3,
        },
        "reference_answer": (
            "2025年9月17日成立、9月24日上市的四只科创债ETF包括：万家159110，期末基金份额"
            "总额21,777,093.00份；天弘159111，107,806,676.00份；银华159112，290,887,830.00份；"
            "华安159115，50,903,487.00份。"
        ),
        "expected_fund_codes": ["159110", "159111", "159112", "159115"],
        "key_facts": ["2025年9月17日", "2025年9月24日", "21,777,093.00", "107,806,676.00", "290,887,830.00", "50,903,487.00"],
        "relevant_keywords": ["2025年9月17日", "2025年9月24日", "期末基金份额总额", "份"],
        "intent": "cross_fund_query",
        "category": "cross_fund_strategy",
        "note": "按基金基本情况章节重新核验成立日、上市日、基金代码和期末份额总额。",
    },
]


def fetch_chunks_by_fund_and_index(
    client: MilvusClient,
    collection: str,
    fund_chunks: dict[str, int],
) -> list[dict]:
    """Resolve the current Milvus chunk IDs from stable fund/report positions."""
    rows = []
    for fund_code, chunk_index in fund_chunks.items():
        result = client.query(
            collection_name=collection,
            filter=f'fund_code == "{fund_code}" and chunk_index == {chunk_index}',
            output_fields=["id", "fund_code", "content", "chunk_index"],
            limit=10,
        )
        matches = [row for row in result if row.get("fund_code") == fund_code]
        if len(matches) != 1:
            raise ValueError(
                f"Expected one Milvus chunk for fund {fund_code}, index {chunk_index}; "
                f"got {len(matches)}"
            )
        rows.append(matches[0])
    return rows


def build_cross_fund_examples(
    client: MilvusClient,
    collection: str,
) -> list[dict]:
    examples = []
    for sequence, spec in enumerate(CROSS_FUND_SPECS, 51):
        selected_chunks = fetch_chunks_by_fund_and_index(
            client, collection, spec["fund_chunks"]
        )
        combined_content = "\n".join(chunk.get("content", "") for chunk in selected_chunks)
        missing_keywords = [
            keyword
            for keyword in spec["relevant_keywords"]
            if keyword not in combined_content
        ]
        if missing_keywords:
            raise ValueError(
                f"cross-fund sample {sequence} is missing keywords: {missing_keywords}"
            )
        examples.append(
            {
                "id": f"answer-{sequence:03d}",
                "query": spec["query"],
                "reference_answer": spec["reference_answer"],
                "expected_fund_codes": spec["expected_fund_codes"],
                "key_facts": spec["key_facts"],
                "should_refuse": False,
                "intent": spec["intent"],
                "category": spec["category"],
                "relevant_keywords": spec["relevant_keywords"],
                "relevant_chunk_ids": [chunk["id"] for chunk in selected_chunks],
                "expected_tool_calls": named_fund_calls(spec["expected_fund_codes"]),
                "note": (
                    f"{spec['note']}；"
                    + "；".join(
                        f"{chunk['fund_code']} chunk={chunk['chunk_index']} id={chunk['id']}"
                        for chunk in selected_chunks
                    )
                ),
            }
        )
    return examples


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for sequence, row in enumerate(rows, 1):
            row["id"] = f"answer-{sequence:03d}"
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


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
