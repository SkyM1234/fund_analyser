"""Build the first batch of single-fund strategy answer examples.

The selected questions come from retrieval.jsonl. Reference answers and key
facts are manually grounded in the corresponding Milvus chunks, while this
script verifies that every chunk ID and keyword still matches the live
collection before writing answer.jsonl.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from pymilvus import MilvusClient

EVAL_DIR = Path(__file__).resolve().parent.parent
DEFAULT_RETRIEVAL_PATH = EVAL_DIR / "datasets" / "retrieval.jsonl"
DEFAULT_ANSWER_PATH = EVAL_DIR / "datasets" / "answer.jsonl"

SELECTED_RETRIEVAL_IDS = [
    "retrieval-002",
    "retrieval-007",
    "retrieval-012",
    "retrieval-022",
    "retrieval-122",
    "retrieval-032",
    "retrieval-037",
    "retrieval-042",
    "retrieval-046",
    "retrieval-052",
    "retrieval-056",
    "retrieval-066",
    "retrieval-072",
    "retrieval-076",
    "retrieval-082",
    "retrieval-092",
    "retrieval-096",
    "retrieval-102",
    "retrieval-106",
    "retrieval-112",
]

ANSWER_SPECS = {
    "retrieval-002": {
        "reference_answer": "159101力争将日均跟踪偏离度的绝对值控制在0.2%以内，年跟踪误差控制在2%以内。",
        "key_facts": ["日均跟踪偏离度", "0.2%", "年跟踪误差", "2%"],
    },
    "retrieval-007": {
        "reference_answer": "159101跟踪国证港股通科技指数，主要采用组合复制策略和适当的替代性策略。报告期内按被动指数基金方式复制指数，并处理日常申购赎回和成份股调整；管理人认为人工智能创新推动港股科技龙头发展。",
        "key_facts": ["国证港股通科技指数", "组合复制策略", "人工智能"],
    },
    "retrieval-012": {
        "reference_answer": "159102采用完全复制、替代性、股指期货、债券、资产支持证券、融资及转融通证券出借和存托凭证等策略，业绩比较基准为经汇率调整的恒生生物科技指数收益率。",
        "key_facts": ["完全复制", "股指期货", "恒生生物科技指数收益率"],
    },
    "retrieval-022": {
        "reference_answer": "159103主要采用完全复制法跟踪中证金融科技主题指数，并在特殊情况下使用其他合理方法调整组合；目标是日均跟踪偏离度绝对值不超过0.20%，年跟踪误差不超过2%。",
        "key_facts": ["完全复制法", "0.20%", "年跟踪误差", "2%"],
    },
    "retrieval-122": {
        "reference_answer": "159201跟踪国证自由现金流指数，主要采用完全复制策略，并可采用替代性等策略。基金力争将日均跟踪偏离度的绝对值控制在0.2%以内，年跟踪误差控制在2%以内。",
        "key_facts": ["国证自由现金流指数", "0.2%", "年跟踪误差", "2%"],
    },
    "retrieval-032": {
        "reference_answer": "159105主要采用完全复制法，特殊情况下可使用成份股替代等指数投资技术。基金力争将日均跟踪偏离度绝对值控制在0.35%以内，年化跟踪误差控制在3%以内。",
        "key_facts": ["完全复制法", "0.35%", "年化跟踪误差", "3%"],
    },
    "retrieval-037": {
        "reference_answer": "159105跟踪恒生生物科技指数，该指数覆盖在香港上市且符合港股通资格的最大30家生物科技公司。报告期内主要采用完全复制法；建仓期采取相对灵活、审慎并尽量减少市场冲击的策略。",
        "key_facts": ["最大30家生物科技公司", "完全复制法", "建仓期", "审慎"],
    },
    "retrieval-042": {
        "reference_answer": "159110跟踪深证AAA科技创新公司债指数，主要采用抽样复制法。策略包括债券指数化投资、其他债券投资、资产支持证券、信用衍生品和国债期货交易，业绩比较基准为该指数收益率。",
        "key_facts": ["深证AAA科技创新公司债指数", "抽样复制法", "国债期货交易"],
    },
    "retrieval-046": {
        "reference_answer": "159110以跟踪指数为前提，通过主体资质、流动性、科创债溢价及一二级价差等维度精细化抽样复制。通常不显著偏离指数久期，仅在少数剧烈调整时降低组合久期，最大调整幅度不超过指数久期的10%。",
        "key_facts": ["精细化抽样复制", "组合久期", "指数久期的10%"],
    },
    "retrieval-052": {
        "reference_answer": "159111主要采用分层抽样复制和动态最优化方法，选择有代表性和流动性的成份券、备选成份券或替代券构建组合。目标是日均跟踪偏离度绝对值不超过0.20%，年化跟踪误差控制在2%以内。",
        "key_facts": ["分层抽样复制", "动态最优化", "0.20%", "2%"],
    },
    "retrieval-056": {
        "reference_answer": "159111四季度以跟踪指数、降低波动为目标，非现金资产中的成份券占比基本保持在100%附近，没有跟随市场热点加杠杆，而是审慎调整仓位。同期业绩比较基准增长率为0.57%。",
        "key_facts": ["成份券占比基本保持在100%附近", "审慎调整仓位", "0.57%"],
    },
    "retrieval-066": {
        "reference_answer": "159112认为2025年信用债收益率整体震荡，信用利差以震荡压缩为主，超长信用债利差部分走阔。基金建仓期以跟踪中证AAA科技创新公司债指数为核心，通过抽样复制策略构建组合并动态优化资产结构。",
        "key_facts": ["信用利差", "建仓期", "抽样复制策略"],
    },
    "retrieval-072": {
        "reference_answer": "159115的业绩比较基准为中证AAA科技创新公司债指数收益率，可采用债券投资、国债期货投资和资产支持证券投资策略。",
        "key_facts": ["中证AAA科技创新公司债指数收益率", "国债期货投资", "资产支持证券"],
    },
    "retrieval-076": {
        "reference_answer": "159115认为2025年国债收益率曲线陡峭化上行，其中30年期国债利率上行35个基点至2.27%。基金作为指数类产品，按照指数结构进行配置与交易。",
        "key_facts": ["30年期国债利率", "曲线陡峭化", "指数结构"],
    },
    "retrieval-082": {
        "reference_answer": "159125采用完全复制法，并在流动性不足等特殊情况下使用成份券替代等方法。基金力争将日均跟踪偏离度绝对值控制在0.35%以内，年化跟踪误差控制在4%以内。",
        "key_facts": ["完全复制法", "0.35%", "4%"],
    },
    "retrieval-092": {
        "reference_answer": "159126跟踪中证港股通50指数，主要采用完全复制策略、替代策略及其他适当策略。目标是日均跟踪偏离度绝对值不超过0.35%，年跟踪误差不超过4%。",
        "key_facts": ["中证港股通50指数", "0.35%", "4%"],
    },
    "retrieval-096": {
        "reference_answer": "159126使用指数化交易系统、日内择时交易模型、跟踪误差归因分析系统和ETF现金流精算系统控制跟踪误差，并针对建仓和申购赎回导致的偏离进行管理。同期业绩比较基准增长率为-1.18%。",
        "key_facts": ["指数化交易系统", "日内择时交易模型", "-1.18%"],
    },
    "retrieval-102": {
        "reference_answer": "159128采用完全复制策略，目标是日均跟踪偏离度控制在0.35%以内、年化跟踪误差控制在4%以内。由于主要投资香港证券市场，还需承担汇率风险和境外市场风险。",
        "key_facts": ["完全复制策略", "0.35%", "4%", "汇率风险"],
    },
    "retrieval-106": {
        "reference_answer": "159128以被动复制方式完全复制标的指数。报告期跟踪误差主要来自申购赎回导致的仓位偏离、指数样本股调整导致的结构偏离和新股申购；同期业绩比较基准增长率为-11.03%。",
        "key_facts": ["申购赎回", "样本股", "跟踪误差", "-11.03%"],
    },
    "retrieval-112": {
        "reference_answer": "159150跟踪深证50指数，主要采用完全复制法，特殊情况下可使用成份股替代等方法。目标是日均跟踪偏离度绝对值控制在0.2%以内，年跟踪误差控制在2%以内。",
        "key_facts": ["深证50指数", "完全复制法", "0.2%", "2%"],
    },
}

FUND_QUERY_MARKERS = {
    "159101": ["华夏国证港股通科技ETF"],
    "159102": ["华安恒生生物科技ETF"],
    "159103": ["汇添富中证金融科技主题ETF"],
    "159105": ["易方达恒生生物科技ETF"],
    "159110": ["万家深证AAA科创债ETF", "科创债ETF万家"],
    "159111": ["天弘中证AAA科创债ETF", "天弘科创债ETF"],
    "159112": ["银华科创债ETF"],
    "159115": ["华安中证AAA科创债ETF", "华安科创债ETF"],
    "159125": ["招商国证港股通科技ETF"],
    "159126": ["南方中证港股通50ETF", "南方港股通50ETF"],
    "159128": ["天弘中证港股通科技ETF"],
    "159150": ["易方达深证50ETF"],
    "159201": ["华夏国证自由现金流ETF"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--retrieval", type=Path, default=DEFAULT_RETRIEVAL_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_ANSWER_PATH)
    parser.add_argument("--milvus-uri", default="http://localhost:19595")
    parser.add_argument("--collection", default="fund_reports_mineru")
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def fetch_chunks(client: MilvusClient, collection: str, ids: list[str]) -> dict[str, dict]:
    quoted_ids = ", ".join(json.dumps(chunk_id) for chunk_id in ids)
    rows = client.query(
        collection_name=collection,
        filter=f"id in [{quoted_ids}]",
        output_fields=["id", "fund_code", "content", "chunk_index"],
        limit=max(len(ids), 1),
    )
    return {row["id"]: row for row in rows}


def validate_single_fund_query(source: dict) -> None:
    fund_code = source["filter_fund_code"]
    query = source["query"]
    markers = [fund_code, *FUND_QUERY_MARKERS.get(fund_code, [])]
    if not any(marker in query for marker in markers):
        raise ValueError(
            f"{source['id']} query does not explicitly identify fund {fund_code}: {query}"
        )


def build_strategy_examples(
    retrieval_rows: list[dict],
    chunks_by_id: dict[str, dict],
) -> list[dict]:
    retrieval_by_id = {row["id"]: row for row in retrieval_rows}
    examples = []

    for sequence, retrieval_id in enumerate(SELECTED_RETRIEVAL_IDS, 1):
        source = retrieval_by_id.get(retrieval_id)
        if source is None:
            raise ValueError(f"Missing retrieval sample: {retrieval_id}")
        spec = ANSWER_SPECS[retrieval_id]
        fund_code = source["filter_fund_code"]
        validate_single_fund_query(source)

        selected_chunks = []
        for chunk_id in source["relevant_chunk_ids"]:
            chunk = chunks_by_id.get(chunk_id)
            if chunk is None:
                raise ValueError(f"Milvus is missing chunk {chunk_id} for {retrieval_id}")
            if chunk.get("fund_code") != fund_code:
                raise ValueError(
                    f"{retrieval_id} chunk {chunk_id} belongs to {chunk.get('fund_code')}"
                )
            selected_chunks.append(chunk)

        combined_content = "\n".join(chunk.get("content", "") for chunk in selected_chunks)
        missing_keywords = [
            keyword
            for keyword in source.get("relevant_keywords", [])
            if keyword not in combined_content
        ]
        if missing_keywords:
            raise ValueError(f"{retrieval_id} is missing keywords: {missing_keywords}")

        chunk_note = "；".join(
            f"chunk={chunk['chunk_index']} id={chunk['id']}" for chunk in selected_chunks
        )
        source_note = source["note"].split("；chunk=", 1)[0]
        examples.append(
            {
                "id": f"answer-{sequence:03d}",
                "query": source["query"],
                "reference_answer": spec["reference_answer"],
                "expected_fund_codes": source["expected_fund_codes"],
                "key_facts": spec["key_facts"],
                "should_refuse": False,
                "intent": "fund_query",
                "category": "single_fund_strategy",
                "relevant_keywords": source["relevant_keywords"],
                "relevant_chunk_ids": source["relevant_chunk_ids"],
                "expected_tool_calls": [
                    {
                        "name": "rag_search",
                        "args": {"filter_fund_code": fund_code},
                    }
                ],
                "note": f"来源 {retrieval_id}；{source_note}；{chunk_note}",
            }
        )

    return examples


def preserve_other_examples(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [
        row
        for row in load_jsonl(path)
        if row.get("category") != "single_fund_strategy"
    ]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for sequence, row in enumerate(rows, 1):
            row["id"] = f"answer-{sequence:03d}"
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    retrieval_rows = load_jsonl(args.retrieval)
    selected_rows = {
        row["id"]: row
        for row in retrieval_rows
        if row["id"] in SELECTED_RETRIEVAL_IDS
    }
    chunk_ids = [
        chunk_id
        for retrieval_id in SELECTED_RETRIEVAL_IDS
        for chunk_id in selected_rows[retrieval_id]["relevant_chunk_ids"]
    ]

    client = MilvusClient(uri=args.milvus_uri)
    chunks_by_id = fetch_chunks(client, args.collection, chunk_ids)
    strategy_examples = build_strategy_examples(retrieval_rows, chunks_by_id)
    other_examples = preserve_other_examples(args.output)
    write_jsonl(args.output, strategy_examples + other_examples)
    print(
        f"Wrote {len(strategy_examples)} strategy examples and "
        f"{len(other_examples)} preserved examples to {args.output}"
    )


if __name__ == "__main__":
    main()
