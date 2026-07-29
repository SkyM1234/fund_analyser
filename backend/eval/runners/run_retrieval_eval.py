"""跑 RAG 检索评测：rag_search 直连 GPU。

用法：
    python -m eval.runners.run_retrieval_eval
    python -m eval.runners.run_retrieval_eval --no-judge          # 跳过 LLM-judge
    python -m eval.runners.run_retrieval_eval --concurrency 8
    python -m eval.runners.run_retrieval_eval --experiment-prefix v2-with-list-filter

会在 LangSmith 创建一次 Experiment，并把结果同步到 reports/ 落盘。
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path

from langsmith import Client
from langsmith.evaluation import aevaluate

from eval.config import get_eval_settings
from eval.evaluators import retrieval_metrics as rm
from eval.evaluators import llm_judge as judge
from eval.targets.rag_target import rag_search_target

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

REPORT_DIR = Path(__file__).resolve().parent.parent / "reports"
REPORT_DIR.mkdir(exist_ok=True)


async def run(experiment_prefix: str, concurrency: int, use_judge: bool):
    s = get_eval_settings()
    s.export_langsmith_env()
    client = Client()

    evaluators = [
        rm.hit_rate, 
        rm.mrr, 
        rm.ndcg, 
        rm.fund_code_recall
    ]
    if use_judge:
        evaluators += [
            judge.context_relevance
        ]

    logger.info(f"开始检索评测：dataset={s.DATASET_RETRIEVAL_NAME} concurrency={concurrency}")
    results = await aevaluate(
        rag_search_target,
        data=s.DATASET_RETRIEVAL_NAME,
        evaluators=evaluators,
        experiment_prefix=experiment_prefix,
        max_concurrency=concurrency,
        client=client,
    )

    # 落盘本地报告
    rows = []
    async for r in results:
        run_obj = r["run"]
        eval_results = r.get("evaluation_results", {}).get("results", [])
        scores = {}
        for res in eval_results:
            # EvaluationResult 用属性访问
            key = getattr(res, "key", None) or ""
            score = getattr(res, "score", None)
            comment = getattr(res, "comment", "") or ""
            scores[key] = {"score": score, "comment": comment}

        rows.append(
            {
                "example_id": str(getattr(r["example"], "id", "")),
                "inputs": getattr(r["example"], "inputs", {}),
                "outputs": getattr(run_obj, "outputs", {}),
                "scores": scores,
            }
        )

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_path = REPORT_DIR / f"retrieval-{experiment_prefix}-{stamp}.json"
    out_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(f"✓ 报告已落盘: {out_path}")

    # 简单聚合
    if rows:
        agg: dict[str, list[float]] = {}
        for row in rows:
            for k, v in row["scores"].items():
                if isinstance(v.get("score"), (int, float)):
                    agg.setdefault(k, []).append(float(v["score"]))
        logger.info("聚合得分：")
        for k, vs in agg.items():
            logger.info(f"  {k:24s}: mean={sum(vs)/len(vs):.3f} n={len(vs)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-prefix", default="retrieval")
    parser.add_argument("--concurrency", type=int, default=None)
    parser.add_argument("--no-judge", action="store_true", help="跳过 LLM-judge 指标")
    args = parser.parse_args()

    s = get_eval_settings()
    asyncio.run(
        run(
            experiment_prefix=args.experiment_prefix,
            concurrency=args.concurrency or s.EVAL_MAX_CONCURRENCY,
            use_judge=not args.no_judge,
        )
    )


if __name__ == "__main__":
    main()
