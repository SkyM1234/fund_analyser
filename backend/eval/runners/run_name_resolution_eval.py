"""跑基金名称识别评测：两级RAG第一级（rag_identify_funds 底层 /fund_index/search）。

用法：
    python -m eval.runners.run_name_resolution_eval
    python -m eval.runners.run_name_resolution_eval --concurrency 8
    python -m eval.runners.run_name_resolution_eval --experiment-prefix v2

会在 LangSmith 创建一次 Experiment，并把结果同步到 reports/ 落盘。
"""
from __future__ import annotations

import argparse
import asyncio
from calendar import c
import json
import logging
from datetime import datetime
from pathlib import Path

from langsmith import Client
from langsmith.evaluation import aevaluate

from eval.config import get_eval_settings
from eval.evaluators import name_resolution_metrics as nrm
from eval.targets.name_resolution_target import name_resolution_target
import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

REPORT_DIR = Path(__file__).resolve().parent.parent / "reports"
REPORT_DIR.mkdir(exist_ok=True)


async def run(experiment_prefix: str, concurrency: int):
    s = get_eval_settings()
    s.export_langsmith_env()
    client = Client()

    evaluators = [
        nrm.top1_accuracy,
        nrm.hit_at_k,
        nrm.miss_rate,
        nrm.false_positive_rate,
    ]

    logger.info(f"开始名称识别评测：dataset={s.DATASET_NAME_RESOLUTION_NAME} concurrency={concurrency}")
    results = await aevaluate(
        name_resolution_target,
        data=s.DATASET_NAME_RESOLUTION_NAME,
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
    out_path = REPORT_DIR / f"name-resolution-{experiment_prefix}-{stamp}.json"
    out_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(f"✓ 报告已落盘: {out_path}")

    # 简单聚合（跳过 None，即"该指标不适用该样本"的情况）
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
    parser.add_argument("--experiment-prefix", default="name-resolution")
    parser.add_argument("--concurrency", type=int, default=None)
    args = parser.parse_args()

    s = get_eval_settings()
    asyncio.run(
        run(
            experiment_prefix=args.experiment_prefix,
            concurrency=args.concurrency or s.EVAL_MAX_CONCURRENCY,
        )
    )


if __name__ == "__main__":
    main()
