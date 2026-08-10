"""跑基金名称识别评测：两级RAG第一级（rag_identify_funds 底层 /fund_index/search）。

用法：
    python -m eval.runners.run_name_resolution_eval
    python -m eval.runners.run_name_resolution_eval --concurrency 8
    python -m eval.runners.run_name_resolution_eval --experiment-prefix v2
    python -m eval.runners.run_name_resolution_eval --concurrency 8 --no-langsmith

默认会在 LangSmith 创建一次 Experiment，并把结果同步到 reports/ 落盘。
--no-langsmith 模式直接读取本地 JSONL、计算指标并落盘，不访问 LangSmith。
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

from eval.config import get_eval_settings
from eval.evaluators import name_resolution_metrics as nrm
from eval.schemas import NameResolutionExample
from eval.targets.name_resolution_target import name_resolution_target

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

REPORT_DIR = Path(__file__).resolve().parent.parent / "reports"
REPORT_DIR.mkdir(exist_ok=True)
DATASET_PATH = Path(__file__).resolve().parent.parent / "datasets" / "fund_name_resolution.jsonl"


EVALUATORS = [
    nrm.top1_accuracy,
    nrm.hit_at_k,
    nrm.miss_rate,
    nrm.false_positive_rate,
]


async def _run_langsmith(experiment_prefix: str, concurrency: int) -> list[dict]:
    from langsmith import Client
    from langsmith.evaluation import aevaluate

    s = get_eval_settings()
    s.prepare_runtime()
    client = Client()
    logger.info(f"开始名称识别评测：dataset={s.DATASET_NAME_RESOLUTION_NAME} concurrency={concurrency}")
    results = await aevaluate(
        name_resolution_target,
        data=s.DATASET_NAME_RESOLUTION_NAME,
        evaluators=EVALUATORS,
        experiment_prefix=experiment_prefix,
        max_concurrency=concurrency,
        client=client,
    )

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
    return rows


def _load_local_examples() -> list[NameResolutionExample]:
    examples = []
    with DATASET_PATH.open("r", encoding="utf-8") as dataset:
        for line_no, line in enumerate(dataset, 1):
            line = line.strip()
            if not line:
                continue
            try:
                if hasattr(NameResolutionExample, "model_validate_json"):
                    example = NameResolutionExample.model_validate_json(line)
                else:
                    example = NameResolutionExample.parse_raw(line)
            except Exception as exc:
                raise ValueError(f"{DATASET_PATH.name}:{line_no} 数据格式错误: {exc}") from exc
            examples.append(example)
    return examples


async def _evaluate_local_example(
    example: NameResolutionExample,
    semaphore: asyncio.Semaphore,
) -> dict:
    inputs = {
        "query": example.query,
        "top_k": example.top_k,
        "min_score": example.min_score,
    }
    expected_outputs = {"expected_fund_code": example.expected_fund_code}

    async with semaphore:
        try:
            outputs = await name_resolution_target(inputs)
        except Exception as exc:
            logger.error("样本 %s 调用失败: %s", example.id, exc)
            return {
                "example_id": example.id,
                "inputs": inputs,
                "outputs": {"error": str(exc)},
                "scores": {},
            }

    run_obj = SimpleNamespace(outputs=outputs)
    example_obj = SimpleNamespace(outputs=expected_outputs)
    scores = {}
    for evaluator in EVALUATORS:
        result = evaluator(run_obj, example_obj)
        scores[result["key"]] = {
            "score": result.get("score"),
            "comment": result.get("comment", ""),
        }

    return {
        "example_id": example.id,
        "inputs": inputs,
        "outputs": outputs,
        "scores": scores,
    }


async def _run_local(concurrency: int) -> list[dict]:
    os.environ["LANGSMITH_TRACING"] = "false"
    os.environ["LANGCHAIN_TRACING_V2"] = "false"

    examples = _load_local_examples()
    logger.info(
        "开始本地名称识别评测：dataset=%s samples=%d concurrency=%d LangSmith=disabled",
        DATASET_PATH,
        len(examples),
        concurrency,
    )

    semaphore = asyncio.Semaphore(concurrency)
    tasks = [
        asyncio.create_task(_evaluate_local_example(example, semaphore))
        for example in examples
    ]

    rows = []
    for completed, task in enumerate(asyncio.as_completed(tasks), 1):
        rows.append(await task)
        if completed % 25 == 0 or completed == len(tasks):
            logger.info("本地评测进度：%d/%d", completed, len(tasks))

    order = {example.id: index for index, example in enumerate(examples)}
    rows.sort(key=lambda row: order[row["example_id"]])
    return rows


def _log_aggregate(rows: list[dict]) -> None:
    if not rows:
        return

    agg: dict[str, list[float]] = {}
    for row in rows:
        for key, value in row["scores"].items():
            if isinstance(value.get("score"), (int, float)):
                agg.setdefault(key, []).append(float(value["score"]))

    logger.info("聚合得分：")
    for key, values in agg.items():
        logger.info("  %-24s: mean=%.3f n=%d", key, sum(values) / len(values), len(values))


async def run(experiment_prefix: str, concurrency: int, no_langsmith: bool = False):
    if concurrency < 1:
        raise ValueError("concurrency 必须大于等于 1")

    if no_langsmith:
        rows = await _run_local(concurrency)
    else:
        rows = await _run_langsmith(experiment_prefix, concurrency)

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    suffix = f"-local-{stamp}" if no_langsmith else f"-{stamp}"
    out_path = REPORT_DIR / f"name-resolution-{experiment_prefix}{suffix}.json"
    out_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(f"[OK] 报告已落盘: {out_path}")
    _log_aggregate(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-prefix", default="name-resolution")
    parser.add_argument("--concurrency", type=int, default=None)
    parser.add_argument(
        "--no-langsmith",
        action="store_true",
        help="完全绕过 LangSmith，读取本地 JSONL 执行评测并生成本地报告",
    )
    args = parser.parse_args()

    s = get_eval_settings()
    asyncio.run(
        run(
            experiment_prefix=args.experiment_prefix,
            concurrency=args.concurrency or s.EVAL_MAX_CONCURRENCY,
            no_langsmith=args.no_langsmith,
        )
    )


if __name__ == "__main__":
    main()
