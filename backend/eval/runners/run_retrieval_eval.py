"""跑 RAG 检索评测：rag_search 直连 GPU。

用法：
    python -m eval.runners.run_retrieval_eval
    python -m eval.runners.run_retrieval_eval --no-judge          # 跳过 LLM-judge
    python -m eval.runners.run_retrieval_eval --concurrency 8
    python -m eval.runners.run_retrieval_eval --experiment-prefix v2-with-list-filter
    python -m eval.runners.run_retrieval_eval --concurrency 8 --no-langsmith

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
from eval.evaluators import llm_judge as judge
from eval.evaluators import retrieval_metrics as rm
from eval.schemas import RetrievalExample
from eval.targets.rag_target import rag_search_target

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

REPORT_DIR = Path(__file__).resolve().parent.parent / "reports"
REPORT_DIR.mkdir(exist_ok=True)
DATASET_PATH = Path(__file__).resolve().parent.parent / "datasets" / "retrieval.jsonl"


def _get_evaluators(use_judge: bool) -> list:
    evaluators = [
        rm.hit_rate,
        rm.mrr,
        rm.ndcg,
        rm.fund_code_recall,
    ]
    if use_judge:
        evaluators.append(judge.context_relevance)
    return evaluators


async def _run_langsmith(
    experiment_prefix: str,
    concurrency: int,
    use_judge: bool,
) -> list[dict]:
    from langsmith import Client
    from langsmith.evaluation import aevaluate

    s = get_eval_settings()
    s.prepare_runtime(use_judge=use_judge)
    client = Client()
    evaluators = _get_evaluators(use_judge)

    logger.info(f"开始检索评测：dataset={s.DATASET_RETRIEVAL_NAME} concurrency={concurrency}")
    results = await aevaluate(
        rag_search_target,
        data=s.DATASET_RETRIEVAL_NAME,
        evaluators=evaluators,
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
    return rows


def _load_local_examples() -> list[RetrievalExample]:
    examples = []
    with DATASET_PATH.open("r", encoding="utf-8") as dataset:
        for line_no, line in enumerate(dataset, 1):
            line = line.strip()
            if not line:
                continue
            try:
                if hasattr(RetrievalExample, "model_validate_json"):
                    example = RetrievalExample.model_validate_json(line)
                else:
                    example = RetrievalExample.parse_raw(line)
            except Exception as exc:
                raise ValueError(f"{DATASET_PATH.name}:{line_no} 数据格式错误: {exc}") from exc
            examples.append(example)
    return examples


async def _evaluate_local_example(
    example: RetrievalExample,
    semaphore: asyncio.Semaphore,
    evaluators: list,
) -> dict:
    inputs = {
        "query": example.query,
        "filter_fund_code": example.filter_fund_code or "",
        "top_k": example.top_k,
    }
    expected_outputs = {
        "expected_fund_codes": example.expected_fund_codes,
        "relevant_chunk_ids": example.relevant_chunk_ids,
        "relevant_keywords": example.relevant_keywords,
        "filter_fund_code": example.filter_fund_code or "",
    }

    async with semaphore:
        try:
            outputs = await rag_search_target(inputs)
            run_obj = SimpleNamespace(outputs=outputs)
            example_obj = SimpleNamespace(inputs=inputs, outputs=expected_outputs)
            scores = {}
            for evaluator in evaluators:
                result = await asyncio.to_thread(evaluator, run_obj, example_obj)
                scores[result["key"]] = {
                    "score": result.get("score"),
                    "comment": result.get("comment", ""),
                }
        except Exception as exc:
            logger.error("样本 %s 评测失败: %s", example.id, exc)
            return {
                "example_id": example.id,
                "inputs": inputs,
                "outputs": {"error": str(exc)},
                "scores": {},
            }

    return {
        "example_id": example.id,
        "inputs": inputs,
        "outputs": outputs,
        "scores": scores,
    }


async def _run_local(concurrency: int, use_judge: bool) -> list[dict]:
    os.environ["LANGSMITH_TRACING"] = "false"
    os.environ["LANGCHAIN_TRACING_V2"] = "false"

    if use_judge:
        get_eval_settings().require_judge_api_key()

    examples = _load_local_examples()
    evaluators = _get_evaluators(use_judge)
    logger.info(
        "开始本地检索评测：dataset=%s samples=%d concurrency=%d judge=%s LangSmith=disabled",
        DATASET_PATH,
        len(examples),
        concurrency,
        use_judge,
    )

    semaphore = asyncio.Semaphore(concurrency)
    tasks = [
        asyncio.create_task(_evaluate_local_example(example, semaphore, evaluators))
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


async def run(
    experiment_prefix: str,
    concurrency: int,
    use_judge: bool,
    no_langsmith: bool = False,
):
    if concurrency < 1:
        raise ValueError("concurrency 必须大于等于 1")

    if no_langsmith:
        rows = await _run_local(concurrency, use_judge)
    else:
        rows = await _run_langsmith(experiment_prefix, concurrency, use_judge)

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    suffix = f"-local-{stamp}" if no_langsmith else f"-{stamp}"
    out_path = REPORT_DIR / f"retrieval-{experiment_prefix}{suffix}.json"
    out_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(f"[OK] 报告已落盘: {out_path}")
    _log_aggregate(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-prefix", default="retrieval")
    parser.add_argument("--concurrency", type=int, default=None)
    parser.add_argument("--no-judge", action="store_true", help="跳过 LLM-judge 指标")
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
            use_judge=not args.no_judge,
            no_langsmith=args.no_langsmith,
        )
    )


if __name__ == "__main__":
    main()
