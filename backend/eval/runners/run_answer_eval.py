"""跑 Agent 端到端评测。

用法：
    python -m eval.runners.run_answer_eval
    python -m eval.runners.run_answer_eval --no-judge --concurrency 2
    python -m eval.runners.run_answer_eval --experiment-prefix v2-multi-agent
    python -m eval.runners.run_answer_eval --no-langsmith --concurrency 2

默认通过 Docker 后端的 /api/chat/stream 评测完整服务链路，避免在评测进程内
重复初始化 LangGraph、MCP 和模型客户端。

服务模式的有效并发受 Celery worker 数量限制。
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
from eval.evaluators import answer_metrics as am
from eval.evaluators import llm_judge as judge
from eval.evaluators import retrieval_metrics as rm
from eval.reporting import build_aggregate_report
from eval.schemas import AnswerExample
from eval.targets.service_target import AnswerServiceTarget

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

REPORT_DIR = Path(__file__).resolve().parent.parent / "reports" / "answer"
DATASET_PATH = Path(__file__).resolve().parent.parent / "datasets" / "answer_cross_fund.jsonl" 
# answer_single_fund.jsonl 是单基金回答评测数据集，包含 50 条样本
# answer_cross_fund.jsonl 是跨基金回答评测数据集，包含 50 条样本


def _get_evaluators(use_judge: bool) -> list:
    evaluators = [
        am.citation_accuracy,
        am.refusal_correctness,
        am.key_fact_coverage,
        am.intent_accuracy,
        am.tool_call_accuracy,
        rm.hit_rate,
        rm.session_mrr,
        rm.session_ndcg,
    ]
    if use_judge:
        evaluators.extend([judge.correctness, judge.answer_relevance])
    return evaluators


async def _run_langsmith(
    experiment_prefix: str,
    concurrency: int,
    use_judge: bool,
    target,
) -> list[dict]:
    from langsmith import Client
    from langsmith.evaluation import aevaluate

    s = get_eval_settings()
    s.prepare_runtime(use_judge=use_judge)
    client = Client()
    evaluators = _get_evaluators(use_judge)

    logger.info(f"开始单基金回答评测：dataset={s.DATASET_SINGLE_FUND_ANSWER_NAME} concurrency={concurrency}")
    results = await aevaluate(
        target,
        data=s.DATASET_SINGLE_FUND_ANSWER_NAME,
        evaluators=evaluators,
        experiment_prefix=experiment_prefix,
        max_concurrency=concurrency,
        client=client,
    )

    rows = []
    async for r in results:
        run_obj = r["run"]
        outputs = getattr(run_obj, "outputs", {}) or {}
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
                "query": (getattr(r["example"], "inputs", {}) or {}).get("query"),
                "answer": outputs.get("answer"),
                "intent": outputs.get("intent"),
                "cited_fund_codes": outputs.get("cited_fund_codes"),
                "tool_calls": outputs.get("tool_calls"),
                "retrieved_chunk_ids": outputs.get("retrieved_chunk_ids"),
                "retrieved_chunk_scores": outputs.get("retrieved_chunk_scores"),
                "scores": scores,
            }
        )
    return rows


def _load_local_examples() -> list[AnswerExample]:
    examples = []
    with DATASET_PATH.open("r", encoding="utf-8") as dataset:
        for line_no, line in enumerate(dataset, 1):
            line = line.strip()
            if not line:
                continue
            try:
                if hasattr(AnswerExample, "model_validate_json"):
                    example = AnswerExample.model_validate_json(line)
                else:
                    example = AnswerExample.parse_raw(line)
            except Exception as exc:
                raise ValueError(f"{DATASET_PATH.name}:{line_no} 数据格式错误: {exc}") from exc
            examples.append(example)
    return examples


def _format_jsonl_row(
    example: AnswerExample,
    outputs: dict,
    scores: dict,
) -> dict:
    row = {
        "example_id": example.id,
        "query": example.query,
        "answer": outputs.get("answer"),
        "intent": outputs.get("intent"),
        "cited_fund_codes": outputs.get("cited_fund_codes"),
        "tool_calls": outputs.get("tool_calls"),
        "retrieved_chunk_ids": outputs.get("retrieved_chunk_ids"),
        "retrieved_chunk_scores": outputs.get("retrieved_chunk_scores"),
        "scores": scores,
    }
    if outputs.get("error"):
        row["error"] = outputs["error"]
    return row


async def _evaluate_jsonl_example(
    example: AnswerExample,
    semaphore: asyncio.Semaphore,
    evaluators: list,
    target,
) -> dict:
    inputs = {"query": example.query}
    expected_outputs = {
        "reference_answer": example.reference_answer,
        "expected_fund_codes": example.expected_fund_codes,
        "key_facts": example.key_facts,
        "should_refuse": example.should_refuse,
        "intent": example.intent,
        "relevant_chunk_ids": example.relevant_chunk_ids,
        "relevant_keywords": example.relevant_keywords,
        "expected_tool_calls": example.expected_tool_calls,
    }

    async with semaphore:
        try:
            outputs = await target(inputs)
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
            return _format_jsonl_row(example, {"error": str(exc)}, {})

    return _format_jsonl_row(example, outputs, scores)


async def _run_jsonl(concurrency: int, use_judge: bool, target) -> list[dict]:
    os.environ["LANGSMITH_TRACING"] = "false"
    os.environ["LANGCHAIN_TRACING_V2"] = "false"

    if use_judge:
        get_eval_settings().require_judge_api_key()

    examples = _load_local_examples()
    evaluators = _get_evaluators(use_judge)
    logger.info(
        "开始 JSONL 回答评测：dataset=%s samples=%d concurrency=%d judge=%s target=HTTP-service",
        DATASET_PATH,
        len(examples),
        concurrency,
        use_judge,
    )

    semaphore = asyncio.Semaphore(concurrency)
    tasks = [
        asyncio.create_task(_evaluate_jsonl_example(example, semaphore, evaluators, target))
        for example in examples
    ]

    rows = []
    for completed, task in enumerate(asyncio.as_completed(tasks), 1):
        rows.append(await task)
        if completed % 10 == 0 or completed == len(tasks):
            logger.info("JSONL 评测进度：%d/%d", completed, len(tasks))

    order = {example.id: index for index, example in enumerate(examples)}
    rows.sort(key=lambda row: order[row["example_id"]])
    return rows


def _log_aggregate(aggregate: dict) -> None:
    logger.info("聚合得分：")
    for key, metric in aggregate["metrics"].items():
        logger.info("  %-24s: mean=%.3f n=%d", key, metric["mean"], metric["count"])
    logger.info(
        "样本：total=%d successful=%d failed=%d",
        aggregate["sample_count"],
        aggregate["successful_sample_count"],
        aggregate["failed_sample_count"],
    )


async def run(
    experiment_prefix: str,
    concurrency: int,
    use_judge: bool,
    no_langsmith: bool = False,
    service_base_url: str | None = None,
    service_username: str | None = None,
    service_password: str | None = None,
    service_timeout: float | None = None,
    service_auto_register: bool = True,
):
    if concurrency < 1:
        raise ValueError("concurrency 必须大于等于 1")

    settings = get_eval_settings()
    service_target = AnswerServiceTarget(
        base_url=service_base_url or settings.ANSWER_SERVICE_BASE_URL,
        username=service_username or settings.ANSWER_SERVICE_USERNAME,
        password=service_password or settings.ANSWER_SERVICE_PASSWORD,
        timeout_seconds=service_timeout or settings.ANSWER_SERVICE_TIMEOUT_SECONDS,
        auto_register=service_auto_register,
        identity_pool_size=concurrency,
    )
    await service_target.start()
    logger.info("回答 target：Docker HTTP 服务 %s", service_base_url or settings.ANSWER_SERVICE_BASE_URL)

    try:
        if no_langsmith:
            rows = await _run_jsonl(concurrency, use_judge, service_target)
        else:
            rows = await _run_langsmith(experiment_prefix, concurrency, use_judge, service_target)
    finally:
        await service_target.aclose()

    generated_at = datetime.now()
    stamp = generated_at.strftime("%Y%m%d-%H%M%S")
    suffix = f"-jsonl-{stamp}" if no_langsmith else f"-{stamp}"
    out_path = REPORT_DIR / f"answer-{experiment_prefix}{suffix}.json"
    summary_path = out_path.with_name(f"{out_path.stem}-summary.json")
    aggregate = build_aggregate_report(
        report_type="answer",
        experiment_prefix=experiment_prefix,
        run_mode="jsonl" if no_langsmith else "langsmith",
        rows=rows,
        generated_at=generated_at,
    )
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    summary_path.write_text(json.dumps(aggregate, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(f"[OK] 报告已落盘: {out_path}")
    logger.info(f"[OK] 聚合得分已落盘: {summary_path}")
    _log_aggregate(aggregate)


def main():
    settings = get_eval_settings()
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-prefix", default="answer")
    parser.add_argument("--concurrency", type=int, default=1, help="评测并发数")
    parser.add_argument("--no-judge", action="store_true", help="跳过 LLM-judge 指标（只跑规则指标）")
    parser.add_argument(
        "--service-base-url",
        default=settings.ANSWER_SERVICE_BASE_URL,
        help="Docker 后端地址",
    )
    parser.add_argument(
        "--service-username",
        default=settings.ANSWER_SERVICE_USERNAME,
        help="服务评测用户",
    )
    parser.add_argument(
        "--service-password",
        default=settings.ANSWER_SERVICE_PASSWORD,
        help="服务评测用户密码",
    )
    parser.add_argument(
        "--service-timeout",
        type=float,
        default=settings.ANSWER_SERVICE_TIMEOUT_SECONDS,
        help="单条回答的 HTTP/SSE 超时秒数",
    )
    parser.add_argument(
        "--no-service-auto-register",
        action="store_true",
        help="不自动注册服务评测用户，只尝试登录",
    )
    parser.add_argument(
        "--no-langsmith",
        action="store_true",
        help="完全绕过 LangSmith，读取本地 JSONL 执行评测并生成本地报告",
    )
    args = parser.parse_args()

    if args.service_timeout <= 0:
        parser.error("--service-timeout 必须大于 0")

    asyncio.run(
        run(
            experiment_prefix=args.experiment_prefix,
            concurrency=args.concurrency,
            use_judge=not args.no_judge,
            no_langsmith=args.no_langsmith,
            service_base_url=args.service_base_url,
            service_username=args.service_username,
            service_password=args.service_password,
            service_timeout=args.service_timeout,
            service_auto_register=not args.no_service_auto_register,
        )
    )


if __name__ == "__main__":
    main()
