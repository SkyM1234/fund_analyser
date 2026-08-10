"""把本地 jsonl 数据集上传到 LangSmith。

用法：
    python -m eval.runners.upload_dataset --kind retrieval
    python -m eval.runners.upload_dataset --kind answer
    python -m eval.runners.upload_dataset --kind name_resolution
    python -m eval.runners.upload_dataset --kind all     # 全部三种

若 LangSmith 上同名数据集已存在：
    - --mode append: 追加新样本（按 id 去重）
    - --mode replace: 删除旧的，全量重建
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from langsmith import Client
from pydantic import ValidationError

from eval.config import get_eval_settings
from eval.schemas import AnswerExample, NameResolutionExample, RetrievalExample

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent.parent / "datasets"


def _load_jsonl(path: Path, schema_cls) -> list:
    items = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                items.append(schema_cls.model_validate_json(line))
            except ValidationError as e:
                logger.error(f"{path.name}:{line_no} schema 校验失败: {e}")
                raise
    return items


def _split_inputs_outputs(item, schema_kind: str) -> tuple[dict, dict]:
    """把单个样本拆成 LangSmith 需要的 inputs / outputs。

    输入：被评测系统真正消费的字段
    输出：ground truth（评测器读取）
    """
    data = item.model_dump()
    if schema_kind == "retrieval":
        inputs = {
            "query": data["query"],
            "filter_fund_code": data.get("filter_fund_code", ""),
            "top_k": data.get("top_k", 10),
        }
        outputs = {
            "expected_fund_codes": data.get("expected_fund_codes", []),
            "relevant_chunk_ids": data.get("relevant_chunk_ids", []),
            "relevant_keywords": data.get("relevant_keywords", []),
            "filter_fund_code": data.get("filter_fund_code", ""),
        }
    elif schema_kind == "name_resolution":
        inputs = {
            "query": data["query"],
            "top_k": data.get("top_k", 5),
            "min_score": data.get("min_score", 0.5),
        }
        outputs = {
            "expected_fund_code": data.get("expected_fund_code"),
        }
    else:  # answer
        inputs = {"query": data["query"]}
        outputs = {
            "reference_answer": data.get("reference_answer", ""),
            "expected_fund_codes": data.get("expected_fund_codes", []),
            "key_facts": data.get("key_facts", []),
            "should_refuse": data.get("should_refuse", False),
            "intent": data.get("intent"),
            # 可选的检索 ground truth（用于 Agent RAG 与直接 RAG 对比）
            "relevant_chunk_ids": data.get("relevant_chunk_ids", []),
            "relevant_keywords": data.get("relevant_keywords", []),
            # 可选的工具调用 ground truth（用于 tool_call_accuracy）
            "expected_tool_calls": data.get("expected_tool_calls", []),
        }
    return inputs, outputs


def _ensure_dataset(client: Client, name: str, description: str, mode: str):
    """根据 mode 创建 / 复用 / 重建数据集。"""
    existing = list(client.list_datasets(dataset_name=name))
    if not existing:
        return client.create_dataset(name, description=description)
    ds = existing[0]
    if mode == "replace":
        logger.warning(f"replace 模式：删除现有数据集 {name}")
        client.delete_dataset(dataset_id=ds.id)
        return client.create_dataset(name, description=description)
    return ds


def _existing_ids(client: Client, dataset_id) -> set[str]:
    """读出已存在样本的 id 字段，用于增量去重。"""
    ids = set()
    for ex in client.list_examples(dataset_id=dataset_id):
        ref_id = (ex.metadata or {}).get("id")
        if ref_id:
            ids.add(ref_id)
    return ids


def upload(kind: str, mode: str = "append") -> None:
    settings = get_eval_settings()
    settings.prepare_runtime()
    client = Client()

    targets = []
    if kind in ("retrieval", "all"):
        targets.append(("retrieval", DATA_DIR / "retrieval.jsonl", RetrievalExample, settings.DATASET_RETRIEVAL_NAME))
    if kind in ("answer", "all"):
        targets.append(("answer", DATA_DIR / "answer.jsonl", AnswerExample, settings.DATASET_ANSWER_NAME))
    if kind in ("name_resolution", "all"):
        targets.append(("name_resolution", DATA_DIR / "fund_name_resolution.jsonl", NameResolutionExample, settings.DATASET_NAME_RESOLUTION_NAME))

    for schema_kind, path, schema_cls, ds_name in targets:
        if not path.exists():
            logger.error(f"数据文件不存在: {path}")
            continue
        items = _load_jsonl(path, schema_cls)
        logger.info(f"[{schema_kind}] 加载 {len(items)} 条样本: {path.name}")

        ds = _ensure_dataset(client, ds_name, f"{schema_kind} eval dataset", mode)
        existing = _existing_ids(client, ds.id) if mode == "append" else set()

        new_count = 0
        for item in items:
            if item.id in existing:
                continue
            inputs, outputs = _split_inputs_outputs(item, schema_kind)
            client.create_example(
                inputs=inputs,
                outputs=outputs,
                dataset_id=ds.id,
                metadata={"id": item.id, "category": item.category or "", "note": item.note or ""},
            )
            new_count += 1
        logger.info(f"[{schema_kind}] 上传 {new_count} 条新样本到 {ds_name}（已跳过 {len(items) - new_count} 条重复）")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--kind",
        choices=["retrieval", "answer", "name_resolution", "all"],
        default="all",
    )
    parser.add_argument("--mode", choices=["append", "replace"], default="append")
    args = parser.parse_args()
    upload(args.kind, args.mode)


if __name__ == "__main__":
    main()
