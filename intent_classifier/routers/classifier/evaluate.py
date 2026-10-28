"""Evaluate a trained intent classifier against a separate JSONL test set."""
import argparse
import hashlib
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from time import perf_counter

from .common import LABELS, LABEL2ID, ROOT, encode_sample, fingerprint, read_jsonl
from intent_classifier.common import REPORTS_ROOT


def audit_test_data(samples, reference_files):
    missing = set(LABELS) - {row["intent"] for row in samples}
    if missing:
        raise ValueError(f"测试集缺少类别：{sorted(missing)}")
    ids = [row.get("id", f"line-{index}") for index, row in enumerate(samples, 1)]
    if len(set(ids)) != len(ids):
        raise ValueError("测试集存在重复 id")
    inputs = {fingerprint(row) for row in samples}
    groups = {row["group_id"] for row in samples if "group_id" in row}
    references = []
    for path in dict.fromkeys(Path(path).resolve() for path in reference_files):
        reference = read_jsonl(path)
        input_overlap = inputs & {fingerprint(row) for row in reference}
        group_overlap = groups & {row["group_id"] for row in reference if "group_id" in row}
        if input_overlap or group_overlap:
            raise ValueError(
                f"测试集与 {path} 重叠：相同输入 {len(input_overlap)} 条，"
                f"相同 group_id {len(group_overlap)} 个"
            )
        references.append({"path": str(path), "sample_count": len(reference),
                           "exact_input_overlap": 0, "group_overlap": 0})
    return {
        "sample_count": len(samples),
        "class_counts": dict(Counter(row["intent"] for row in samples)),
        "with_history": sum(bool(row["history"]) for row in samples),
        "without_history": sum(not row["history"] for row in samples),
        "references": references,
    }


def compute_report(predictions):
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

    truth = [row["expected_intent"] for row in predictions]
    predicted = [row["predicted_intent"] for row in predictions]

    def subset(rows):
        return {"sample_count": len(rows),
                "correct_count": sum(row["correct"] for row in rows),
                "accuracy": sum(row["correct"] for row in rows) / len(rows) if rows else None}

    tags = sorted({tag for row in predictions for tag in row.get("tags", [])})
    return {
        "sample_count": len(predictions),
        "correct_count": sum(row["correct"] for row in predictions),
        "error_count": sum(not row["correct"] for row in predictions),
        "accuracy": accuracy_score(truth, predicted),
        "macro_f1": f1_score(truth, predicted, labels=LABELS, average="macro", zero_division=0),
        "weighted_f1": f1_score(truth, predicted, labels=LABELS, average="weighted", zero_division=0),
        "classification_report": classification_report(
            truth, predicted, labels=LABELS, output_dict=True, zero_division=0,
        ),
        "confusion_matrix": {
            "labels": LABELS, "rows": "expected_intent", "columns": "predicted_intent",
            "values": confusion_matrix(truth, predicted, labels=LABELS).tolist(),
        },
        "by_context": {
            "with_history": subset([row for row in predictions if row["history"]]),
            "without_history": subset([row for row in predictions if not row["history"]]),
        },
        "by_tag": {tag: subset([row for row in predictions if tag in row.get("tags", [])])
                   for tag in tags},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, help="train.py 导出的 best 模型目录")
    parser.add_argument("--test-file", type=Path, default=ROOT / "intent_classifier/routers/classifier/data/test.jsonl")
    parser.add_argument("--reference-file", type=Path, action="append", default=[],
                        help="额外检查是否重叠的训练/验证文件，可重复指定")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--warmup-steps", type=int, default=3)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--check-data-only", action="store_true")
    args = parser.parse_args()
    if not args.check_data_only and args.model_path is None:
        parser.error("评测时必须指定 --model-path，或使用 --check-data-only 只检查数据")
    if args.warmup_steps < 0:
        parser.error("warmup-steps 不能为负数")
    samples = read_jsonl(args.test_file)
    reference_files = [ROOT / "intent_classifier/routers/classifier/data/demo.jsonl", *args.reference_file]
    if args.model_path:
        # The trainer saves both actual splits beside the exported best model.
        for name in ["train_split.jsonl", "validation_split.jsonl"]:
            path = args.model_path.parent / name
            if path.is_file():
                reference_files.append(path)
    audit = audit_test_data(samples, reference_files)
    if args.check_data_only:
        print(json.dumps(audit, ensure_ascii=False, indent=2))
        return

    config_path = args.model_path / "intent_classifier_config.json"
    if not config_path.is_file():
        parser.error("缺少训练输入配置，请指定 train.py 导出的 best 目录")
    input_config = json.loads(config_path.read_text(encoding="utf-8"))
    if input_config.get("input_format") != "query_then_recent_history_v1":
        parser.error("模型输入格式不兼容")
    output_dir = args.output_dir or (
        REPORTS_ROOT / "classifier" / datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    )
    if output_dir.exists() and any(output_dir.iterdir()):
        parser.error("output-dir 已有内容，请换一个目录，避免覆盖评测报告")

    import numpy as np
    import torch
    import transformers
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    load_start = perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(str(args.model_path), local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        str(args.model_path), local_files_only=True,
    )
    if model.config.label2id != LABEL2ID or model.config.id2label != dict(enumerate(LABELS)):
        raise ValueError("模型标签映射与当前六类定义不一致")
    if not 8 <= input_config["max_length"] <= model.config.max_position_embeddings:
        raise ValueError("保存的输入长度与模型位置编码范围不兼容")
    if input_config["history_rounds"] < 0:
        raise ValueError("保存的 history_rounds 不能为负数")
    model.to(device).eval()
    if device.type == "cuda":
        torch.cuda.synchronize()
    model_load_ms = (perf_counter() - load_start) * 1000

    @torch.inference_mode()
    def predict_one(sample):
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = perf_counter()
        encoded = encode_sample(
            tokenizer, sample, input_config["max_length"], input_config["history_rounds"],
        )
        inputs = {name: torch.tensor([value], device=device) for name, value in encoded.items()}
        probabilities = model(**inputs).logits.float().softmax(dim=-1)[0].cpu().tolist()
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed_ms = (perf_counter() - start) * 1000
        if len(probabilities) != len(LABELS) or not np.isfinite(probabilities).all():
            raise ValueError("模型输出无效的类别概率")
        return probabilities, elapsed_ms, len(encoded["input_ids"])

    for step in range(args.warmup_steps):
        predict_one(samples[step % len(samples)])
    predictions = []
    for index, sample in enumerate(samples, 1):
        probabilities, elapsed_ms, token_count = predict_one(sample)
        best = max(range(len(LABELS)), key=probabilities.__getitem__)
        predicted = LABELS[best]
        predictions.append({
            "id": sample.get("id", f"line-{index}"),
            "query": sample["query"], "history": sample["history"],
            "group_id": sample.get("group_id"), "tags": sample.get("tags", []),
            "expected_intent": sample["intent"], "predicted_intent": predicted,
            "correct": sample["intent"] == predicted,
            "confidence": probabilities[best], "probabilities": dict(zip(LABELS, probabilities)),
            "input_tokens": token_count, "latency_ms": elapsed_ms,
        })
    report = compute_report(predictions)
    latencies = [row["latency_ms"] for row in predictions]
    report.update({
        "model_path": str(args.model_path.resolve()), "test_file": str(args.test_file.resolve()),
        "test_sha256": hashlib.sha256(args.test_file.read_bytes()).hexdigest(),
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "precision": str(next(model.parameters()).dtype),
        "torch_version": torch.__version__, "transformers_version": transformers.__version__,
        "input_config": input_config, "data_audit": audit,
        "latency": {
            "batch_size": 1, "warmup_steps": args.warmup_steps,
            "model_load_ms": model_load_ms,
            "mean_ms": float(np.mean(latencies)),
            "p50_ms": float(np.percentile(latencies, 50)),
            "p95_ms": float(np.percentile(latencies, 95)),
            "scope": "tokenization + tensor/device transfer + forward + softmax + CPU result; excludes model loading and warmup",
        },
    })
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    for name, rows in [("predictions.jsonl", predictions),
                       ("errors.jsonl", [row for row in predictions if not row["correct"]])]:
        (output_dir / name).write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8",
        )
    print(json.dumps({
        "sample_count": report["sample_count"], "correct_count": report["correct_count"],
        "accuracy": report["accuracy"], "macro_f1": report["macro_f1"],
        "by_context": report["by_context"], "device": str(device),
        "latency": report["latency"], "report_dir": str(output_dir.resolve()),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
