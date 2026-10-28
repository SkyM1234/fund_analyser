"""Unified entry point: choose offline routers, run the cascade and save results."""
import argparse
import hashlib
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from time import perf_counter

# Support both `python -m intent_classifier.cascade` and the direct script command.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from intent_classifier.common import ROOT, REPORTS_ROOT, read_queries
from intent_classifier.pipeline import CascadeRouter, STAGES
from intent_classifier.validation.reporting import summarize, write_reports

DEFAULT_THRESHOLDS = {
    "vector_threshold": 0.85, "vector_margin": 0.06,
    "classifier_threshold": 0.75, "classifier_margin": 0.15,
}


def build_pipeline(args, enabled):
    if set(enabled) - set(STAGES):
        raise ValueError("未知路由名称")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    device = "none"
    if {"vector", "classifier"} & set(enabled):
        import torch
        torch.set_num_threads(args.cpu_threads)
        device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    routers = []
    if "rules" in enabled:
        from intent_classifier.routers.rules.router import RulesRouter
        routers.append(RulesRouter(args.rules_file))
    if "vector" in enabled:
        from intent_classifier.routers.vector.router import VectorRouter
        routers.append(VectorRouter(args.embedding_model, args.routes_file, device,
                                    args.vector_threshold, args.vector_margin))
    if "classifier" in enabled:
        from intent_classifier.routers.classifier.router import ClassifierRouter
        routers.append(ClassifierRouter(args.classifier_model, device,
                                        args.classifier_threshold, args.classifier_margin))
    return CascadeRouter(routers), device


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--query")
    source.add_argument("--input-file", type=Path)
    parser.add_argument("--routes", nargs="+", choices=[*STAGES, "none"], default=list(STAGES),
                        help="启用的层；始终按 rules → vector → classifier 执行。none 表示全部关闭")
    parser.add_argument("--embedding-model", type=Path, default=ROOT / "embedding_model/bge-base-zh-v1.5")
    parser.add_argument("--classifier-model", type=Path, default=ROOT / "models/intent-macbert-200-v1/best")
    parser.add_argument("--routes-file", type=Path, default=ROOT / "intent_classifier/routers/vector/data/examples.jsonl")
    parser.add_argument("--rules-file", type=Path, default=ROOT / "intent_classifier/routers/rules/data/rules.json")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--warmup-steps", type=int, default=3)
    parser.add_argument("--cpu-threads", type=int, default=2)
    parser.add_argument("--cpu", action="store_true")
    for name, value in DEFAULT_THRESHOLDS.items():
        parser.add_argument("--" + name.replace("_", "-"), type=float, default=value)
    args = parser.parse_args()
    if "none" in args.routes and len(args.routes) != 1:
        parser.error("none 不能与其他路由一起启用")
    if args.warmup_steps < 0 or args.cpu_threads < 1:
        parser.error("warmup-steps 必须非负，cpu-threads 必须为正")
    thresholds = {name: getattr(args, name) for name in DEFAULT_THRESHOLDS}
    if any(not 0 <= value <= 1 for value in thresholds.values()):
        parser.error("阈值和分差必须在 0 到 1 之间")
    enabled = [stage for stage in STAGES if stage in args.routes]
    if args.query is not None:
        if not args.query.strip():
            parser.error("query 不能为空")
        samples = [{"id": "query", "query": args.query.strip()}]
    else:
        args.input_file = args.input_file or ROOT / "intent_classifier/validation/queries.jsonl"
        samples = read_queries(args.input_file)
    output_dir = args.output_dir or REPORTS_ROOT / "cascade" / datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    if output_dir.exists() and any(output_dir.iterdir()):
        parser.error("output-dir 已有内容，请换一个目录")
    start = perf_counter()
    pipeline, device = build_pipeline(args, enabled)
    initialization_ms = (perf_counter() - start) * 1000
    for _ in range(args.warmup_steps):
        for router in pipeline.routers.values():
            router.route("查询159103的费率")
    predictions = []
    for index, sample in enumerate(samples, 1):
        result = pipeline.route(sample["query"])
        result["id"] = sample.get("id", f"line-{index}")
        if "intent" in sample:
            result["expected_intent"] = sample["intent"]
        predictions.append(result)
    summary = summarize(predictions)
    references = pipeline.routers["vector"].references if "vector" in enabled else []
    overlap = {row["query"] for row in references} & {row["query"] for row in samples}
    report = {
        **summary, "enabled_routes": enabled, "thresholds": thresholds,
        "thresholds_calibrated": False, "input_mode": "current_query_only", "offline": True,
        "device": device, "initialization_ms": initialization_ms,
        "warmup_steps": args.warmup_steps,
        "latency_scope": "resident models; includes reached stages; excludes initialization and warmup",
        "embedding_model": str(args.embedding_model.resolve()) if "vector" in enabled else None,
        "classifier_model": str(args.classifier_model.resolve()) if "classifier" in enabled else None,
        "reference_count": len(references), "reference_query_overlap_count": len(overlap),
        "routes_file": str(args.routes_file.resolve()) if "vector" in enabled else None,
        "rules_file": str(args.rules_file.resolve()) if "rules" in enabled else None,
        "input_file": str(args.input_file.resolve()) if args.input_file else None,
        "input_sha256": hashlib.sha256(args.input_file.read_bytes()).hexdigest() if args.input_file else None,
    }
    write_reports(output_dir, report, predictions)
    print(json.dumps({"report_dir": str(output_dir.resolve()), **report}, ensure_ascii=False, indent=2))
    if args.query is not None:
        print(json.dumps(predictions[0], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
