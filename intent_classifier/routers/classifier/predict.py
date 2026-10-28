"""Run the trained intent model locally and print all six class probabilities."""
import argparse
import json
from pathlib import Path

from .common import LABELS, LABEL2ID, ROOT, encode_sample, normalize_history


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, default=ROOT / "models/intent-macbert/best")
    parser.add_argument("--query", required=True)
    parser.add_argument("--history-file", type=Path, help="JSON 历史消息列表，格式与训练数据一致")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    if not args.query.strip():
        parser.error("query 不能为空")
    config_path = args.model_path / "intent_classifier_config.json"
    if not config_path.is_file():
        parser.error("缺少训练输入配置，请使用 train.py 导出的 best 模型目录")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("input_format") != "query_then_recent_history_v1":
        parser.error("模型输入格式不兼容")
    history = normalize_history(json.loads(args.history_file.read_text(encoding="utf-8-sig"))) \
        if args.history_file else []

    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(args.model_path), local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        str(args.model_path), local_files_only=True,
    )
    if model.config.label2id != LABEL2ID:
        raise ValueError("模型标签映射与当前六类定义不一致")
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model.to(device).eval()
    encoded = encode_sample(
        tokenizer, {"query": args.query.strip(), "history": history},
        config["max_length"], config["history_rounds"],
    )
    inputs = {name: torch.tensor([value], device=device) for name, value in encoded.items()}
    with torch.inference_mode():
        probabilities = model(**inputs).logits.float().softmax(dim=-1)[0].cpu().tolist()
    best = max(range(len(LABELS)), key=probabilities.__getitem__)
    print(json.dumps({
        "intent": LABELS[best], "confidence": probabilities[best],
        "probabilities": dict(zip(LABELS, probabilities)),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
