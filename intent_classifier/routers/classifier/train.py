"""Fine-tune a local Chinese BERT model for the project's six intent labels."""
import argparse
import json
from collections import Counter
from pathlib import Path

from .common import LABELS, LABEL2ID, ROOT, check_split, encode_sample, read_jsonl, split_samples


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-file", type=Path, required=True)
    parser.add_argument("--validation-file", type=Path)
    parser.add_argument("--model-path", type=Path, default=ROOT / "models/chinese-macbert-base")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "models/intent-macbert")
    parser.add_argument("--validation-ratio", type=float, default=0.2)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--history-rounds", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--epochs", type=float, default=5)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=-1, help="调试时限制优化步数")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--fp16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    if args.max_length < 8 or args.history_rounds < 0:
        parser.error("max-length 至少为 8，history-rounds 不能为负数")
    if args.batch_size < 1 or args.gradient_accumulation_steps < 1 or args.patience < 1:
        parser.error("batch-size、gradient-accumulation-steps、patience 必须为正整数")
    if args.epochs <= 0 or args.learning_rate <= 0 or args.max_steps == 0 or args.max_steps < -1:
        parser.error("epochs、learning-rate 必须大于零，max-steps 为 -1 或正整数")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error("output-dir 已有内容，请指定新的目录，避免覆盖已有训练结果")
    return args


def main():
    args = parse_args()
    train = read_jsonl(args.train_file)
    if args.validation_file:
        validation = read_jsonl(args.validation_file)
        check_split(train, validation)
    else:
        train, validation = split_samples(train, args.validation_ratio, args.seed)

    import numpy as np
    import torch
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
    from transformers import (
        AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding,
        EarlyStoppingCallback, Trainer, TrainingArguments, set_seed,
    )

    set_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(str(args.model_path), local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        str(args.model_path), num_labels=len(LABELS),
        id2label=dict(enumerate(LABELS)), label2id=LABEL2ID,
        local_files_only=True,
    )
    if args.max_length > model.config.max_position_embeddings:
        raise ValueError(f"max-length 不能超过模型上限 {model.config.max_position_embeddings}")

    class IntentDataset(torch.utils.data.Dataset):
        def __init__(self, samples):
            self.rows = []
            for index, sample in enumerate(samples):
                try:
                    encoded = encode_sample(tokenizer, sample, args.max_length, args.history_rounds)
                except ValueError as exc:
                    raise ValueError(f"样本 {sample.get('id', index)}: {exc}") from exc
                encoded["labels"] = LABEL2ID[sample["intent"]]
                self.rows.append(encoded)

        def __len__(self):
            return len(self.rows)

        def __getitem__(self, index):
            return self.rows[index]

    train_dataset, val_dataset = IntentDataset(train), IntentDataset(validation)
    use_cuda = torch.cuda.is_available() and not args.cpu
    use_fp16 = use_cuda and args.fp16
    print(json.dumps({
        "device": "cuda" if use_cuda else "cpu", "fp16": use_fp16,
        "train_counts": dict(Counter(row["intent"] for row in train)),
        "validation_counts": dict(Counter(row["intent"] for row in validation)),
        "effective_batch_size": args.batch_size * args.gradient_accumulation_steps,
    }, ensure_ascii=False), flush=True)

    def metrics(prediction):
        predictions = np.argmax(prediction.predictions, axis=-1)
        return {
            "accuracy": accuracy_score(prediction.label_ids, predictions),
            "macro_f1": f1_score(
                prediction.label_ids, predictions, labels=list(range(len(LABELS))),
                average="macro", zero_division=0,
            ),
        }

    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        num_train_epochs=args.epochs, max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate, weight_decay=0.01, warmup_ratio=0.1,
        fp16=use_fp16, use_cpu=not use_cuda,
        gradient_checkpointing=args.gradient_checkpointing,
        eval_strategy="epoch", save_strategy="epoch", save_total_limit=1,
        load_best_model_at_end=True, metric_for_best_model="macro_f1",
        greater_is_better=True, save_safetensors=True,
        logging_steps=10, logging_first_step=True,
        dataloader_num_workers=0, dataloader_pin_memory=use_cuda,
        report_to="none", seed=args.seed, data_seed=args.seed,
    )
    trainer = Trainer(
        model=model, args=training_args,
        train_dataset=train_dataset, eval_dataset=val_dataset,
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8),
        compute_metrics=metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.patience)],
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, samples in [("train_split.jsonl", train), ("validation_split.jsonl", validation)]:
        with (args.output_dir / name).open("w", encoding="utf-8") as target:
            for sample in samples:
                target.write(json.dumps(sample, ensure_ascii=False) + "\n")
    trainer.train()
    best_dir = args.output_dir / "best"
    trainer.save_model(str(best_dir))
    tokenizer.save_pretrained(str(best_dir))
    input_config = {"max_length": args.max_length, "history_rounds": args.history_rounds,
                    "input_format": "query_then_recent_history_v1"}
    (best_dir / "intent_classifier_config.json").write_text(
        json.dumps(input_config, indent=2), encoding="utf-8",
    )
    prediction = trainer.predict(val_dataset)
    predicted_ids = np.argmax(prediction.predictions, axis=-1)
    report = {
        "metrics": prediction.metrics,
        "labels": LABELS,
        "classification_report": classification_report(
            prediction.label_ids, predicted_ids, labels=list(range(len(LABELS))),
            target_names=LABELS, output_dict=True, zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(
            prediction.label_ids, predicted_ids, labels=list(range(len(LABELS))),
        ).tolist(),
        "best_checkpoint": trainer.state.best_model_checkpoint,
        "arguments": {key: str(value) if isinstance(value, Path) else value
                      for key, value in vars(args).items()},
    }
    (args.output_dir / "validation_metrics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    print(f"Best model saved to: {best_dir}", flush=True)
    print(json.dumps(prediction.metrics, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
