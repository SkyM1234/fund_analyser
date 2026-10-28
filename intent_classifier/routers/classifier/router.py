"""Query-only inference with the exported local six-class classifier."""
import json

from intent_classifier.common import LABELS, LABEL2ID, select_candidate
from .common import encode_sample


class ClassifierRouter:
    name = "classifier"

    def __init__(self, model_path, device, threshold, margin):
        if not 0 <= threshold <= 1 or not 0 <= margin <= 1:
            raise ValueError("阈值和分差必须在 0 到 1 之间")
        self.threshold, self.margin, self.device = threshold, margin, device
        self.input_config = json.loads((model_path / "intent_classifier_config.json").read_text(encoding="utf-8"))
        if self.input_config.get("input_format") != "query_then_recent_history_v1":
            raise ValueError("分类模型输入格式不兼容")
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_path.resolve()), local_files_only=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            str(model_path.resolve()), local_files_only=True,
        ).to(device).eval()
        if self.model.config.label2id != LABEL2ID or self.model.config.id2label != dict(enumerate(LABELS)):
            raise ValueError("分类模型标签映射不一致")

    def route(self, query):
        inputs = encode_sample(self.tokenizer, {"query": query, "history": []},
                               self.input_config["max_length"], history_rounds=0)
        tensors = {name: self.torch.tensor([value], device=self.device) for name, value in inputs.items()}
        with self.torch.inference_mode():
            probabilities = self.model(**tensors).logits.float().softmax(dim=-1)[0].cpu().tolist()
        return {**select_candidate(dict(zip(LABELS, probabilities)), self.threshold, self.margin),
                "score_type": "softmax_probability"}
