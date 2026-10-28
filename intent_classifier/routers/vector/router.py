"""BGE embeddings and per-intent maximum cosine similarity."""
from intent_classifier.common import LABELS, read_queries, select_candidate


def aggregate_similarities(similarities, references):
    scores = {label: -1.0 for label in LABELS}
    for similarity, row in zip(similarities, references):
        scores[row["intent"]] = max(scores[row["intent"]], float(similarity))
    return scores


class VectorRouter:
    name = "vector"

    def __init__(self, model_path, examples_file, device, threshold, margin):
        if not 0 <= threshold <= 1 or not 0 <= margin <= 1:
            raise ValueError("阈值和分差必须在 0 到 1 之间")
        self.threshold, self.margin = threshold, margin
        self.references = read_queries(examples_file)
        if {row["intent"] for row in self.references} != set(LABELS):
            raise ValueError("向量路由例句必须覆盖六类")
        if not model_path.is_dir():
            raise ValueError(f"本地向量模型不存在：{model_path}")
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(
            str(model_path.resolve()), device=device, local_files_only=True, trust_remote_code=False,
        )
        texts = [row["query"] for row in self.references]
        for text in texts:
            self._check_length(text)
        self.reference_vectors = self.model.encode(
            texts, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False,
        )

    def _check_length(self, query):
        if len(self.model.tokenizer(query, truncation=False)["input_ids"]) > self.model.max_seq_length:
            raise ValueError("当前问题超过向量模型 token 上限")

    def route(self, query):
        self._check_length(query)
        vector = self.model.encode(
            [query], normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False,
        )[0]
        scores = aggregate_similarities(self.reference_vectors @ vector, self.references)
        return {**select_candidate(scores, self.threshold, self.margin),
                "score_type": "cosine_similarity"}
