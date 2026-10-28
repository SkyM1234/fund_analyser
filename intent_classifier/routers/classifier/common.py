"""Shared labels, data checks, splitting and tokenization for intent classification."""
import hashlib
import json
from collections import Counter
from pathlib import Path
from intent_classifier.common import LABELS, LABEL2ID, ROOT


def normalize_history(history):
    if not isinstance(history, list):
        raise ValueError("history 必须是列表")
    roles = {"human": "user", "user": "user", "ai": "assistant", "assistant": "assistant"}
    result = []
    for message in history:
        if isinstance(message, dict):
            role, content = message.get("role"), message.get("content")
        elif isinstance(message, (list, tuple)) and len(message) == 2:
            role, content = message
        else:
            raise ValueError("历史消息必须是 [role, content] 或包含 role/content 的对象")
        if not isinstance(role, str) or role not in roles or not isinstance(content, str):
            raise ValueError("历史角色只支持 human/ai/user/assistant，content 必须是字符串")
        result.append([roles[role], content.strip()])
    return result


def fingerprint(sample):
    text = json.dumps([sample["query"], sample["history"]], ensure_ascii=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def read_jsonl(path):
    samples, seen = [], set()
    with Path(path).open(encoding="utf-8-sig") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                sample = json.loads(line)
                if not isinstance(sample, dict):
                    raise ValueError("每行必须是 JSON 对象")
                if not isinstance(sample.get("query"), str) or not sample["query"].strip():
                    raise ValueError("query 必须是非空字符串")
                if not isinstance(sample.get("intent"), str) or sample["intent"] not in LABEL2ID:
                    raise ValueError(f"intent 必须属于 {LABELS}")
                sample["query"] = sample["query"].strip()
                sample["history"] = normalize_history(sample.get("history", []))
                if "group_id" in sample and (
                    not isinstance(sample["group_id"], str) or not sample["group_id"].strip()
                ):
                    raise ValueError("group_id 必须是非空字符串")
                key = fingerprint(sample)
                if key in seen:
                    raise ValueError("存在重复或标签冲突的 query/history，请先去重")
                seen.add(key)
                samples.append(sample)
            except (ValueError, TypeError) as exc:
                raise ValueError(f"{path}:{line_number}: {exc}") from exc
    if not samples:
        raise ValueError(f"数据文件为空：{path}")
    return samples


def check_split(train, validation):
    for name, samples in [("训练集", train), ("验证集", validation)]:
        missing = set(LABELS) - {sample["intent"] for sample in samples}
        if missing:
            raise ValueError(f"{name}缺少类别 {sorted(missing)}，请补充数据或调整划分")
    if {fingerprint(row) for row in train} & {fingerprint(row) for row in validation}:
        raise ValueError("训练集和验证集存在相同 query/history")
    train_groups = {row["group_id"] for row in train if "group_id" in row}
    val_groups = {row["group_id"] for row in validation if "group_id" in row}
    if train_groups & val_groups:
        raise ValueError("训练集和验证集存在相同 group_id")


def split_samples(samples, validation_ratio, seed):
    from sklearn.model_selection import GroupShuffleSplit, train_test_split

    if not 0 < validation_ratio < 1:
        raise ValueError("validation-ratio 必须在 0 和 1 之间")
    if not any("group_id" in row for row in samples):
        try:
            train, validation = train_test_split(
                samples, test_size=validation_ratio, random_state=seed,
                stratify=[row["intent"] for row in samples],
            )
        except ValueError as exc:
            raise ValueError("无法分层划分，请增加每类样本或提供独立验证集") from exc
        check_split(train, validation)
        return train, validation

    # A group may contain multiple intents (for example, a complete conversation).
    groups = ["group:" + row["group_id"] if "group_id" in row
              else "sample:" + fingerprint(row) for row in samples]
    splitter = GroupShuffleSplit(n_splits=128, test_size=validation_ratio, random_state=seed)
    best = None
    distribution = Counter(row["intent"] for row in samples)
    for train_indices, val_indices in splitter.split(samples, groups=groups):
        train = [samples[i] for i in train_indices]
        validation = [samples[i] for i in val_indices]
        if {row["intent"] for row in train} != set(LABELS) or {
            row["intent"] for row in validation
        } != set(LABELS):
            continue
        val_counts = Counter(row["intent"] for row in validation)
        score = abs(len(validation) / len(samples) - validation_ratio) + sum(
            abs(val_counts[label] / len(validation) - distribution[label] / len(samples))
            for label in LABELS
        )
        if best is None or score < best[0]:
            best = (score, train, validation)
    if best is None:
        raise ValueError("无法在隔离 group_id 的同时覆盖六类，请增加独立分组或指定验证集")
    check_split(best[1], best[2])
    return best[1], best[2]


def encode_sample(tokenizer, sample, max_length=256, history_rounds=3):
    """Keep the current query first and spend the remaining budget on recent history."""
    budget = max_length - tokenizer.num_special_tokens_to_add(pair=True)
    query_ids = tokenizer.encode(sample["query"], add_special_tokens=False)
    if len(query_ids) > budget:
        raise ValueError("当前问题超过 token 预算，请增加 max-length 或预先处理超长问题")
    recent = sample["history"][-2 * history_rounds:] if history_rounds else []
    history_text = "\n".join(
        ("用户：" if role == "user" else "助手：") + content for role, content in recent
    )
    history_ids = tokenizer.encode(history_text, add_special_tokens=False)
    remaining = budget - len(query_ids)
    history_ids = history_ids[-remaining:] if remaining else []
    return tokenizer.prepare_for_model(
        query_ids, pair_ids=history_ids, return_attention_mask=True,
        return_token_type_ids=True, truncation=False,
    )
