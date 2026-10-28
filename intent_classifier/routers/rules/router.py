"""Full-query rules, independent of either model router."""
import json
import re

from intent_classifier.common import LABELS


class RulesRouter:
    name = "rules"

    def __init__(self, rules_file):
        self.rules = []
        for rule in json.loads(rules_file.read_text(encoding="utf-8")):
            if rule["intent"] not in LABELS:
                raise ValueError(f"规则标签无效：{rule['intent']}")
            self.rules.append((rule["id"], rule["intent"], re.compile(rule["pattern"], re.IGNORECASE)))

    def route(self, query):
        for rule_id, intent, pattern in self.rules:
            if pattern.fullmatch(query):
                return {
                    "accepted": True, "candidate_intent": intent, "rule": rule_id,
                    "score_type": "rule_match",
                    "scores": {label: int(label == intent) for label in LABELS},
                }
        return {"accepted": False, "rule": None, "score_type": "rule_match",
                "scores": {label: 0 for label in LABELS}}
