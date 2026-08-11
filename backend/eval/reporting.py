"""Shared helpers for writing evaluation aggregate reports."""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from typing import Any


def build_aggregate_report(
    *,
    report_type: str,
    experiment_prefix: str,
    run_mode: str,
    rows: list[dict[str, Any]],
    generated_at: datetime,
) -> dict[str, Any]:
    """Build a serializable metric summary from per-sample evaluation rows."""
    metric_values: dict[str, list[float]] = defaultdict(list)
    failed_sample_count = 0

    for row in rows:
        outputs = row.get("outputs")
        has_error = bool(row.get("error")) or (
            isinstance(outputs, dict) and bool(outputs.get("error"))
        )
        if has_error:
            failed_sample_count += 1

        scores = row.get("scores")
        if not isinstance(scores, dict):
            continue

        for metric_name, metric_result in scores.items():
            if not isinstance(metric_result, dict):
                continue
            score = metric_result.get("score")
            if isinstance(score, (int, float)) and not isinstance(score, bool):
                metric_values[metric_name].append(float(score))

    metrics = {
        metric_name: {
            "mean": sum(values) / len(values),
            "count": len(values),
        }
        for metric_name, values in sorted(metric_values.items())
    }
    sample_count = len(rows)
    return {
        "report_type": report_type,
        "experiment_prefix": experiment_prefix,
        "run_mode": run_mode,
        "generated_at": generated_at.isoformat(timespec="seconds"),
        "sample_count": sample_count,
        "successful_sample_count": sample_count - failed_sample_count,
        "failed_sample_count": failed_sample_count,
        "metrics": metrics,
    }
