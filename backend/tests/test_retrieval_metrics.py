import math
import unittest
from types import SimpleNamespace

from eval.evaluators import retrieval_metrics as metrics


def _example(outputs: dict) -> SimpleNamespace:
    return SimpleNamespace(outputs=outputs)


def _run(contexts: list[dict]) -> SimpleNamespace:
    return SimpleNamespace(
        outputs={
            "intent": "cross_fund_query",
            "retrieved_contexts": contexts,
            "retrieved_chunks": [
                chunk for context in contexts for chunk in context["chunks"]
            ],
        }
    )


class SessionRetrievalMetricTests(unittest.TestCase):
    def setUp(self) -> None:
        self.example = _example(
            {
                "relevant_chunk_ids": [
                    "good-1",
                    "good-2",
                    "good-3",
                    "good-4",
                    "good-5",
                ],
            }
        )

    def test_session_metrics_use_flat_ground_truth_chunk_ids(self) -> None:
        run = _run(
            [
                {
                    "task_id": "t1",
                    "chunks": [
                        {"id": "noise-1"},
                        {"id": "good-1"},
                        {"id": "good-2"},
                    ],
                },
                {
                    "task_id": "t2",
                    "chunks": [{"id": "noise-2"}],
                },
            ]
        )

        self.assertEqual(metrics.session_hit_rate(run, self.example)["score"], 0.4)
        self.assertAlmostEqual(
            metrics.session_mrr(run, self.example)["score"],
            (1 / 2 + 1 / 3) / 5,
        )
        self.assertAlmostEqual(
            metrics.session_ndcg(run, self.example)["score"],
            (1 / math.log2(3) + 1 / 2)
            / sum(1 / math.log2(rank + 1) for rank in range(1, 6)),
        )

    def test_duplicate_returned_chunk_does_not_change_rank_or_score(self) -> None:
        run = SimpleNamespace(
            outputs={
                "results": [
                    {"id": "noise"},
                    {"id": "good-1"},
                    {"id": "good-1"},
                    {"id": "good-2"},
                ]
            }
        )
        example = _example({"relevant_chunk_ids": ["good-1", "good-2"]})

        self.assertEqual(metrics.session_hit_rate(run, example)["score"], 1.0)
        self.assertAlmostEqual(metrics.session_mrr(run, example)["score"], (1 / 2 + 1 / 3) / 2)
        self.assertAlmostEqual(
            metrics.session_ndcg(run, example)["score"],
            (1 / math.log2(3) + 1 / 2) / (1 + 1 / math.log2(3)),
        )

    def test_ndcg_uses_all_ground_truth_chunks_for_idcg(self) -> None:
        run = SimpleNamespace(outputs={"results": [{"id": "first"}]})
        example = _example({"relevant_chunk_ids": ["first", "second"]})

        self.assertAlmostEqual(
            metrics.ndcg(run, example)["score"],
            1 / (1 + 1 / math.log2(3)),
        )


if __name__ == "__main__":
    unittest.main()
