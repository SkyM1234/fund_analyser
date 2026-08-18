import unittest
from unittest.mock import patch

from langchain_core.messages import ToolMessage

from app.agent.retrieval_agent import (
    _RagToolContext,
    _unique_rag_chunks_for_observability,
)


class RetrievalContextTests(unittest.TestCase):
    def test_observability_dedup_does_not_mutate_tool_messages(self) -> None:
        first = ToolMessage(
            tool_call_id="call-1",
            name="rag_search",
            content="query one result",
        )
        second = ToolMessage(
            tool_call_id="call-2",
            name="rag_search",
            content="query two result",
        )
        contexts = [
            _RagToolContext(message=first, output="query one result"),
            _RagToolContext(message=second, output="query two result"),
        ]

        with patch(
            "app.agent.retrieval_agent.parse_rag_search_result",
            side_effect=[
                [{"id": "chunk-1", "score": 0.8}],
                [{"id": "chunk-1", "score": 0.9}],
            ],
        ):
            chunks = _unique_rag_chunks_for_observability(contexts)

        self.assertEqual(first.content, "query one result")
        self.assertEqual(second.content, "query two result")
        self.assertEqual(chunks, [{"id": "chunk-1", "score": 0.9}])

    def test_observability_keeps_distinct_chunks_from_multiple_queries(self) -> None:
        contexts = [
            _RagToolContext(
                message=ToolMessage(
                    tool_call_id="call-1",
                    name="rag_search",
                    content="query one result",
                ),
                output="query one result",
            ),
            _RagToolContext(
                message=ToolMessage(
                    tool_call_id="call-2",
                    name="rag_search",
                    content="query two result",
                ),
                output="query two result",
            ),
        ]

        with patch(
            "app.agent.retrieval_agent.parse_rag_search_result",
            side_effect=[
                [{"id": "chunk-1", "score": 0.8}],
                [{"id": "chunk-2", "score": 0.7}],
            ],
        ):
            chunks = _unique_rag_chunks_for_observability(contexts)

        self.assertEqual([chunk["id"] for chunk in chunks], ["chunk-1", "chunk-2"])

