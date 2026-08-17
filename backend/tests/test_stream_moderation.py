import unittest

from app.services.stream_moderation import StreamModerationBuffer


class StreamModerationTests(unittest.TestCase):
    def test_holds_window_until_safe_to_release(self) -> None:
        buffer = StreamModerationBuffer(window_chars=6, overlap_chars=2)

        released, hit = buffer.feed("这是一段普通")
        self.assertEqual(released, "")
        self.assertIsNone(hit)

        released, hit = buffer.feed("内容")
        self.assertEqual(released, "这是一段普通")
        self.assertIsNone(hit)

    def test_detects_phrase_split_across_chunks(self) -> None:
        buffer = StreamModerationBuffer(window_chars=20, overlap_chars=10)

        released, hit = buffer.feed("请考虑")
        self.assertEqual(released, "")
        self.assertIsNone(hit)

        released, hit = buffer.feed("建议买入ETF")
        self.assertEqual(released, "")
        self.assertIsNotNone(hit)
        self.assertIn("投资建议", hit.reason)

    def test_flushes_safe_tail(self) -> None:
        buffer = StreamModerationBuffer()

        released, hit = buffer.feed("这是普通的基金科普内容")
        self.assertEqual(released, "")
        self.assertIsNone(hit)

        released, hit = buffer.flush()
        self.assertEqual(released, "这是普通的基金科普内容")
        self.assertIsNone(hit)
