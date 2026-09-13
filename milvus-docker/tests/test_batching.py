import asyncio
import threading
import unittest

import embedding_service as service


class Vector:
    def tolist(self):
        return [1.0]


class FakeModel:
    def __init__(self, fail=False, mismatch=False, block=False):
        self.fail = fail
        self.mismatch = mismatch
        self.block = block
        self.started = threading.Event()
        self.release = threading.Event()
        self.sizes = []

    def encode(self, queries, **kwargs):
        self.sizes.append(len(queries))
        self.started.set()
        if self.block and not self.release.wait(3):
            raise RuntimeError("test synchronization timeout")
        if self.fail:
            raise RuntimeError("simulated GPU failure")
        count = 0 if self.mismatch else len(queries)
        return {"dense_vecs": [Vector() for _ in range(count)], "lexical_weights": [{} for _ in range(count)]}

    def compute_score(self, pairs, **kwargs):
        self.encode(pairs)
        return [0.5] * len(pairs)


class BatchingTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        service.GPU_SEMAPHORE = asyncio.Semaphore(1)

    async def test_encode_failure_and_mismatch_finish_all_waiters(self):
        for model in (FakeModel(fail=True), FakeModel(mismatch=True)):
            encoder = service.BatchEncoder(model, max_wait_ms=1)
            results = await asyncio.wait_for(asyncio.gather(
                encoder.encode("a"), encoder.encode("b"), return_exceptions=True,
            ), 2)
            self.assertTrue(all(isinstance(result, RuntimeError) for result in results))
            model.fail = model.mismatch = False
            self.assertEqual((await asyncio.wait_for(encoder.encode("recovered"), 2))["dense"], [1.0])

    async def test_encode_burst_respects_batch_limit(self):
        model = FakeModel()
        encoder = service.BatchEncoder(model, max_wait_ms=10, max_batch=2)
        results = await asyncio.wait_for(asyncio.gather(*(encoder.encode(str(i)) for i in range(7))), 2)
        self.assertEqual(len(results), 7)
        self.assertLessEqual(max(model.sizes), 2)

    async def test_cancellation_during_inference_does_not_strand_other_requests(self):
        for kind in ("encoder", "reranker"):
            with self.subTest(kind=kind):
                model = FakeModel(block=True)
                batcher = (service.BatchEncoder if kind == "encoder" else service.BatchReranker)(model, max_wait_ms=10)
                invoke = batcher.encode if kind == "encoder" else batcher.compute
                value = "query" if kind == "encoder" else [["q", "text"]]
                cancelled = asyncio.create_task(invoke(value))
                survivor = asyncio.create_task(invoke(value))
                try:
                    self.assertTrue(await asyncio.to_thread(model.started.wait, 2))
                    cancelled.cancel()
                    with self.assertRaises(asyncio.CancelledError):
                        await cancelled
                finally:
                    model.release.set()
                result = await asyncio.wait_for(survivor, 2)
                self.assertIsNotNone(result)

    async def test_reranker_failure_finishes_all_waiters(self):
        batcher = service.BatchReranker(FakeModel(fail=True), max_wait_ms=1)
        results = await asyncio.wait_for(asyncio.gather(
            batcher.compute([["a", "b"]]), batcher.compute([["c", "d"]]), return_exceptions=True,
        ), 2)
        self.assertTrue(all(isinstance(result, RuntimeError) for result in results))
