"""Compose router decisions in a fixed order, stopping at the first acceptance."""
from time import perf_counter

STAGES = ("rules", "vector", "classifier")


class CascadeRouter:
    def __init__(self, routers):
        self.routers = {router.name: router for router in routers}
        if len(self.routers) != len(routers) or set(self.routers) - set(STAGES):
            raise ValueError("路由名称重复或未知")

    def route(self, query):
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query 不能为空")
        query = query.strip()
        trace = []
        start = perf_counter()
        for stage in STAGES:
            router = self.routers.get(stage)
            if router is None:
                trace.append({"stage": stage, "status": "disabled"})
                continue
            stage_start = perf_counter()
            decision = router.route(query)
            trace.append({
                "stage": stage,
                "status": "accepted" if decision["accepted"] else "rejected",
                **decision, "latency_ms": (perf_counter() - stage_start) * 1000,
            })
            if decision["accepted"]:
                return {
                    "query": query, "intent": decision["candidate_intent"], "stage": stage,
                    "needs_clarification": False, "clarification_question": None,
                    "trace": trace, "latency_ms": (perf_counter() - start) * 1000,
                }
        return {
            "query": query, "intent": None, "stage": "clarification",
            "needs_clarification": True,
            "clarification_question": "请补充你的问题：是查询具体基金、按条件筛选基金，还是了解金融知识？",
            "trace": trace, "latency_ms": (perf_counter() - start) * 1000,
        }
