"""Run the live routing dataset inside Docker: python -m scripts.verify_routing."""
import asyncio
import json
from pathlib import Path
from time import perf_counter

from langchain_core.messages import AIMessage, HumanMessage

from app.agent.multi_agent_controller import route_after_intent
from app.services.router import route_query


async def main():
    dataset = Path(__file__).resolve().parents[1] / "eval/datasets/routing.jsonl"
    cases = [json.loads(line) for line in dataset.read_text(encoding="utf-8").splitlines() if line.strip()]
    semaphore = asyncio.Semaphore(3)

    async def check(case):
        async with semaphore:
            history = [
                (HumanMessage if role == "human" else AIMessage)(content=content)
                for role, content in case.get("history", [])
            ]
            start = perf_counter()
            try:
                result = await route_query(case["query"], history)
                errors = []
                if result.intent != case["intent"]:
                    errors.append("intent")
                if result.needs_clarification != case.get("needs_clarification", False):
                    errors.append("needs_clarification")
                actual_basis = {(basis.kind, basis.value, basis.source) for basis in result.scope_basis}
                if "basis" in case and actual_basis != {tuple(basis) for basis in case["basis"]}:
                    errors.append("scope_basis")
                for fragment in case.get("resolved_contains", []):
                    if fragment not in result.resolved_query:
                        errors.append("missing:" + fragment)
                for fragment in case.get("resolved_excludes", []):
                    if fragment in result.resolved_query:
                        errors.append("unexpected:" + fragment)
                details = {**result.model_dump(), "branch": route_after_intent({"route_result": result})}
            except Exception as exc:
                errors = [type(exc).__name__]
                details = {}
            print(json.dumps({
                "id": case["id"], "passed": not errors, "errors": errors,
                "seconds": round(perf_counter() - start, 2), **details,
            }, ensure_ascii=False), flush=True)
            return not errors

    results = await asyncio.gather(*(check(case) for case in cases))
    print(f"Live routing: {sum(results)}/{len(results)} passed", flush=True)
    if not all(results):
        raise SystemExit(1)


if __name__ == "__main__":
    asyncio.run(main())
