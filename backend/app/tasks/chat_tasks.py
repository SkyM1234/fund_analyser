"""Celery 任务：一次完整的 chat 轮次（等价于原 chat.py 的 _stream_agent）。

与原 FastAPI 内联实现的行为差异（均为迁移到任务队列后的必然结果，已与用户
确认可接受）：
- 不再检测浏览器端是否断开连接（Celery 任务不持有 HTTP 连接），任务会跑到
  完成/超时为止；FastAPI 侧只是不再转发事件，不影响 worker 的执行与
  checkpoint 落盘。
- 超时通过 Celery 的 soft_time_limit（见 celery_app.py）强制执行，取代了
  之前声明但未生效的 AGENT_TIMEOUT。
"""
import asyncio
import logging
import random
from collections.abc import Iterable

from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage

from app.core.celery_app import celery_app
from app.core.config import get_settings
from app.core.worker_lifecycle import run_coro
from app.db.mysql import get_session_factory
from app.models.chat import ChatMessage, ChatRequest
from app.services.rag_result_parser import tool_output_to_text
from app.services.task_events import is_task_cancel_signalled, publish_event
from app.services.task_runs import (
    claim_task_run,
    is_task_cancel_requested,
    mark_task_cancelled,
    mark_task_finished,
    mark_task_retrying,
    renew_task_lease,
    set_task_checkpoint_id,
)

# 锁自动过期时间必须大于 AGENT_TIMEOUT，确保即使任务超时，finally 块也有机会
# 主动释放锁，避免 LockNotOwnedError（锁已被 Redis 自动过期删除）。

logger = logging.getLogger(__name__)


class LeaseLostError(RuntimeError):
    """Raised when this worker no longer owns the task run lease."""


class SessionBusyError(RuntimeError):
    """Raised when another worker is currently processing the session."""


class TaskCancelledError(asyncio.CancelledError):
    """Raised when a task observes a user cancellation request."""


async def _session_lock_heartbeat(
    lock,
    timeout_seconds: int,
    stop_event: asyncio.Event,
    lock_lost_event: asyncio.Event | None = None,
) -> None:
    """Keep the session lock alive while the agent is running."""
    interval = max(1, timeout_seconds // 3)
    while True:
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=interval)
            return
        except asyncio.TimeoutError:
            pass

        try:
            await lock.reacquire()
        except Exception:
            logger.exception("[chat_task] session lock heartbeat failed")
            if lock_lost_event is not None:
                lock_lost_event.set()
            return


async def _claim_task_run(
    run_id: str,
    attempt: int,
    timeout_seconds: int,
    lease_seconds: int,
):
    async with get_session_factory()() as db:
        return await claim_task_run(
            db,
            run_id=run_id,
            attempt=attempt,
            timeout_seconds=timeout_seconds,
            lease_seconds=lease_seconds,
        )


async def _renew_lease(
    run_id: str,
    lease_token: str,
    lease_seconds: int,
) -> bool:
    async with get_session_factory()() as db:
        return await renew_task_lease(
            db,
            run_id=run_id,
            lease_token=lease_token,
            lease_seconds=lease_seconds,
        )


async def _mark_retrying(
    run_id: str,
    lease_token: str,
    error_code: str,
    error_message: str,
) -> bool:
    async with get_session_factory()() as db:
        return await mark_task_retrying(
            db,
            run_id=run_id,
            lease_token=lease_token,
            error_code=error_code,
            error_message=error_message,
        )


async def _mark_finished(
    run_id: str,
    lease_token: str,
    status: str,
    error_code: str | None = None,
    error_message: str | None = None,
) -> bool:
    async with get_session_factory()() as db:
        return await mark_task_finished(
            db,
            run_id=run_id,
            lease_token=lease_token,
            status=status,
            error_code=error_code,
            error_message=error_message,
        )


async def _is_cancel_requested(run_id: str, lease_token: str) -> bool:
    async with get_session_factory()() as db:
        return await is_task_cancel_requested(
            db,
            run_id=run_id,
            lease_token=lease_token,
        )


async def _mark_cancelled(run_id: str, lease_token: str) -> bool:
    async with get_session_factory()() as db:
        return await mark_task_cancelled(
            db,
            run_id=run_id,
            lease_token=lease_token,
        )


async def _persist_checkpoint_id(
    run_id: str,
    lease_token: str,
    checkpoint_id: str,
) -> bool:
    async with get_session_factory()() as db:
        return await set_task_checkpoint_id(
            db,
            run_id=run_id,
            lease_token=lease_token,
            checkpoint_id=checkpoint_id,
        )


async def _heartbeat_loop(
    run_id: str,
    lease_token: str,
    lease_seconds: int,
    interval_seconds: int,
    stop_event: asyncio.Event,
    lease_lost_event: asyncio.Event,
) -> None:
    interval = max(1, min(interval_seconds, max(1, lease_seconds // 2)))
    while True:
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=interval)
            return
        except asyncio.TimeoutError:
            pass

        try:
            renewed = await _renew_lease(run_id, lease_token, lease_seconds)
        except Exception:
            logger.exception("[chat_task] task lease heartbeat failed: run_id=%s", run_id)
            continue

        if not renewed:
            lease_lost_event.set()
            logger.warning("[chat_task] task lease lost: run_id=%s", run_id)
            return


async def _cancel_monitor_loop(
    run_id: str,
    lease_token: str,
    stop_event: asyncio.Event,
    cancel_requested_event: asyncio.Event,
) -> None:
    """Poll durable cancellation state so blocked agent calls can be stopped."""
    while True:
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=1)
            return
        except asyncio.TimeoutError:
            pass

        try:
            if (
                await is_task_cancel_signalled(run_id)
                or await _is_cancel_requested(run_id, lease_token)
            ):
                cancel_requested_event.set()
                logger.info("[chat_task] cancellation requested: run_id=%s", run_id)
                return
        except Exception:
            logger.exception(
                "[chat_task] cancellation monitor failed: run_id=%s",
                run_id,
            )


def _failure_code(exc: Exception) -> str:
    return type(exc).__name__.upper()[:64]


def _is_retryable_failure(exc: Exception) -> bool:
    return isinstance(exc, (ConnectionError, OSError, SessionBusyError)) or type(exc).__name__ in {
        "APIConnectionError",
        "APITimeoutError",
        "ConnectError",
        "ReadTimeout",
        "RemoteProtocolError",
    }


def _resolve_task_context(
    event: dict,
    task_context_by_run_id: dict[str, dict[str, str]],
) -> dict[str, str]:
    """Find the task context inherited by a LangChain child run."""
    run_id = str(event.get("run_id", ""))
    if run_id in task_context_by_run_id:
        return task_context_by_run_id[run_id]

    parent_ids = event.get("parent_ids", [])
    for parent_id in reversed(parent_ids):
        context = task_context_by_run_id.get(str(parent_id))
        if context:
            return context
    return {}


def _extract_reasoning_text(message: object) -> str:
    """Extract provider-exposed reasoning without treating normal content as reasoning."""
    candidates: list[object] = []
    if isinstance(message, dict):
        content = message.get("content")
        candidates.extend([
            message.get("reasoning_content"),
            message.get("reasoning"),
            message.get("thinking"),
            message.get("additional_kwargs", {}).get("reasoning_content")
            if isinstance(message.get("additional_kwargs"), dict)
            else None,
            message.get("response_metadata", {}).get("reasoning_content")
            if isinstance(message.get("response_metadata"), dict)
            else None,
            content if isinstance(content, Iterable) and not isinstance(content, (str, bytes, dict)) else None,
        ])
    else:
        content = getattr(message, "content", None)
        candidates.extend([
            getattr(message, "reasoning_content", None),
            getattr(message, "reasoning", None),
            getattr(message, "thinking", None),
            getattr(message, "additional_kwargs", {}).get("reasoning_content")
            if isinstance(getattr(message, "additional_kwargs", None), dict)
            else None,
            getattr(message, "response_metadata", {}).get("reasoning_content")
            if isinstance(getattr(message, "response_metadata", None), dict)
            else None,
            content if isinstance(content, Iterable) and not isinstance(content, (str, bytes, dict)) else None,
        ])

    for candidate in candidates:
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()
        if isinstance(candidate, Iterable) and not isinstance(candidate, (str, bytes, dict)):
            parts: list[str] = []
            for block in candidate:
                if not isinstance(block, dict):
                    continue
                if block.get("type") not in {"reasoning", "thinking"}:
                    continue
                text = block.get("text") or block.get("content")
                if isinstance(text, str) and text.strip():
                    parts.append(text.strip())
            if parts:
                return "\n".join(parts)
    return ""


async def _run_chat_turn(
    run_id: str,
    req: ChatRequest,
    user_id: int,
    lease_token: str,
    checkpoint_id: str | None = None,
    lease_lost_event: asyncio.Event | None = None,
    cancel_requested_event: asyncio.Event | None = None,
) -> None:
    from app.agent.multi_agent_controller import build_multi_agent_graph
    from app.db.redis import get_redis_client
    from app.services.checkpoint import get_checkpointer
    from app.services.mcp_client import get_mcp_client

    redis_client = get_redis_client()
    settings = get_settings()
    # The lock is a short-lived Redis mutex. Healthy tasks renew it; after a
    # worker crash it should expire no later than the task lease.
    lock_timeout_seconds = max(15, settings.TASK_LEASE_SECONDS)
    lock = redis_client.lock(
        f"chat:lock:{req.session_id}",
        timeout=lock_timeout_seconds,
        blocking_timeout=0,  # 拿不到锁立刻失败，不排队等待
    )
    acquired = await lock.acquire()
    if not acquired:
        raise SessionBusyError(f"Session is busy: {req.session_id}")
        logger.warning(f"[chat_task] 会话正在处理中，拒绝并发请求: session_id={req.session_id}")
        publish_event(run_id, "error", {"message": "该会话正在处理中，请稍候再试"})
        return

    lock_stop_event = asyncio.Event()
    lock_heartbeat = asyncio.create_task(
        _session_lock_heartbeat(
            lock,
            lock_timeout_seconds,
            lock_stop_event,
            lease_lost_event,
        )
    )
    try:
        logger.info("=" * 80)
        logger.info(f"[chat_task] 新请求: run_id={run_id}, session_id={req.session_id}, user_id={user_id}")
        logger.info(f"[chat_task] 用户消息: {req.message[:100]}")

        checkpointer = await get_checkpointer()

        logger.info("[chat_task] 使用多Agent架构")
        app = build_multi_agent_graph(checkpointer)
        agent_node_names = ["synthesizer", "direct_answer"]
        worker_agent_names = {
            "rag_agent",
            "market_agent",
            "arbiter_agent",
            "analysis_agent",
        }
        scope_node_name = "fund_scope"

        config = {"configurable": {"thread_id": req.session_id, "user_id": str(user_id)}}
        if checkpoint_id is not None:
            if not checkpoint_id:
                raise RuntimeError(
                    f"Task checkpoint is unavailable for recovery: {run_id}"
                )
            config["configurable"]["checkpoint_id"] = checkpoint_id
            logger.info(
                "[chat_task] resuming from checkpoint: run_id=%s checkpoint_id=%s",
                run_id,
                checkpoint_id,
            )

        checkpoint_tuple = await checkpointer.aget_tuple(config)
        has_checkpoint = checkpoint_tuple is not None and checkpoint_tuple.checkpoint is not None

        if checkpoint_id is not None and not has_checkpoint:
            raise RuntimeError(
                f"Task checkpoint no longer exists for recovery: {run_id}"
            )

        if checkpoint_id is None and not has_checkpoint and req.history:
            logger.info(f"[chat_task] 从前端 history 初始化，共 {len(req.history)} 条消息")
            init_messages = []
            for h in req.history:
                if h.role == "user":
                    init_messages.append(HumanMessage(content=h.content))
                elif h.role == "assistant":
                    init_messages.append(AIMessage(content=h.content))

            if init_messages:
                await app.aupdate_state(
                    config=config,
                    values={"messages": init_messages},
                    as_node="__start__",
                )
                logger.info(f"[chat_task] 已初始化 {len(init_messages)} 条历史消息到 checkpoint")

        if checkpoint_id is not None:
            input_state = None
        else:
            await app.aupdate_state(
                config=config,
                values={"messages": [HumanMessage(content=req.message)]},
                as_node="__start__",
            )
            latest_config = {
                "configurable": {
                    key: value
                    for key, value in config["configurable"].items()
                    if key != "checkpoint_id"
                }
            }
            initial_checkpoint = await checkpointer.aget_tuple(latest_config)
            checkpoint_id = (
                initial_checkpoint.checkpoint.get("id")
                if initial_checkpoint and initial_checkpoint.checkpoint
                else None
            )
            if not checkpoint_id:
                raise RuntimeError(
                    f"Failed to create task checkpoint before execution: {run_id}"
                )
            if not await _persist_checkpoint_id(
                run_id,
                lease_token,
                checkpoint_id,
            ):
                raise LeaseLostError(f"Task lease lost before execution: {run_id}")
            input_state = None

        retry_events: asyncio.Queue = asyncio.Queue()

        async def _on_agent_retry(agent_name: str, task_id: str, attempt: int, reason: str):
            await retry_events.put(("tool_retry", {
                "agent_name": agent_name,
                "task_id": task_id,
                "attempt": attempt,
                "reason": reason,
            }))

        async def _on_final_rag_context(
            agent_name: str,
            task_id: str,
            chunks: list[dict],
        ) -> None:
            publish_event(run_id, "retrieval_context", {
                "agent_name": agent_name,
                "task_id": task_id,
                "chunks": chunks,
            })

        config["configurable"]["_sse_retry_callback"] = _on_agent_retry
        config["configurable"]["_sse_final_rag_context_callback"] = _on_final_rag_context

        try:
            logger.info("[chat_task] 开始流式处理...")
            event_count = 0
            seen_run_ids: set[str] = set()
            task_context_by_run_id: dict[str, dict[str, str]] = {}
            trace_events: list[dict] = []
            trace_sequence = 0
            streamed_answer = ""
            decision_by_task: dict[str, str] = {}
            decision_tools: dict[str, list[str]] = {}
            decision_trace_event: dict[str, dict] = {}
            _route_emitted = False
            _plan_emitted = False

            def emit_trace(event_name: str, data: dict) -> None:
                nonlocal trace_sequence
                trace_sequence += 1
                payload = {
                    **data,
                    "event_id": f"{run_id}:{trace_sequence}",
                    "sequence": trace_sequence,
                }
                trace_events.append({"type": event_name, **payload})
                publish_event(run_id, event_name, payload)

            async for event in app.astream_events(input_state, config=config, version="v2"):
                if lease_lost_event is not None and lease_lost_event.is_set():
                    raise LeaseLostError(f"Task lease lost: {run_id}")
                if (
                    cancel_requested_event is not None
                    and cancel_requested_event.is_set()
                ):
                    raise TaskCancelledError()

                while not retry_events.empty():
                    evt_type, evt_data = retry_events.get_nowait()
                    logger.info(f"[chat_task] event: {evt_type} -> {evt_data['agent_name']} task={evt_data['task_id']} attempt={evt_data['attempt']}")
                    emit_trace(evt_type, evt_data)

                kind = event["event"]
                event_count += 1

                if kind == "on_chain_end":
                    latest_config = {
                        "configurable": {
                            key: value
                            for key, value in config["configurable"].items()
                            if key != "checkpoint_id"
                        }
                    }
                    latest_checkpoint = await checkpointer.aget_tuple(latest_config)
                    checkpoint_id = (
                        latest_checkpoint.checkpoint.get("id")
                        if latest_checkpoint and latest_checkpoint.checkpoint
                        else None
                    )
                    if checkpoint_id and not await _persist_checkpoint_id(
                        run_id,
                        lease_token,
                        checkpoint_id,
                    ):
                        raise LeaseLostError(
                            f"Task lease lost while saving checkpoint: {run_id}"
                        )

                if kind == "on_chain_start":
                    name = event["name"]
                    meta_node = event.get("metadata", {}).get("langgraph_node")
                    if name in worker_agent_names and name == meta_node:
                        node_input = event.get("data", {}).get("input", {})
                        task_id = node_input.get("current_task_id", "")
                        task_input = node_input.get("task_input", {})
                        description = (
                            task_input.get("description", "")
                            if isinstance(task_input, dict)
                            else ""
                        )
                        if task_id:
                            task_context_by_run_id[str(event.get("run_id", ""))] = {
                                "task_id": task_id,
                                "agent_name": name,
                            }
                        logger.info(f"[chat_task] event: agent_start -> {name} task={task_id}")
                        publish_event(run_id, "agent_start", {
                            "agent_name": name,
                            "task_id": task_id,
                            "description": description,
                            "sequence": event_count,
                        })
                    elif name == scope_node_name and name == meta_node:
                        task_context_by_run_id[str(event.get("run_id", ""))] = {
                            "task_id": "fund_scope",
                            "agent_name": "fund_scope_agent",
                        }
                        logger.info("[chat_task] event: agent_start -> fund_scope")
                        publish_event(run_id, "agent_start", {
                            "agent_name": "fund_scope_agent",
                            "task_id": "fund_scope",
                            "description": "确认当前问题涉及的基金范围",
                            "sequence": event_count,
                        })

                elif kind == "on_chat_model_stream":
                    metadata = event.get("metadata", {})
                    node_name = metadata.get("langgraph_node")

                    if node_name in agent_node_names:
                        evt_run_id = event.get("run_id")
                        if evt_run_id not in seen_run_ids:
                            seen_run_ids.add(evt_run_id)
                            streamed_answer = ""
                            publish_event(run_id, "message_start", {})

                        chunk: AIMessageChunk = event["data"]["chunk"]
                        if chunk.content:
                            streamed_answer += str(chunk.content)
                            publish_event(run_id, "token", {"delta": chunk.content})

                elif kind == "on_chat_model_end":
                    metadata = event.get("metadata", {})
                    node_name = metadata.get("langgraph_node")
                    if node_name in worker_agent_names or node_name == scope_node_name:
                        output = event.get("data", {}).get("output")
                        reasoning = _extract_reasoning_text(output)
                        task_context = _resolve_task_context(
                            event,
                            task_context_by_run_id,
                        )
                        if node_name == scope_node_name and not task_context:
                            task_context = {
                                "task_id": "fund_scope",
                                "agent_name": "fund_scope_agent",
                            }
                        if reasoning and task_context.get("task_id"):
                            task_id = task_context["task_id"]
                            decision_id = f"{run_id}:decision:{trace_sequence + 1}"
                            decision_by_task[task_id] = decision_id
                            decision_tools[decision_id] = []
                            emit_trace("agent_thought", {
                                "thought_id": str(event.get("run_id", "")),
                                "agent_name": task_context.get("agent_name") or node_name,
                                "task_id": task_id,
                                "decision_id": decision_id,
                                "related_tool_call_ids": [],
                                "content": reasoning,
                            })
                            decision_trace_event[decision_id] = trace_events[-1]

                elif kind == "on_chain_end":
                    metadata = event.get("metadata", {})
                    node_name = metadata.get("langgraph_node")
                    is_node_level_event = event.get("name") == node_name

                    if node_name == "route" and is_node_level_event and not _route_emitted:
                        output = event.get("data", {}).get("output")
                        route_result = output.get("route_result") if isinstance(output, dict) else None
                        intent = (
                            route_result.get("intent")
                            if isinstance(route_result, dict)
                            else getattr(route_result, "intent", None)
                        )
                        if intent:
                            _route_emitted = True
                            logger.info(f"[chat_task] event: route_result -> intent={intent}")
                            publish_event(run_id, "route_result", {"intent": intent})

                    if node_name == "supervisor" and is_node_level_event and not _plan_emitted:
                        output = event.get("data", {}).get("output")
                        if isinstance(output, dict) and "plan" in output:
                            plan = output["plan"]
                            plan_list = plan if isinstance(plan, list) else getattr(plan, "tasks", [])
                            if plan_list:
                                _plan_emitted = True
                                plan_summary = [
                                    {
                                        "task_id": t.get("task_id", ""),
                                        "task_type": t.get("task_type", ""),
                                        "description": t.get("description", ""),
                                        "assigned_agent": t.get("assigned_agent", ""),
                                        "fund_codes": t.get("fund_codes", []),
                                    }
                                    for t in plan_list
                                ]
                                reasoning = output.get("reasoning", "") if isinstance(output, dict) else ""
                                logger.info(f"[chat_task] event: plan_created -> {len(plan_summary)} tasks")
                                publish_event(run_id, "plan_created", {
                                    "plan": plan_summary,
                                    "reasoning": reasoning,
                                })

                    if node_name in worker_agent_names and is_node_level_event:
                        output = event.get("data", {}).get("output")
                        status = "completed"
                        task_id = ""
                        if isinstance(output, dict):
                            plan_update = output.get("plan")
                            failed_tasks = output.get("failed_tasks", [])

                            # 并发 Agent 现在返回 TaskPatch，而不是整份 plan 快照。
                            # 保留 list 分支以兼容尚未迁移的节点或历史实现。
                            task_id = getattr(plan_update, "task_id", "")
                            changes = getattr(plan_update, "changes", {})
                            if isinstance(changes, dict):
                                status = changes.get("status", status)

                            if task_id and task_id in failed_tasks:
                                status = "failed"
                        logger.info(f"[chat_task] event: agent_end -> {node_name} status={status}")
                        publish_event(run_id, "agent_end", {
                            "agent_name": node_name,
                            "task_id": task_id,
                            "status": status,
                        })

                    if node_name == scope_node_name and is_node_level_event:
                        output = event.get("data", {}).get("output")
                        status = (
                            "completed"
                            if isinstance(output, dict) and output.get("fund_scope")
                            else "failed"
                        )
                        logger.info("[chat_task] event: agent_end -> fund_scope status=%s", status)
                        publish_event(run_id, "agent_end", {
                            "agent_name": "fund_scope_agent",
                            "task_id": "fund_scope",
                            "status": status,
                        })

                    if node_name == "compliance" and is_node_level_event:
                        output = event.get("data", {}).get("output")
                        if isinstance(output, dict) and output.get("compliance_passed") is False:
                            reason = output.get("compliance_reason") or "内容不符合合规要求"
                            logger.warning(f"[chat_task] event: compliance未通过，reason={reason}")
                            publish_event(run_id, "retry_notice", {"reason": reason})

                    if node_name in (
                        "compliance_failure_handler",
                        "sensitive_refusal",
                        "out_of_scope_refusal",
                        "direct_answer",
                    ) and is_node_level_event:
                        output = event.get("data", {}).get("output")
                        if isinstance(output, dict) and node_name == "direct_answer":
                            content = output.get("draft_answer", "")
                            if content:
                                streamed_answer = str(content)
                                publish_event(run_id, "message_start", {})
                                for char in content:
                                    publish_event(run_id, "token", {"delta": char})
                        elif isinstance(output, dict) and "messages" in output:
                            last_msg = output["messages"][-1]
                            if hasattr(last_msg, "content") and last_msg.content:
                                logger.info(f"[chat_task] event: {node_name}节点返回预设回复")
                                publish_event(run_id, "message_start", {})
                                content = last_msg.content
                                streamed_answer = str(content)
                                for char in content:
                                    publish_event(run_id, "token", {"delta": char})

                elif kind == "on_tool_start":
                    tool_name = event["name"]
                    tool_args = event["data"].get("input", {})
                    parent_node = event.get("metadata", {}).get("langgraph_node")
                    task_context = _resolve_task_context(
                        event,
                        task_context_by_run_id,
                    )
                    # LangChain 为同一次工具执行的 start/end 事件使用同一个 run_id。
                    # 透传该 ID，避免并发同名工具按到达顺序在前端错误配对。
                    payload: dict = {
                        "name": tool_name,
                        "args": tool_args,
                        "tool_call_id": str(event.get("run_id", "")),
                    }
                    agent_name = task_context.get("agent_name") or (
                        parent_node if parent_node in worker_agent_names else ""
                    )
                    if agent_name:
                        payload["agent_name"] = agent_name
                    if task_context.get("task_id"):
                        payload["task_id"] = task_context["task_id"]
                    logger.info(
                        f"[chat_task] event: tool_call -> {tool_name}"
                        + (f" (agent={agent_name})" if agent_name else "")
                        + (f" task={task_context['task_id']}" if task_context.get("task_id") else "")
                    )
                    task_id = payload.get("task_id", "")
                    decision_id = decision_by_task.get(task_id) if task_id else None
                    if decision_id:
                        payload["decision_id"] = decision_id
                        decision_tools.setdefault(decision_id, []).append(payload["tool_call_id"])
                        decision_trace_event[decision_id]["related_tool_call_ids"].append(
                            payload["tool_call_id"]
                        )
                    payload["related_tool_call_ids"] = (
                        [payload["tool_call_id"]] if payload.get("tool_call_id") else []
                    )
                    emit_trace("tool_call", payload)

                elif kind == "on_tool_end":
                    output = event["data"].get("output")
                    output_str = tool_output_to_text(output)
                    task_context = _resolve_task_context(
                        event,
                        task_context_by_run_id,
                    )
                    logger.info(f"[chat_task] event: tool_result -> {event['name']}")
                    payload = {
                        "name": event["name"],
                        "output": output_str,
                        "tool_call_id": str(event.get("run_id", "")),
                    }
                    if task_context.get("agent_name"):
                        payload["agent_name"] = task_context["agent_name"]
                    if task_context.get("task_id"):
                        payload["task_id"] = task_context["task_id"]
                    tool_call_id = payload.get("tool_call_id")
                    decision_id = next(
                        (
                            candidate
                            for candidate, tool_ids in decision_tools.items()
                            if tool_call_id in tool_ids
                        ),
                        None,
                    )
                    if decision_id:
                        payload["decision_id"] = decision_id
                    payload["related_tool_call_ids"] = (
                        [tool_call_id] if tool_call_id else []
                    )
                    emit_trace("tool_result", payload)
            while not retry_events.empty():
                evt_type, evt_data = retry_events.get_nowait()
                logger.info(f"[chat_task] event (final drain): {evt_type} -> {evt_data['agent_name']}")
                emit_trace(evt_type, evt_data)

            if (
                cancel_requested_event is not None
                and cancel_requested_event.is_set()
            ):
                raise TaskCancelledError()

            latest_config = {
                "configurable": {
                    key: value
                    for key, value in config["configurable"].items()
                    if key != "checkpoint_id"
                }
            }
            await app.aupdate_state(
                config=latest_config,
                values={"trace_events": {run_id: trace_events}},
                as_node="__start__",
            )
            final_state = await app.aget_state(latest_config)
            final_messages = (
                final_state.values.get("messages", [])
                if final_state
                else []
            )
            final_message_types = [
                (
                    getattr(message, "type", None)
                    or message.get("type")
                    if isinstance(message, dict)
                    else getattr(message, "type", type(message).__name__)
                )
                for message in final_messages
            ]
            final_has_ai_message = any(
                message_type == "ai" for message_type in final_message_types
            )
            final_checkpoint = await checkpointer.aget_tuple(latest_config)
            final_checkpoint_id = (
                final_checkpoint.checkpoint.get("id")
                if final_checkpoint and final_checkpoint.checkpoint
                else None
            )
            logger.info(
                "[chat_task] final checkpoint saved: run_id=%s session_id=%s "
                "thread_id=%s checkpoint_id=%s streamed_answer=%s "
                "message_count=%s message_types=%s",
                run_id,
                req.session_id,
                latest_config["configurable"].get("thread_id"),
                final_checkpoint_id,
                bool(streamed_answer.strip()),
                len(final_messages),
                final_message_types,
            )
            if streamed_answer.strip() and not final_has_ai_message:
                logger.error(
                    "[chat_task] checkpoint missing AIMessage; "
                    "realtime answer was not persisted: run_id=%s session_id=%s "
                    "thread_id=%s checkpoint_id=%s",
                    run_id,
                    req.session_id,
                    latest_config["configurable"].get("thread_id"),
                    final_checkpoint_id,
                )
            if final_checkpoint_id and not await _persist_checkpoint_id(
                run_id,
                lease_token,
                final_checkpoint_id,
            ):
                raise LeaseLostError(
                    f"Task lease lost while saving final checkpoint: {run_id}"
                )
            if streamed_answer.strip() and not final_has_ai_message:
                raise RuntimeError(
                    "Agent produced a realtime answer, but the final LangGraph "
                    f"checkpoint has no AIMessage: run_id={run_id}, "
                    f"session_id={req.session_id}, "
                    f"checkpoint_id={final_checkpoint_id}"
                )

            logger.info(f"[chat_task] 流式处理完成，共 {event_count} 个事件")
            if lease_lost_event is not None and lease_lost_event.is_set():
                raise LeaseLostError(f"Task lease lost: {run_id}")
            if (
                cancel_requested_event is not None
                and cancel_requested_event.is_set()
            ):
                raise TaskCancelledError()
            publish_event(run_id, "done", {"finish_reason": "stop"})

        except TaskCancelledError:
            raise
        except (asyncio.TimeoutError, asyncio.CancelledError):
            if lease_lost_event is not None and lease_lost_event.is_set():
                raise LeaseLostError(f"Task lease lost: {run_id}")
            if (
                cancel_requested_event is not None
                and cancel_requested_event.is_set()
            ):
                raise TaskCancelledError()
            logger.error(f"[chat_task] 超时: run_id={run_id}, session_id={req.session_id}")
            publish_event(run_id, "error", {"message": "处理超时，请稍后重试"})
            raise
        except LeaseLostError:
            raise
        except Exception as e:
            logger.exception(f"[chat_task] 处理异常: session_id={req.session_id}")
            publish_event(run_id, "error", {"message": str(e)})
            raise

        try:
            final_state = await app.aget_state(config)
            token_usage = final_state.values.get("token_usage", {}) if final_state else {}
            if token_usage:
                total_tokens = sum(u.get("total_tokens", 0) for u in token_usage.values())
                logger.info(f"[chat_task] Token 用量: session_id={req.session_id}, total={total_tokens}, 明细={token_usage}")
        except Exception:
            logger.exception(f"[chat_task] 读取 token 用量失败: session_id={req.session_id}")

        try:
            mcp_client = await get_mcp_client()
            stats = await mcp_client.get_call_stats(user_id=str(user_id))
            logger.info(f"[chat_task] MCP 调用统计（用户当前窗口）: {stats}")
        except Exception:
            logger.exception(f"[chat_task] 读取 MCP 调用统计失败: session_id={req.session_id}")

        logger.info(f"[chat_task] 请求结束: session_id={req.session_id}")
        logger.info("=" * 80)

    finally:
        lock_stop_event.set()
        lock_heartbeat.cancel()
        try:
            await lock_heartbeat
        except asyncio.CancelledError:
            pass
        try:
            await lock.release()
        except Exception as e:
            # LockNotOwnedError 是预期行为：任务超时时锁可能已被 Redis 自动过期。
            # 此时只需 warning，不需要完整 traceback。
            if "LockNotOwnedError" in type(e).__name__ or "no longer owned" in str(e):
                logger.warning(
                    f"[chat_task] 锁已自动过期（任务可能超时）: session_id={req.session_id}"
                )
            else:
                logger.exception(f"[chat_task] 释放会话锁失败: session_id={req.session_id}")


async def _run_chat_turn_with_lease(
    run_id: str,
    req: ChatRequest,
    user_id: int,
    lease_token: str,
    checkpoint_id: str | None,
    lease_seconds: int,
    heartbeat_interval_seconds: int,
) -> None:
    stop_event = asyncio.Event()
    lease_lost_event = asyncio.Event()
    cancel_requested_event = asyncio.Event()
    heartbeat = asyncio.create_task(
        _heartbeat_loop(
            run_id,
            lease_token,
            lease_seconds,
            heartbeat_interval_seconds,
            stop_event,
            lease_lost_event,
        )
    )
    chat_task = asyncio.create_task(
        _run_chat_turn(
            run_id=run_id,
            req=req,
            user_id=user_id,
            lease_token=lease_token,
            checkpoint_id=checkpoint_id,
            lease_lost_event=lease_lost_event,
            cancel_requested_event=cancel_requested_event,
        )
    )
    lease_lost_waiter = asyncio.create_task(lease_lost_event.wait())
    cancel_waiter = asyncio.create_task(cancel_requested_event.wait())
    cancel_monitor = asyncio.create_task(
        _cancel_monitor_loop(
            run_id,
            lease_token,
            stop_event,
            cancel_requested_event,
        )
    )
    try:
        done, _ = await asyncio.wait(
            {chat_task, lease_lost_waiter, cancel_waiter},
            return_when=asyncio.FIRST_COMPLETED,
        )
        if lease_lost_waiter in done:
            chat_task.cancel()
            try:
                await chat_task
            except asyncio.CancelledError:
                pass
            raise LeaseLostError(f"Task lease lost: {run_id}")
        if cancel_waiter in done:
            chat_task.cancel()
            try:
                await chat_task
            except (asyncio.CancelledError, TaskCancelledError):
                pass
            raise TaskCancelledError()

        await chat_task
        if lease_lost_event.is_set():
            raise LeaseLostError(f"Task lease lost: {run_id}")
    finally:
        stop_event.set()
        heartbeat.cancel()
        lease_lost_waiter.cancel()
        cancel_waiter.cancel()
        cancel_monitor.cancel()
        if not chat_task.done():
            chat_task.cancel()
        try:
            await heartbeat
        except asyncio.CancelledError:
            pass
        try:
            await lease_lost_waiter
        except asyncio.CancelledError:
            pass
        try:
            await cancel_waiter
        except asyncio.CancelledError:
            pass
        try:
            await cancel_monitor
        except asyncio.CancelledError:
            pass
        if not chat_task.done():
            try:
                await chat_task
            except asyncio.CancelledError:
                pass


@celery_app.task(bind=True, name="app.tasks.chat_tasks.run_chat_turn")
def run_chat_turn(
    self,
    run_id: str,
    req_payload: dict,
    user_id: int,
) -> None:
    """Celery 任务入口（同步）：把请求还原为 ChatRequest，在 worker 的后台事件
    循环上跑完整的 agent 流程，逐事件通过 Redis 发布给 FastAPI 侧转发。

    超时不用 Celery 的 soft_time_limit（基于主线程信号，无法中断跑在后台
    事件循环线程里的协程，见 worker_lifecycle.run_coro 的说明），而是把
    AGENT_TIMEOUT 作为 asyncio.wait_for 的超时传给 run_coro，在协程内部
    真正取消。Celery 的 task_soft_time_limit/task_time_limit 仍保留作为
    兜底（防止事件循环线程本身死锁导致 run_coro 永久阻塞主线程）。
    """
    req = ChatRequest(
        message=req_payload["message"],
        session_id=req_payload["session_id"],
        history=[ChatMessage(**h) for h in req_payload.get("history", [])],
    )
    settings = get_settings()
    attempt = self.request.retries + 1
    task_run = run_coro(
        _claim_task_run(
            run_id,
            attempt,
            settings.AGENT_TIMEOUT,
            settings.TASK_LEASE_SECONDS,
        ),
        timeout=30,
    )
    if task_run is None:
        logger.info("[chat_task] task was already claimed or completed: run_id=%s", run_id)
        return
    lease_token = task_run.lease_token
    if lease_token is None:
        raise RuntimeError(f"Claimed task has no lease token: {run_id}")
    max_attempts = max(1, task_run.max_attempts)
    logger.info(
        "[chat_task] claimed task: celery_task_id=%s run_id=%s session_id=%s "
        "thread_id=%s checkpoint_id=%s recovery_from_checkpoint=%s attempt=%s",
        self.request.id,
        run_id,
        req.session_id,
        req.session_id,
        task_run.checkpoint_id,
        task_run.checkpoint_id is not None,
        task_run.attempt,
    )
    try:
        run_coro(
            _run_chat_turn_with_lease(
                run_id,
                req,
                user_id,
                lease_token,
                task_run.checkpoint_id,
                settings.TASK_LEASE_SECONDS,
                settings.TASK_HEARTBEAT_INTERVAL_SECONDS,
            ),
            timeout=settings.AGENT_TIMEOUT,
        )
    except TaskCancelledError:
        updated = run_coro(
            _mark_cancelled(run_id, lease_token),
            timeout=30,
        )
        if updated:
            publish_event(run_id, "cancelled", {"message": "Task cancelled by user"})
        return
    except (asyncio.TimeoutError, asyncio.CancelledError):
        updated = run_coro(
            _mark_finished(
                run_id,
                lease_token,
                "TIMED_OUT",
                "AGENT_TIMEOUT",
                f"Agent exceeded {settings.AGENT_TIMEOUT} seconds",
            ),
            timeout=30,
        )
        if updated:
            publish_event(run_id, "error", {"message": "Task execution timed out"})
        raise
    except LeaseLostError:
        logger.warning("[chat_task] task stopped after losing lease: run_id=%s", run_id)
        raise
    except Exception as exc:
        error_code = _failure_code(exc)
        error_message = str(exc) or type(exc).__name__
        if _is_retryable_failure(exc) and attempt < max_attempts:
            updated = run_coro(
                _mark_retrying(run_id, lease_token, error_code, error_message),
                timeout=30,
            )
            if not updated:
                cancelled = run_coro(
                    _mark_cancelled(run_id, lease_token),
                    timeout=30,
                )
                if cancelled:
                    publish_event(
                        run_id,
                        "cancelled",
                        {"message": "Task cancelled by user"},
                    )
                    return
                logger.warning(
                    "[chat_task] retry state rejected because lease was lost: run_id=%s",
                    run_id,
                )
                raise
            publish_event(
                run_id,
                "retry_notice",
                {
                    "reason": (
                        f"Temporary dependency failure; retrying "
                        f"({attempt + 1}/{max_attempts})"
                    ),
                },
            )
            countdown = min(
                settings.CELERY_RETRY_BACKOFF_SECONDS * (2 ** (self.request.retries + 1)),
                settings.CELERY_RETRY_BACKOFF_MAX_SECONDS,
            )
            raise self.retry(
                exc=exc,
                countdown=countdown,
                max_retries=max_attempts - 1,
            )
        updated = run_coro(
            _mark_finished(
                run_id,
                lease_token,
                "FAILED",
                error_code,
                error_message,
            ),
            timeout=30,
        )
        if updated:
            publish_event(run_id, "error", {"message": "Task execution failed"})
        else:
            cancelled = run_coro(
                _mark_cancelled(run_id, lease_token),
                timeout=30,
            )
            if cancelled:
                publish_event(
                    run_id,
                    "cancelled",
                    {"message": "Task cancelled by user"},
                )
                return
        raise
    else:
        updated = run_coro(
            _mark_finished(run_id, lease_token, "SUCCESS"),
            timeout=30,
        )
        if not updated:
            cancelled = run_coro(
                _mark_cancelled(run_id, lease_token),
                timeout=30,
            )
            if cancelled:
                publish_event(
                    run_id,
                    "cancelled",
                    {"message": "Task cancelled by user"},
                )
            else:
                logger.warning(
                    "[chat_task] success state rejected because lease was lost: run_id=%s",
                    run_id,
                )
        return
