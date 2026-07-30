"""Celery app 实例与配置。

Worker 是独立的 OS 进程，不与 FastAPI 共享任何内存状态（MCP client、
checkpoint pool、MySQL engine、Redis client 等均需在 worker 进程内自行
初始化，见 app.core.worker_lifecycle）。

Windows 开发：prefork（默认）在 Windows 上不可用，启动时必须加 --pool=solo：
    celery -A app.core.celery_app worker --pool=solo --loglevel=info

Docker/Linux 部署后可切换为 prefork 或 gevent 以获得并发（mcp还是无法并发）：
    celery -A app.core.celery_app worker --pool=prefork --concurrency=4 --loglevel=info
"""
from celery import Celery
from celery.signals import worker_init, worker_shutdown

from app.core.config import get_settings

settings = get_settings()

celery_app = Celery(
    "fund_analyser",
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND,
    include=["app.tasks.chat_tasks"],
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="Asia/Shanghai",
    enable_utc=True,

    # 一个 worker 进程一次只处理一个 agent 任务再取下一个，避免长任务把同进程
    # 的其他任务饿死（agent 任务通常是分钟级，不是秒级）
    worker_prefetch_multiplier=1,

    # 任务开始执行后才 ack，worker 崩溃时任务会被重新投递，不会静默丢失
    task_acks_late=True,
    task_reject_on_worker_lost=True,

    # 让 flower / inspect 能看到 STARTED 状态，便于观察长任务
    task_track_started=True,

    # 结果保留时间，超时后从 result backend 清理
    result_expires=settings.CELERY_RESULT_EXPIRES,

    # 超时兜底：对齐 AGENT_TIMEOUT，防止某个 agent 运行卡死占住 worker
    task_soft_time_limit=settings.CELERY_TASK_SOFT_TIME_LIMIT,
    task_time_limit=settings.CELERY_TASK_TIME_LIMIT,

    # chat 任务单独分到 agent_queue，便于后续按队列优先级/独立扩容（例如给
    # agent 任务单开更多 worker，或给非 agent 任务设更短的超时）。
    # 启动 worker 必须带 -Q default,agent_queue，否则任务会投递到没人消费的
    # 队列里静默卡死（不报错、无日志、无 SSE 事件）——见 README「Celery 任务队列」。
    task_routes={
        "app.tasks.chat_tasks.*": {"queue": "agent_queue"},
    },
    task_default_queue="default",
)

from app.core import worker_lifecycle  # noqa: E402  (注册 signal handlers)

# --pool=solo 下 worker 本身即主进程，不会 fork 子进程，worker_process_init/
# shutdown 不保证触发（见 celery#5405）；worker_init/worker_shutdown 在所有
# pool 类型（solo/prefork/gevent）下都可靠触发，因此用它们承载资源初始化。
# prefork/gevent 下 worker_init 在主进程触发一次，实际任务在子进程/greenlet
# 中执行；本项目每个 worker 只用 --concurrency=1（Windows 用 solo 天然如此，
# Docker 化后如需要多并发，应改为多个独立 worker 进程而非单进程内多子进程，
# 以保持“每进程一份 MCP/连接池”的模型简单可靠）。
worker_init.connect(worker_lifecycle.on_worker_process_init)
worker_shutdown.connect(worker_lifecycle.on_worker_process_shutdown)
