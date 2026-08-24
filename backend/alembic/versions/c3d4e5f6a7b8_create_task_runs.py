"""create task_runs

Revision ID: c3d4e5f6a7b8
Revises: a1b2c3d4e5f6
Create Date: 2026-08-24 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "c3d4e5f6a7b8"
down_revision: Union[str, Sequence[str], None] = "a1b2c3d4e5f6"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

_task_run_status = sa.Enum(
    "QUEUED",
    "RUNNING",
    "SUCCESS",
    "FAILED",
    "CANCELLED",
    "TIMED_OUT",
    "LOST",
    name="task_run_status",
)


def upgrade() -> None:
    _task_run_status.create(op.get_bind(), checkfirst=True)
    op.create_table(
        "task_runs",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("run_id", sa.String(length=64), nullable=False),
        sa.Column("idempotency_key", sa.String(length=128), nullable=False),
        sa.Column("celery_task_id", sa.String(length=255), nullable=True),
        sa.Column("user_id", sa.BigInteger(), nullable=False),
        sa.Column("session_id", sa.String(length=64), nullable=False),
        sa.Column("status", _task_run_status, server_default="QUEUED", nullable=False),
        sa.Column("attempt", sa.Integer(), server_default="0", nullable=False),
        sa.Column("max_attempts", sa.Integer(), server_default="1", nullable=False),
        sa.Column("cancel_requested", sa.Boolean(), server_default=sa.text("0"), nullable=False),
        sa.Column("created_at", sa.DateTime(), server_default=sa.text("now()"), nullable=False),
        sa.Column("started_at", sa.DateTime(), nullable=True),
        sa.Column("finished_at", sa.DateTime(), nullable=True),
        sa.Column("heartbeat_at", sa.DateTime(), nullable=True),
        sa.Column("deadline_at", sa.DateTime(), nullable=True),
        sa.Column("error_code", sa.String(length=64), nullable=True),
        sa.Column("error_message", sa.String(length=1024), nullable=True),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"]),
        sa.ForeignKeyConstraint(["session_id"], ["sessions.thread_id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "user_id",
            "idempotency_key",
            name="uq_task_runs_user_idempotency",
        ),
    )
    op.create_index("ix_task_runs_run_id", "task_runs", ["run_id"], unique=True)
    op.create_index("ix_task_runs_celery_task_id", "task_runs", ["celery_task_id"], unique=True)
    op.create_index("ix_task_runs_user_id", "task_runs", ["user_id"], unique=False)
    op.create_index("ix_task_runs_session_id", "task_runs", ["session_id"], unique=False)
    op.create_index("ix_task_runs_status", "task_runs", ["status"], unique=False)


def downgrade() -> None:
    op.drop_index("ix_task_runs_status", table_name="task_runs")
    op.drop_index("ix_task_runs_session_id", table_name="task_runs")
    op.drop_index("ix_task_runs_user_id", table_name="task_runs")
    op.drop_index("ix_task_runs_celery_task_id", table_name="task_runs")
    op.drop_index("ix_task_runs_run_id", table_name="task_runs")
    op.drop_table("task_runs")
    _task_run_status.drop(op.get_bind(), checkfirst=True)
