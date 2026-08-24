"""allow task runs after session delete

Revision ID: d4e5f6a7b8c9
Revises: c3d4e5f6a7b8
Create Date: 2026-08-24 10:20:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "d4e5f6a7b8c9"
down_revision: Union[str, Sequence[str], None] = "c3d4e5f6a7b8"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.drop_constraint("task_runs_ibfk_2", "task_runs", type_="foreignkey")
    op.alter_column(
        "task_runs",
        "session_id",
        existing_type=sa.String(length=64),
        nullable=True,
    )
    op.create_foreign_key(
        "fk_task_runs_session_id_sessions",
        "task_runs",
        "sessions",
        ["session_id"],
        ["thread_id"],
        ondelete="SET NULL",
    )


def downgrade() -> None:
    op.drop_constraint(
        "fk_task_runs_session_id_sessions",
        "task_runs",
        type_="foreignkey",
    )
    op.alter_column(
        "task_runs",
        "session_id",
        existing_type=sa.String(length=64),
        nullable=False,
    )
    op.create_foreign_key(
        "task_runs_ibfk_2",
        "task_runs",
        "sessions",
        ["session_id"],
        ["thread_id"],
    )
