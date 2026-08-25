"""add task run checkpoint id

Revision ID: a7b8c9d0e1f2
Revises: f6a7b8c9d0e1
Create Date: 2026-08-24 14:40:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "a7b8c9d0e1f2"
down_revision: Union[str, Sequence[str], None] = "f6a7b8c9d0e1"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "task_runs",
        sa.Column("checkpoint_id", sa.String(length=64), nullable=True),
    )
    op.create_index(
        "ix_task_runs_checkpoint_id",
        "task_runs",
        ["checkpoint_id"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index("ix_task_runs_checkpoint_id", table_name="task_runs")
    op.drop_column("task_runs", "checkpoint_id")
