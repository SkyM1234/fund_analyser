"""add task run lease

Revision ID: e5f6a7b8c9d0
Revises: d4e5f6a7b8c9
Create Date: 2026-08-24 10:35:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "e5f6a7b8c9d0"
down_revision: Union[str, Sequence[str], None] = "d4e5f6a7b8c9"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "task_runs",
        sa.Column("lease_token", sa.String(length=64), nullable=True),
    )
    op.add_column(
        "task_runs",
        sa.Column("lease_expires_at", sa.DateTime(), nullable=True),
    )
    op.create_index(
        "ix_task_runs_lease_expires_at",
        "task_runs",
        ["lease_expires_at"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index("ix_task_runs_lease_expires_at", table_name="task_runs")
    op.drop_column("task_runs", "lease_expires_at")
    op.drop_column("task_runs", "lease_token")
