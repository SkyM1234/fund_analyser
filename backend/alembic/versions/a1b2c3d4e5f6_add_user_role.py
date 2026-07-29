"""add user role

Revision ID: a1b2c3d4e5f6
Revises: 88116c7fba19
Create Date: 2026-07-23 10:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'a1b2c3d4e5f6'
down_revision: Union[str, Sequence[str], None] = '88116c7fba19'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

_user_role = sa.Enum("user", "admin", name="user_role")


def upgrade() -> None:
    """Upgrade schema."""
    _user_role.create(op.get_bind(), checkfirst=True)
    op.add_column(
        'users',
        sa.Column('role', _user_role, nullable=False, server_default='user'),
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column('users', 'role')
    _user_role.drop(op.get_bind(), checkfirst=True)
