"""triage_runs.reason: why the prefilter dropped a post

Revision ID: 0006
Revises: 0005
"""
from alembic import op
import sqlalchemy as sa

revision = "0006"
down_revision = "0005"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("triage_runs", sa.Column("reason", sa.Text(), nullable=True))


def downgrade() -> None:
    op.drop_column("triage_runs", "reason")
