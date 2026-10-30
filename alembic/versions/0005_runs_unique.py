"""triage_runs: one row per (post, post version, criteria version)

Revision ID: 0005
Revises: 0004
"""
from alembic import op

revision = "0005"
down_revision = "0004"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_unique_constraint("uq_runs_post_version_criteria", "triage_runs", ["post_id", "post_version", "criteria_version"])


def downgrade() -> None:
    op.drop_constraint("uq_runs_post_version_criteria", "triage_runs", type_="unique")
