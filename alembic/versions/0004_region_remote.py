"""companies.region: add REMOTE (remote-first employers added to widen the pool)

Revision ID: 0004
Revises: 0003
"""
from alembic import op

revision = "0004"
down_revision = "0003"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.drop_constraint("ck_companies_region", "companies", type_="check")
    op.create_check_constraint("ck_companies_region", "companies", "region IN ('AU', 'GULF', 'REMOTE', 'BACKFILL')")


def downgrade() -> None:
    op.execute("UPDATE companies SET region = 'BACKFILL' WHERE region = 'REMOTE'")
    op.drop_constraint("ck_companies_region", "companies", type_="check")
    op.create_check_constraint("ck_companies_region", "companies", "region IN ('AU', 'GULF', 'BACKFILL')")
