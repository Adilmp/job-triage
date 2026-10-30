"""labels: allow 'stretch', add tags

Revision ID: 0003
Revises: 0002
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "0003"
down_revision = "0002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.drop_constraint("ck_labels_label", "labels", type_="check")
    op.create_check_constraint("ck_labels_label", "labels", "label IN ('fit', 'not_fit', 'stretch')")
    op.add_column("labels", sa.Column("tags", postgresql.ARRAY(sa.Text()), server_default="{}", nullable=False))


def downgrade() -> None:
    op.drop_column("labels", "tags")
    op.execute("UPDATE labels SET label = 'fit' WHERE label = 'stretch'")
    op.drop_constraint("ck_labels_label", "labels", type_="check")
    op.create_check_constraint("ck_labels_label", "labels", "label IN ('fit', 'not_fit')")
