from datetime import datetime

from sqlalchemy import (
    BigInteger,
    Boolean,
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import ARRAY, JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

REGIONS = ("AU", "GULF", "REMOTE", "BACKFILL")
BOARDS = ("greenhouse", "lever", "ashby", "recruitee", "workable", "teamtailor")
REMOTE = ("eligible", "region_locked", "hybrid", "onsite", "unclear")
LABELS = ("fit", "not_fit", "stretch")


def _in(col: str, values: tuple[str, ...]) -> str:
    return f"{col} IN ({', '.join(repr(v) for v in values)})"


class Base(DeclarativeBase):
    pass


class Company(Base):
    __tablename__ = "companies"
    __table_args__ = (
        CheckConstraint(_in("region", REGIONS), name="ck_companies_region"),
        CheckConstraint(f"board IS NULL OR {_in('board', BOARDS)}", name="ck_companies_board"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(Text, unique=True)
    website: Mapped[str | None] = mapped_column(Text)
    region: Mapped[str] = mapped_column(String(16))
    board: Mapped[str | None] = mapped_column(String(16))
    slug: Mapped[str | None] = mapped_column(Text)
    verified: Mapped[bool] = mapped_column(Boolean, server_default="false")
    last_fetched_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    last_error: Mapped[str | None] = mapped_column(Text)


class Post(Base):
    __tablename__ = "posts"
    __table_args__ = (
        UniqueConstraint("provider", "external_id", name="uq_posts_provider_external_id"),
        CheckConstraint(_in("remote_eligibility", REMOTE), name="ck_posts_remote"),
        Index("ix_posts_fingerprint", "fingerprint"),
        Index("ix_posts_status", "status"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    company_id: Mapped[int] = mapped_column(ForeignKey("companies.id"))
    provider: Mapped[str] = mapped_column(String(16))
    external_id: Mapped[str] = mapped_column(Text)
    url: Mapped[str] = mapped_column(Text)
    title: Mapped[str] = mapped_column(Text)
    location: Mapped[str | None] = mapped_column(Text)
    description_text: Mapped[str] = mapped_column(Text)
    raw: Mapped[dict] = mapped_column(JSONB)
    remote_eligibility: Mapped[str] = mapped_column(String(16), server_default="unclear")
    comp_min: Mapped[float | None] = mapped_column(Numeric)
    comp_max: Mapped[float | None] = mapped_column(Numeric)
    comp_currency: Mapped[str | None] = mapped_column(String(8))
    fingerprint: Mapped[str] = mapped_column(String(64))
    content_hash: Mapped[str] = mapped_column(String(64))
    status: Mapped[str] = mapped_column(String(24), server_default="new")
    first_seen_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    last_seen_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    closed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    version: Mapped[int] = mapped_column(Integer, server_default="1")
    duplicate_of: Mapped[int | None] = mapped_column(ForeignKey("posts.id"))


class TriageRun(Base):
    __tablename__ = "triage_runs"
    __table_args__ = (
        CheckConstraint("decided_by IS NULL OR decided_by IN ('auto', 'human')", name="ck_runs_decided_by"),
        UniqueConstraint("post_id", "post_version", "criteria_version", name="uq_runs_post_version_criteria"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    post_id: Mapped[int] = mapped_column(ForeignKey("posts.id"), index=True)
    post_version: Mapped[int] = mapped_column(Integer)
    criteria_version: Mapped[str] = mapped_column(String(64))
    jev_p: Mapped[float | None] = mapped_column(Float)
    hi: Mapped[float | None] = mapped_column(Float)
    lo: Mapped[float | None] = mapped_column(Float)
    route: Mapped[str | None] = mapped_column(String(24))
    deep_read: Mapped[dict | None] = mapped_column(JSONB)
    final_status: Mapped[str | None] = mapped_column(String(24))
    decided_by: Mapped[str | None] = mapped_column(String(8))
    decision: Mapped[str | None] = mapped_column(String(16))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    latency_ms: Mapped[int | None] = mapped_column(BigInteger)
    error: Mapped[str | None] = mapped_column(Text)
    reason: Mapped[str | None] = mapped_column(Text)    # why the prefilter dropped it


class Label(Base):
    __tablename__ = "labels"
    __table_args__ = (
        CheckConstraint(_in("label", LABELS), name="ck_labels_label"),
        CheckConstraint("source IN ('gold', 'approval')", name="ck_labels_source"),
    )

    post_id: Mapped[int] = mapped_column(ForeignKey("posts.id"), primary_key=True)
    source: Mapped[str] = mapped_column(String(8), primary_key=True)
    label: Mapped[str] = mapped_column(String(8))
    notes: Mapped[str | None] = mapped_column(Text)
    tags: Mapped[list[str]] = mapped_column(ARRAY(Text), server_default="{}")   # #hashtags typed in the note, e.g. research
    labeled_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
