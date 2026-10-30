import pytest
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine

from triage.models import Base


@pytest.fixture(scope="session")
def pg_url():
    """A throwaway Postgres 16 (P16): upsert and versioning SQL is tested against the real thing."""
    testcontainers = pytest.importorskip("testcontainers.postgres")
    try:
        container = testcontainers.PostgresContainer("postgres:16")
        container.start()
    except Exception as e:      # no Docker daemon
        pytest.skip(f"Docker unavailable: {e}")
    try:
        yield (f"postgresql+asyncpg://{container.username}:{container.password}@"
               f"{container.get_container_host_ip()}:{container.get_exposed_port(5432)}/{container.dbname}")
    finally:
        container.stop()


@pytest.fixture
async def session(pg_url):
    engine = create_async_engine(pg_url)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all)
    async with AsyncSession(engine, expire_on_commit=False) as s:
        yield s
    await engine.dispose()
