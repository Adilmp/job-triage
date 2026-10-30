FROM python:3.12-slim
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy PYTHONUNBUFFERED=1
WORKDIR /app
COPY pyproject.toml uv.lock* ./
RUN uv sync --no-install-project --no-dev
COPY src ./src
COPY alembic ./alembic
COPY alembic.ini ./
RUN uv sync --no-dev
ENV PATH="/app/.venv/bin:$PATH"
CMD ["uvicorn", "triage.api:app", "--host", "0.0.0.0", "--port", "8000"]
