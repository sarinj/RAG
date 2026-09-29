FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

WORKDIR /app

COPY --from=ghcr.io/astral-sh/uv:0.12.20 /uv /uvx /bin/
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
COPY scripts ./scripts

RUN uv sync --locked --no-dev

EXPOSE 8000
CMD ["uv", "run", "--no-sync", "uvicorn", "rag_app.main:app", "--host", "0.0.0.0", "--port", "8000"]
