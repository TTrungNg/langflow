FROM ghcr.io/astral-sh/uv:python3.14-bookworm-slim
ENV TZ=UTC
ENV UV_HTTP_TIMEOUT=300
ENV UV_CONCURRENT_DOWNLOADS=4
WORKDIR /app

RUN apt-get update \
    && apt-get upgrade -y \
    && apt-get install -y build-essential curl git libpq5 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy full source tree — uv needs workspace member dirs to resolve the graph.
# The src/ directory will be overridden by a volume mount at runtime for hot-reload.
COPY . /app

# Mirror the exact bind-mount pattern from dev.Dockerfile that is known to work.
# Bind mounts overlay the COPYed files during this RUN step with the build-context
# versions, ensuring the freshest lockfiles are used even if the COPY layer is cached.
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=README.md,target=README.md \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    --mount=type=bind,source=src/backend/base/README.md,target=src/backend/base/README.md \
    --mount=type=bind,source=src/backend/base/uv.lock,target=src/backend/base/uv.lock \
    --mount=type=bind,source=src/backend/base/pyproject.toml,target=src/backend/base/pyproject.toml \
    --mount=type=bind,source=src/lfx/README.md,target=src/lfx/README.md \
    --mount=type=bind,source=src/lfx/pyproject.toml,target=src/lfx/pyproject.toml \
    --mount=type=bind,source=src/sdk/README.md,target=src/sdk/README.md \
    --mount=type=bind,source=src/sdk/pyproject.toml,target=src/sdk/pyproject.toml \
    uv sync --frozen --no-dev --extra postgresql

# Activate the venv directly — faster than going through uv run at every startup
ENV PATH="/app/.venv/bin:$PATH"

EXPOSE 7860

# --reload-dir scoped to /app/src so uvicorn only watches the mounted source tree
CMD ["uvicorn", \
     "--factory", "langflow.main:create_app", \
     "--host", "0.0.0.0", \
     "--port", "7860", \
     "--reload", \
     "--reload-dir", "/app/src", \
     "--loop", "asyncio"]
