# The prediction API. The model is not in the image: at startup the API loads the version
# @MODEL_ALIAS points to from the MLflow registry, so changing the model needs no rebuild.

# ---- build: resolve and install the serving dependencies into a virtualenv ----
FROM python:3.12-slim AS builder
# ONNX models need only the serving group; serving a PyTorch model needs torch too:
#   docker compose build --build-arg UV_GROUPS="--group serving --group pytorch" api
ARG UV_GROUPS="--group serving"
COPY --from=ghcr.io/astral-sh/uv:0.13.5 /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy
WORKDIR /app
COPY pyproject.toml uv.lock .python-version ./
# Dependencies first: this layer is reused until the lockfile changes (the cache keeps downloads between builds)
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-default-groups $UV_GROUPS --no-install-project
COPY README.md ./
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-default-groups $UV_GROUPS --no-editable

# ---- runtime: only what runs ----
FROM python:3.12-slim
RUN useradd --create-home appuser
WORKDIR /app
# The whole virtualenv, not only site-packages: bin/ holds the uvicorn command
COPY --from=builder /app/.venv /app/.venv
ENV PATH="/app/.venv/bin:$PATH" PYTHONUNBUFFERED=1
USER appuser
EXPOSE 8000
# Ready only once the model is loaded; allow time to download and load it
HEALTHCHECK --interval=10s --timeout=5s --start-period=300s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/v1/ready')"
CMD ["uvicorn", "arabic_banking_intents.serving.main:app", "--host", "0.0.0.0", "--port", "8000"]
