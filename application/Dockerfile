# ==========================================
# Stage 1: Build Environment
# ==========================================
FROM python:3.12-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app
ENV UV_PROJECT_ENVIRONMENT="/opt/venv"

# Copy lock files and install dependencies
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project --no-dev --extra production

# ==========================================
# Stage 2: Runtime Environment
# ==========================================
FROM python:3.12-slim

WORKDIR /app

# Bring in the built virtual environment
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# ONLY copy the source code folder into the container
COPY ./src ./src

EXPOSE 8000

# Run gunicorn pointing to the app inside the src folder
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "src.app:app"]
# CMD ["gunicorn", "--bind", "0.0.0.0:8000", "--error-logfile", "-", "--access-logfile", "-", "--log-level", "debug", "src.app:app"]