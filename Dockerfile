FROM python:3.12-slim

WORKDIR /app

# Copy the application source
COPY application/ /app/

# Install uv then sync all dependencies from pyproject.toml / uv.lock
RUN pip install uv && uv sync --frozen

# uv creates a venv at /app/.venv — add it to PATH
ENV PATH="/app/.venv/bin:$PATH"

# SQLite DB will be written to /app/rag_logs.db inside the container.
# Mount a volume here in production to persist data across restarts:
#   docker run -v /host/path:/app/rag_logs.db ...
ENV RAG_DB_PATH="/app/rag_logs.db"

CMD ["gunicorn", "-w", "2", "--threads", "2", "-b", "0.0.0.0:8000", "app:app"]
