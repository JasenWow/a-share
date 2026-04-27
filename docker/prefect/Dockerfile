# syntax=docker/dockerfile:1
FROM python:3.12-slim

# Install uv for fast dependency management
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Create app directory
WORKDIR /app/big-a

# Copy lockfile and project manifest first (for better cache)
COPY big-a/pyproject.toml big-a/uv.lock ./

# Install dependencies (without project source, then install project later)
RUN uv sync --frozen --no-install-project

# Copy project source code
COPY big-a/src ./src

# Install the project itself
RUN uv sync --frozen

# Set Python path
ENV PYTHONPATH=/app/big-a/src

# Default: run the Prefect scheduled flow
CMD ["python", "-c", "from big_a.scheduler.flows import daily_pipeline; daily_pipeline.serve(cron='0 19 * * 1-5')"]
