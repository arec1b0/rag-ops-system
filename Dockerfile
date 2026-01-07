# Base image
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

# Install system dependencies (curl needed for healthchecks if not present)
RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*

# Set work directory
WORKDIR /app

# Install uv for fast dependency management
RUN pip install uv

# Copy dependency files first (for caching)
COPY pyproject.toml .

# Install dependencies using uv
# --system flag installs into the system python, avoiding the need for a venv inside the container
RUN uv pip install --system -r pyproject.toml

# Copy the rest of the application
COPY src/ src/
COPY tests/ tests/

# Create a non-root user for security
RUN adduser --disabled-password --gecos '' appuser
USER appuser

# Expose the port
EXPOSE 8000

# Command to run the application
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]