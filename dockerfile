# Use a lightweight official Python image
FROM python:3.13-slim

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Set environment variables to optimize Python/Poetry
ENV POETRY_VERSION=2.3.2 \
    POETRY_VIRTUALENVS_CREATE=false \
    PYTHONUNBUFFERED=1 \
    DEBUG=true

# Install Poetry
RUN pip install "poetry==$POETRY_VERSION"

# Copy dependency files first for better caching
COPY pyproject.toml poetry.lock ./

# Install dependencies (skip dev dependencies for production)
RUN poetry install --no-root --no-interaction --no-ansi

# Copy application code
COPY . .

# Create non-root user
RUN useradd --create-home --shell /bin/bash app \
    && chown -R app:app /app
USER app


# Expose the port Cloud Run expects (defaults to 8080)
EXPOSE 8080

# Run the API using Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]