# Multi-stage build for optimized image size
FROM python:3.10-slim AS builder

# Set working directory
WORKDIR /build

COPY src/requirements.txt .

# Install dependencies in a virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# Final stage
FROM python:3.10-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH" \
    MODEL_OUTPUT_DIR="/app/models"

# Create non-root user for security
RUN useradd -m -u 1000 mluser && \
    mkdir -p /app/models && \
    chown -R mluser:mluser /app

# Set working directory
WORKDIR /app

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv

# Copy application code
COPY --chown=mluser:mluser src/ .

# Switch to non-root user
USER mluser

# Run the script when the container launches
CMD ["python", "main.py"]
