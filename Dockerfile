FROM python:3.12-slim

WORKDIR /app

# Install dependencies first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code and metadata
COPY src/ ./src/
COPY pyproject.toml README.md ./

# Install the package (non-editable for production)
RUN pip install --no-cache-dir .

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app/src

# Expose the server port
EXPOSE 8000

# Default command - can be overridden by k8s
CMD ["python", "-m", "fractal_agents.server"]
