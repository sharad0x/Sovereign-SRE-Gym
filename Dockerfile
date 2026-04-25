# --------------------------------------------------
# Base Image (Lightweight + Stable)
# --------------------------------------------------
FROM python:3.10-slim

# --------------------------------------------------
# Environment Variables
# --------------------------------------------------
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PORT=8000

# --------------------------------------------------
# System Dependencies (minimal but sufficient)
# --------------------------------------------------
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# --------------------------------------------------
# Create non-root user (HF best practice)
# --------------------------------------------------
RUN useradd -m appuser

# --------------------------------------------------
# Set Working Directory
# --------------------------------------------------
WORKDIR /app

# --------------------------------------------------
# Copy Dependency Files First (cache optimization)
# --------------------------------------------------
COPY requirements.txt .

# --------------------------------------------------
# Install Python Dependencies
# --------------------------------------------------
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# --------------------------------------------------
# Copy Full Project
# --------------------------------------------------
COPY . .

# --------------------------------------------------
# Set Ownership (important for non-root execution)
# --------------------------------------------------
RUN chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# --------------------------------------------------
# Expose Port (HF uses 8000)
# --------------------------------------------------
EXPOSE 8000

# --------------------------------------------------
# Healthcheck (optional but recommended)
# --------------------------------------------------
HEALTHCHECK CMD curl --fail http://localhost:8000 || exit 1

# --------------------------------------------------
# Start Server
# --------------------------------------------------
# IMPORTANT:
# Replace "server.app:app" if your FastAPI app path differs

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000"]