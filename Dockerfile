
# Base Image
FROM python:3.10-slim

# Prevent Python buffering issues
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# System Dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Working Directory
WORKDIR /app

# Install Python Dependencies
COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy Project Files
COPY . .

# Expose Port (HF expects this)
EXPOSE 8000

# Health Check (optional but useful)
HEALTHCHECK CMD curl --fail http://localhost:8000/ || exit 1

# Start Server
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000"]