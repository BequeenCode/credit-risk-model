# Credit Risk API - FastAPI served by uvicorn
FROM python:3.11-slim

# Avoid .pyc files and unbuffered logs for clean container output.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

WORKDIR /app

# Install dependencies first to leverage Docker layer caching.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source and the trained model artifacts.
COPY src/ ./src/
COPY models/ ./models/

EXPOSE 8000

# Serve the FastAPI app.
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
