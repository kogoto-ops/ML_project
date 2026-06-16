# ── Base image ──────────────────────────────────────────────────────────────
FROM python:3.11-slim

# ── Working directory inside the container ───────────────────────────────────
WORKDIR /app

# ── Install dependencies ─────────────────────────────────────────────────────
# Copy requirements first to leverage Docker layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Copy source code ──────────────────────────────────────────────────────────
# Copy backend source and trained model artifacts
COPY backend/ ./backend/
COPY models/ ./models/

# ── Set working directory to backend folder ───────────────────────────────────
WORKDIR /app/backend

# ── Expose the port the app runs on ──────────────────────────────────────────
EXPOSE 8011

# ── Start the FastAPI server ──────────────────────────────────────────────────
CMD ["uvicorn", "backend:app", "--host", "0.0.0.0", "--port", "8011"]
