# ── Base image ──────────────────────────────────────────────────────────────
FROM python:3.11-slim

# ── Working directory inside the container ───────────────────────────────────
WORKDIR /app

# ── Install dependencies ─────────────────────────────────────────────────────
# Copy requirements first to leverage Docker layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Copy source code ──────────────────────────────────────────────────────────
COPY frontend/ ./frontend/

# ── Set working directory to frontend folder ──────────────────────────────────
WORKDIR /app/frontend

# ── Expose the port Streamlit runs on ────────────────────────────────────────
EXPOSE 8501

# ── Start the Streamlit app ───────────────────────────────────────────────────
CMD ["streamlit", "run", "frontend.py", "--server.port=8501", "--server.address=0.0.0.0"]
