FROM python:3.10-slim

LABEL org.opencontainers.image.title="WorkSim-AI"
LABEL org.opencontainers.image.description="OpenEnv-compliant real-world office task simulation"

WORKDIR /app
ENV PYTHONPATH=/app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
# Default: run the demo (no API key required)
# Override CMD to run the baseline: docker run -e HF_TOKEN=... worksim python run_baseline.py
CMD ["python", "run_env.py"]
