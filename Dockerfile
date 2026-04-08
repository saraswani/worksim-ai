FROM python:3.10-slim

LABEL org.opencontainers.image.title="WorkSim-AI"
LABEL org.opencontainers.image.description="OpenEnv-compliant real-world office task simulation"

WORKDIR /app
ENV PYTHONPATH=/app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
# Use port 7860 for HF Spaces compatibility
EXPOSE 7860

# Run the FastAPI server
CMD ["python", "app.py"]
