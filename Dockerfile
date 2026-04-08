FROM python:3.10-slim

LABEL org.opencontainers.image.title="WorkSim-AI"
LABEL org.opencontainers.image.description="OpenEnv-compliant real-world office task simulation"

WORKDIR /app
ENV PYTHONPATH=/app

# Install dependencies
COPY pyproject.toml requirements.txt uv.lock ./
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir .

# Copy everything else
COPY . .

# Use port 7860 for HF Spaces compatibility
EXPOSE 7860

# Start the server using the entry point defined in pyproject.toml
CMD ["server"]
