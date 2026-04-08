FROM python:3.10-slim

LABEL org.opencontainers.image.title="WorkSim-AI"
LABEL org.opencontainers.image.description="OpenEnv-compliant real-world office task simulation"

WORKDIR /app
ENV PYTHONPATH=/app

# Install dependencies first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project code
COPY . .

# Install the project itself as a package to register the 'server' script
RUN pip install --no-cache-dir .

# Use port 7860 for HF Spaces compatibility (default for Spaces)
EXPOSE 7860

# Start the server using the entry point defined in pyproject.toml
CMD ["server"]
