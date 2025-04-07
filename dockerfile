# Use Python 3.9 slim image as base
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies including CIFS utils for mounting
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    cifs-utils \
    && rm -rf /var/lib/apt/lists/*

# Create mount point
RUN mkdir -p /models

# Copy and install requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install blobfuse

# Copy application file
COPY . .

# Clean up
RUN apt-get purge -y --auto-remove build-essential

# Configure app
ENV PORT=80
ENV MODEL_PATH=/models/moviesModel.pkl

# Expose port 80
EXPOSE 80

# Run app
CMD ["python", "app.py"]
