# Use Python 3.9 slim image as base
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install build essentials for Python packages
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy and install requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application file
COPY app.py .

# Clean up
RUN apt-get purge -y --auto-remove build-essential

# Set environment variables for Azure Web App
ENV PORT=80
ENV RECOMMENDATION_MODEL_PATH=/home/model/moviesModel.pkl

# Create the mount directory that Azure Web App will use
RUN mkdir -p /home/model

# Expose port 80
EXPOSE 80

# Run app
CMD ["python", "app.py"]
