FROM python:3.12.7-slim

WORKDIR /app

# Install dependencies
RUN pip install --no-cache-dir flask prometheus-client psutil pandas scikit-learn joblib

# Copy semua file
COPY . .

# Expose port
EXPOSE 8000

# Run exporter
CMD ["python", "prometheus_exporter.py"]