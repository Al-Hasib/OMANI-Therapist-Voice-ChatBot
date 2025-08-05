FROM python:3.10-slim

WORKDIR /app

# Install system dependencies including supervisor
RUN apt-get update && apt-get install -y \
    supervisor \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Create log directory for supervisor
RUN mkdir -p /var/log/supervisor

COPY . .
RUN pip install --no-cache-dir -r requirements.txt

# Copy supervisor configuration
COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf

# Expose both ports
EXPOSE 5001 7860

# Start supervisor
CMD ["/usr/bin/supervisord", "-c", "/etc/supervisor/conf.d/supervisord.conf"]