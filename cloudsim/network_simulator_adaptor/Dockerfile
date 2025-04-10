FROM python:3.10-slim

WORKDIR /app

# Install dependencies
RUN pip install --no-cache-dir protobuf grpcio requests grpcio-tools

# Copy protos directory
COPY protos /app/protos/

# Copy network simulator adaptor code
COPY network_simulator_adaptor /app/network_simulator_adaptor/

# Set environment variables
ENV PYTHONPATH=/app
ENV META_SIMULATOR_URL=http://meta_simulator:5000
ENV ADAPTOR_ID=network_simulator_adaptor
ENV POLL_INTERVAL=0.1
ENV NETWORK_SIMULATOR_TYPE=simple

# Run the application
CMD ["python", "network_simulator_adaptor/network_simulator_adaptor.py"] 