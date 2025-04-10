FROM ros:humble-ros-base

WORKDIR /app

# Install Python dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir protobuf requests grpcio-tools

# Copy protos directory
COPY protos /app/protos/

# Copy algorithm adaptor code
COPY algorithm_adaptor /app/algorithm_adaptor/

# Set environment variables
ENV PYTHONPATH=/app

# Set entrypoint
CMD ["python3", "algorithm_adaptor/algorithm_adaptor.py"] 