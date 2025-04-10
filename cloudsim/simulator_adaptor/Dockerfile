FROM ros:humble-ros-base

WORKDIR /app

# Install Python dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir protobuf requests grpcio-tools

# Copy protos directory
COPY protos /app/protos/

# Copy simulator adaptor code
COPY simulator_adaptor /app/simulator_adaptor/

# Set environment variables
ENV PYTHONPATH=/app

# Set entrypoint to ros2 run
CMD ["python3", "simulator_adaptor/simulator_adaptor.py"] 