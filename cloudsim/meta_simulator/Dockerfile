FROM python:3.10-slim

WORKDIR /app

# Install dependencies
RUN pip install --no-cache-dir flask protobuf requests grpcio-tools

# Copy protos directory
COPY protos /app/protos/

# Generate Python protobuf files
RUN pip install --no-cache-dir protobuf grpcio-tools && \
    cd /app && \
    python -m grpc_tools.protoc -I. --python_out=. protos/messages.proto && \
    # Ensure we touch the __init__.py to create it if needed
    touch /app/protos/__init__.py

# Copy meta simulator code
COPY meta_simulator /app/meta_simulator/

# Set environment variables
ENV PYTHONPATH=/app

# Expose port
EXPOSE 5000

# Run the application
CMD ["python", "meta_simulator/meta_simulator.py"] 