# This file makes the protos directory a proper Python package
import os
import sys
import subprocess
import glob
import time

# Check if the protobuf and gRPC modules exist or if the proto file has been updated
proto_file = os.path.join(os.path.dirname(__file__), 'messages.proto')
pb2_file = os.path.join(os.path.dirname(__file__), 'messages_pb2.py')
grpc_file = os.path.join(os.path.dirname(__file__), 'messages_pb2_grpc.py')

# Check if proto files need to be regenerated
should_regenerate = False
if not os.path.exists(pb2_file) or not os.path.exists(grpc_file):
    should_regenerate = True
elif os.path.exists(proto_file):
    # Check if proto file is newer than generated files
    proto_mtime = os.path.getmtime(proto_file)
    pb2_mtime = os.path.getmtime(pb2_file) if os.path.exists(pb2_file) else 0
    grpc_mtime = os.path.getmtime(grpc_file) if os.path.exists(grpc_file) else 0
    if proto_mtime > pb2_mtime or proto_mtime > grpc_mtime:
        should_regenerate = True

if should_regenerate:
    print(f"Generating protobuf and gRPC files from {proto_file}")
    try:
        subprocess.check_call([
            sys.executable, 
            '-m', 
            'grpc_tools.protoc',
            '-I' + os.path.dirname(os.path.dirname(__file__)),
            '--python_out=' + os.path.dirname(os.path.dirname(__file__)),
            '--grpc_python_out=' + os.path.dirname(os.path.dirname(__file__)),
            proto_file
        ])
        print(f"Generated: {pb2_file} and {grpc_file}")
    except Exception as e:
        print(f"Error generating protobuf files: {e}")
        print(f"Current directory: {os.getcwd()}")
        print(f"Files in directory: {os.listdir(os.path.dirname(__file__))}")

# Import messages_pb2 and messages_pb2_grpc so they're available when importing the protos package
try:
    from . import messages_pb2
    from . import messages_pb2_grpc
except ImportError as e:
    print(f"Failed to import messages_pb2 or messages_pb2_grpc: {e}")
    print(f"Files in protos directory: {glob.glob(os.path.dirname(__file__) + '/*')}")
    # Wait a moment and try again - sometimes the files need a moment to be properly written
    time.sleep(1)
    try:
        from . import messages_pb2
        from . import messages_pb2_grpc
    except ImportError as e:
        print(f"Still failed to import after waiting: {e}")
        raise 