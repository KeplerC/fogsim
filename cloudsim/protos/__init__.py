# This file makes the protos directory a proper Python package
import os
import sys
import subprocess
import glob

# Check if the protobuf module exists, if not, generate it
proto_file = os.path.join(os.path.dirname(__file__), 'messages.proto')
pb2_file = os.path.join(os.path.dirname(__file__), 'messages_pb2.py')

if not os.path.exists(pb2_file):
    print(f"Generating protobuf files from {proto_file}")
    try:
        subprocess.check_call([
            sys.executable, 
            '-m', 
            'grpc_tools.protoc',
            '-I' + os.path.dirname(os.path.dirname(__file__)),
            '--python_out=' + os.path.dirname(os.path.dirname(__file__)),
            proto_file
        ])
        print(f"Generated: {pb2_file}")
    except Exception as e:
        print(f"Error generating protobuf files: {e}")
        print(f"Current directory: {os.getcwd()}")
        print(f"Files in directory: {os.listdir(os.path.dirname(__file__))}")

# Import messages_pb2 so it's available when importing the protos package
try:
    from . import messages_pb2
except ImportError as e:
    print(f"Failed to import messages_pb2: {e}")
    print(f"Files in protos directory: {glob.glob(os.path.dirname(__file__) + '/*')}")
    raise 