#!/usr/bin/env python3

import time
import json
import uuid
import threading
import concurrent.futures
import logging
import sys
import os
import grpc
from concurrent import futures

# Add the parent directory to the path for importing protos
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Import the protobuf and gRPC modules
from protos import messages_pb2
from protos import messages_pb2_grpc

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('meta_simulator')

class MetaSimulator(messages_pb2_grpc.MetaSimulatorServicer):
    def __init__(self):
        self.current_time = {'seconds': 0, 'nanoseconds': 0}
        self.message_queue = {}  # Timestamp -> [messages]
        self.message_store = {}  # message_id -> message
        self.adaptors = set()
        self.lock = threading.RLock()
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=10)
        logger.info("Meta Simulator initialized")
    
    def Poll(self, request, context):
        """gRPC Poll implementation - returns messages for requested topics"""
        adaptor_id = request.adaptor_id
        topics = list(request.topics)
        
        # Register adaptor if not already registered
        if adaptor_id and adaptor_id not in self.adaptors:
            with self.lock:
                self.adaptors.add(adaptor_id)
                logger.info(f"Registered adaptor via Poll: {adaptor_id}")
        
        # Check if this request is sending a message (empty topics list with algorithm_response or simulator_state)
        is_sending_message = False
        if hasattr(context, 'invocation_metadata'):
            # Extract metadata to see if this is a message-sending request
            metadata = dict(context.invocation_metadata())
            if 'send_message' in metadata and metadata['send_message'] == 'true':
                is_sending_message = True
                logger.info("Detected message-sending request")
        
        # Get messages for time and topics
        messages = []
        with self.lock:
            # Parse last_update_time if provided
            last_update_time = None
            if request.HasField('last_update_time'):
                last_update_time = {
                    'seconds': request.last_update_time.seconds,
                    'nanoseconds': request.last_update_time.nanoseconds
                }
            
            # Get state
            state = self.get_state()
            state_proto = messages_pb2.SimulatorState(
                current_time=messages_pb2.TimeStamp(
                    seconds=state['current_time']['seconds'],
                    nanoseconds=state['current_time']['nanoseconds']
                ),
                pending_messages=state['pending_messages'],
                registered_adaptors=state['registered_adaptors']
            )
            
            # Get messages for topics
            message_ids = self.get_messages_for_time(self.current_time)
            
            for msg_id in message_ids:
                if msg_id in self.message_store:
                    message = self.message_store[msg_id]
                    # Check if message matches requested topics
                    if self._message_matches_topics(message, topics):
                        # Convert message to protobuf
                        proto_msg = self._convert_to_proto_message(message, msg_id)
                        if proto_msg:
                            messages.append(proto_msg)
            
            # Handle adding new messages from the request context
            # This supports the "sending" part of our bidirectional communication
            if not topics and hasattr(context, 'additional_context'):
                # This would be set by the client when sending a message
                if 'message' in context.additional_context:
                    message = context.additional_context['message']
                    self.add_message(message)
                    logger.info(f"Added message from Poll request, source: {message.get('source_id', 'unknown')}")
            
            # Create response
            response = messages_pb2.PollResponse(
                success=True,
                state=state_proto,
                messages=messages
            )
            
            return response
    
    def SendMessage(self, request, context):
        """gRPC SendMessage implementation - store messages sent by adaptors"""
        try:
            with self.lock:
                # Extract the message from the request
                message = {
                    'source_id': request.source_id,
                    'topic': request.topic,
                    'message_type': request.message_type,
                    'binary_data': request.data
                }
                
                # Store the message
                message_id = self.add_message(message)
                logger.info(f"Added binary message from SendMessage, ID: {message_id}, source: {message['source_id']}, topic: {message['topic']}")
                
                # Return success response
                return messages_pb2.StatusResponse(
                    success=True,
                    message=f"Message stored with ID: {message_id}",
                    current_time=messages_pb2.TimeStamp(
                        seconds=self.current_time['seconds'],
                        nanoseconds=self.current_time['nanoseconds']
                    )
                )
        except Exception as e:
            logger.error(f"Error in SendMessage: {e}")
            return messages_pb2.StatusResponse(
                success=False,
                message=f"Error: {str(e)}",
                current_time=messages_pb2.TimeStamp(
                    seconds=self.current_time['seconds'],
                    nanoseconds=self.current_time['nanoseconds']
                )
            )
    
    def AdvanceTime(self, request, context):
        """gRPC AdvanceTime implementation - advances simulation time"""
        new_time = {
            'seconds': request.time.seconds,
            'nanoseconds': request.time.nanoseconds
        }
        
        with self.lock:
            success = self.advance_time(new_time)
            
            # Create response
            response = messages_pb2.StatusResponse(
                success=success,
                message="Time advanced" if success else "Failed to advance time",
                current_time=messages_pb2.TimeStamp(
                    seconds=self.current_time['seconds'],
                    nanoseconds=self.current_time['nanoseconds']
                )
            )
            
            return response
    
    def GetState(self, request, context):
        """gRPC GetState implementation - returns current simulator state"""
        with self.lock:
            state = self.get_state()
            
            # Create response
            state_proto = messages_pb2.SimulatorState(
                current_time=messages_pb2.TimeStamp(
                    seconds=state['current_time']['seconds'],
                    nanoseconds=state['current_time']['nanoseconds']
                ),
                pending_messages=state['pending_messages'],
                registered_adaptors=state['registered_adaptors']
            )
            
            response = messages_pb2.StateResponse(
                success=True,
                state=state_proto
            )
            
            return response
    
    def _message_matches_topics(self, message, topics):
        """Check if message matches any of the requested topics"""
        # If no topics requested, no match
        if not topics:
            return False
            
        # If wildcard topic requested, match all
        if '*' in topics:
            return True
            
        # Check if message has a topic that matches
        if 'topic' in message and message['topic'] in topics:
            return True
                
        return False
    
    def _convert_to_proto_message(self, message, message_id):
        """Convert a message dict to a protobuf Message"""
        try:
            proto_msg = messages_pb2.Message(
                message_id=message_id,
                source_id=message.get('source_id', ''),
                topic=message.get('topic', ''),
                message_type=message.get('message_type', ''),
                data=message.get('binary_data', b'')
            )
            
            return proto_msg
        except Exception as e:
            logger.error(f"Error converting message to proto: {e}")
            return None
    
    def advance_time(self, new_time):
        """Advance simulation time"""
        with self.lock:
            if (new_time['seconds'] > self.current_time['seconds'] or 
                (new_time['seconds'] == self.current_time['seconds'] and 
                 new_time['nanoseconds'] > self.current_time['nanoseconds'])):
                self.current_time = new_time
                logger.info(f"Advanced time to: {self.current_time}")
                return True
            return False
    
    def get_messages_for_time(self, target_time):
        """Get messages ready for delivery at target_time"""
        with self.lock:
            messages = []
            for timestamp, msgs in list(self.message_queue.items()):
                ts_secs, ts_nsecs = map(int, timestamp.split('.'))
                if (ts_secs < target_time['seconds'] or 
                    (ts_secs == target_time['seconds'] and ts_nsecs <= target_time['nanoseconds'])):
                    messages.extend(msgs)
                    del self.message_queue[timestamp]
            return messages
    
    def add_message(self, message, delay_ns=0):
        """Add a message to the queue with optional delay"""
        with self.lock:
            # Calculate delivery time
            delivery_time_secs = self.current_time['seconds']
            delivery_time_nsecs = self.current_time['nanoseconds'] + delay_ns
            
            # Handle nanosecond overflow
            if delivery_time_nsecs >= 1000000000:
                delivery_time_secs += delivery_time_nsecs // 1000000000
                delivery_time_nsecs = delivery_time_nsecs % 1000000000
            
            timestamp_key = f"{delivery_time_secs}.{delivery_time_nsecs}"
            
            if timestamp_key not in self.message_queue:
                self.message_queue[timestamp_key] = []
            
            # Store the message
            message_id = str(uuid.uuid4())
            message['message_id'] = message_id
            self.message_store[message_id] = message
            self.message_queue[timestamp_key].append(message_id)
            
            logger.info(f"Scheduled message {message_id} for delivery at {timestamp_key}")
            return message_id
    
    def get_state(self):
        """Get current simulator state"""
        with self.lock:
            state = {
                'current_time': self.current_time,
                'pending_messages': len(sum(self.message_queue.values(), [])),
                'registered_adaptors': list(self.adaptors)
            }
            return state

def serve():
    """Start gRPC server"""
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    messages_pb2_grpc.add_MetaSimulatorServicer_to_server(
        MetaSimulator(), server
    )
    server_port = os.environ.get('GRPC_PORT', '50051')
    server.add_insecure_port(f'[::]:{server_port}')
    server.start()
    logger.info(f"Meta Simulator gRPC server started on port {server_port}")
    
    try:
        # Keep server running until Ctrl+C
        server.wait_for_termination()
    except KeyboardInterrupt:
        logger.info("Shutting down Meta Simulator gRPC server")
        server.stop(0)

if __name__ == '__main__':
    serve()