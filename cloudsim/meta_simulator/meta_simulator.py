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

# Embedded network simulator adaptor code
class SimpleNetworkSimulator:
    def __init__(self):
        self.base_latency_ns = 1000000  # 1ms base latency
        self.bandwidth_bps = 1000000000  # 1Gbps
    
    def calculate_latency(self, source, destination, packet_size_bytes):
        # Simple latency model: base_latency + (packet_size / bandwidth)
        transmission_latency_ns = (packet_size_bytes * 8 * 1000000000) // self.bandwidth_bps
        return self.base_latency_ns + transmission_latency_ns

class NS3NetworkSimulator:
    def __init__(self):
        # This would initialize the NS3 simulator, potentially via a subprocess or API
        logging.getLogger('meta_simulator').info("Initializing NS3 Network Simulator")
        self.ns3_process = None
    
    def calculate_latency(self, source, destination, packet_size_bytes):
        # This would call into the NS3 simulator to calculate latency
        # Placeholder implementation
        logging.getLogger('meta_simulator').info(f"NS3 calculating latency from {source} to {destination} for {packet_size_bytes} bytes")
        # Call into NS3 API or use IPC to get the latency
        return 5000000  # 5ms in nanoseconds

class NetworkSimulatorAdaptor:
    def __init__(self):
        # Get configuration from environment variables
        self.adaptor_id = os.environ.get('ADAPTOR_ID', 'network_simulator_adaptor')
        self.network_simulator_type = os.environ.get('NETWORK_SIMULATOR_TYPE', 'simple')  # 'simple', 'ns3', etc.
        
        # Initialize network simulator based on type
        self.init_network_simulator()
        
        logging.getLogger('meta_simulator').info(f"Network Simulator Adaptor {self.adaptor_id} initialized with {self.network_simulator_type} simulator")
    
    def init_network_simulator(self):
        if self.network_simulator_type == 'simple':
            # Simple constant latency simulator
            self.network_simulator = SimpleNetworkSimulator()
        elif self.network_simulator_type == 'ns3':
            # NS3 simulator (placeholder)
            self.network_simulator = NS3NetworkSimulator()
        else:
            logging.getLogger('meta_simulator').error(f"Unknown network simulator type: {self.network_simulator_type}")
            raise ValueError(f"Unknown network simulator type: {self.network_simulator_type}")
    
    def calculate_delay(self, message):
        """
        Calculate network delay for a message in nanoseconds
        
        Args:
            message (dict): The message to calculate delay for
            
        Returns:
            int: Delay in nanoseconds
        """
        # Extract relevant information from message
        source = message.get('source_id', '')
        destination = message.get('destination_id', '')
        
        # Calculate message size
        if 'binary_data' in message:
            message_size = len(message['binary_data'])
        else:
            # Estimate size if binary data not available
            message_size = 1024  # Default 1KB
        
        # Calculate latency using network simulator
        latency_ns = self.network_simulator.calculate_latency(source, destination, message_size)
        
        logging.getLogger('meta_simulator').info(f"Calculated network delay: {latency_ns}ns from {source} to {destination}")
        return latency_ns

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
        # Initialize network simulator adaptor
        self.network_simulator = NetworkSimulatorAdaptor()
        logger.info("Meta Simulator initialized with Network Simulator Adaptor")
    
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
                    'binary_data': request.data,
                    'destination_id': request.destination_id if hasattr(request, 'destination_id') else ''
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
            # Find the next message timestamp if any
            next_message_time = self._get_next_message_time()
            
            # If there's a message scheduled before the requested time,
            # advance only to that message's time
            if next_message_time and self._compare_times(next_message_time, new_time) < 0:
                success = self.advance_time(next_message_time)
                message = f"Time advanced to next message time: {next_message_time['seconds']}.{next_message_time['nanoseconds']}"
            else:
                success = self.advance_time(new_time)
                message = "Time advanced as requested"
            
            # Create response
            response = messages_pb2.StatusResponse(
                success=success,
                message=message,
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
    
    def _compare_times(self, time1, time2):
        """Compare two timestamps, return -1 if time1 < time2, 0 if equal, 1 if time1 > time2"""
        if time1['seconds'] < time2['seconds']:
            return -1
        elif time1['seconds'] > time2['seconds']:
            return 1
        else:
            if time1['nanoseconds'] < time2['nanoseconds']:
                return -1
            elif time1['nanoseconds'] > time2['nanoseconds']:
                return 1
            else:
                return 0
    
    def _get_next_message_time(self):
        """Get the timestamp of the next scheduled message"""
        with self.lock:
            if not self.message_queue:
                return None
                
            # Find the earliest timestamp
            earliest_ts = min(self.message_queue.keys())
            ts_secs, ts_nsecs = map(int, earliest_ts.split('.'))
            
            return {'seconds': ts_secs, 'nanoseconds': ts_nsecs}
    
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
    
    def add_message(self, message):
        """Add a message to the queue with network delay"""
        with self.lock:
            # Calculate network delay using the network simulator
            delay_ns = self.network_simulator.calculate_delay(message)
            
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
            
            logger.info(f"Scheduled message {message_id} for delivery at {timestamp_key} with delay {delay_ns}ns")
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