#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import threading
import grpc
import json
import time
import sys
import os
import logging
import importlib
import re
import base64
from rclpy.qos import QoSProfile
from std_msgs.msg import String
import subprocess
from rosidl_runtime_py import message_to_ordereddict, get_message_interfaces
from rosidl_runtime_py.utilities import get_message
from rclpy.topic_endpoint_info import TopicEndpointTypeEnum

# Add the parent directory to the path for importing protos
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Import the protobuf and gRPC modules
from protos import messages_pb2
from protos import messages_pb2_grpc

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('algorithm_adaptor')

class AlgorithmAdaptor(Node):
    def __init__(self):
        super().__init__('algorithm_adaptor')
        
        # Get configuration from parameters or environment variables
        self.declare_parameter('meta_simulator_url', os.environ.get('META_SIMULATOR_URL', 'localhost:50051'))
        self.declare_parameter('adaptor_id', os.environ.get('ADAPTOR_ID', 'algorithm_adaptor_1'))
        self.declare_parameter('algorithm_topic_prefix', os.environ.get('ALGORITHM_TOPIC_PREFIX', ''))
        self.declare_parameter('poll_interval', float(os.environ.get('POLL_INTERVAL', '0.5')))
        self.declare_parameter('topic_discovery_interval', float(os.environ.get('TOPIC_DISCOVERY_INTERVAL', '5.0')))
        
        # Get parameters
        self.meta_simulator_url = self.get_parameter('meta_simulator_url').value
        self.adaptor_id = self.get_parameter('adaptor_id').value
        self.algorithm_topic_prefix = self.get_parameter('algorithm_topic_prefix').value
        self.poll_interval = self.get_parameter('poll_interval').value
        self.topic_discovery_interval = self.get_parameter('topic_discovery_interval').value
        
        logger.info(f"Algorithm adaptor configuration:")
        logger.info(f"  meta_simulator_url: {self.meta_simulator_url}")
        logger.info(f"  adaptor_id: {self.adaptor_id}")
        logger.info(f"  poll_interval: {self.poll_interval}")
        logger.info(f"  topic_discovery_interval: {self.topic_discovery_interval}")
        
        # Initialize state
        self.current_time = {'seconds': 0, 'nanoseconds': 0}
        self.running = True
        self.timing_data = {}  # message_id -> start_time
        
        # Track subscriptions and publishers
        self._topic_subscriptions = {}  # topic_name -> subscription
        self._topic_publishers = {}     # topic_name -> publisher
        self._topic_msg_types = {}      # topic_name -> message_type
        
        # Setup gRPC channel
        self.channel = grpc.insecure_channel(self.meta_simulator_url)
        self.stub = messages_pb2_grpc.MetaSimulatorStub(self.channel)
        
        # Start topic discovery and create initial subscriptions
        self.discover_timer = self.create_timer(self.topic_discovery_interval, self.discover_topics)
        
        # Start polling thread
        self.poll_thread = threading.Thread(target=self.poll_meta_simulator)
        self.poll_thread.daemon = True
        self.poll_thread.start()
        
        logger.info(f"Algorithm Adaptor {self.adaptor_id} initialized")
    
    def discover_topics(self):
        """Discover ROS topics and create subscriptions/publishers as needed"""
        try:
            # Get all available topics
            topics_and_types = self.get_topic_names_and_types()
            
            # Filter for algorithm-related topics
            algorithm_topics = [(topic, types) for topic, types in topics_and_types 
                                if topic.startswith(self.algorithm_topic_prefix)]
            
            logger.info(f"Discovered algorithm topics: {algorithm_topics}")
            
            # Count publishers and subscribers for each topic
            pub_counts = {}
            sub_counts = {}
            
            # Get publisher and subscriber counts for each topic individually
            for topic_name, _ in algorithm_topics:
                # Count publishers for this topic
                pub_counts[topic_name] = self.count_publishers(topic_name)
                
                # Count subscribers for this topic
                sub_counts[topic_name] = self.count_subscribers(topic_name)
            
            # Process each algorithm topic
            for topic_name, topic_types in algorithm_topics:
                if not topic_types:
                    continue
                    
                msg_type_str = topic_types[0]  # Use the first type
                
                if topic_name not in self._topic_subscriptions and topic_name not in self._topic_publishers:
                    # Determine if this is an input or output topic based on publisher/subscriber counts
                    pub_count = pub_counts.get(topic_name, 0)
                    sub_count = sub_counts.get(topic_name, 0)
                    
                    logger.info(f"Topic {topic_name}: {pub_count} publishers, {sub_count} subscribers")
                    
                    if pub_count > sub_count:
                        # This is likely an output topic, create a subscription
                        self.create_topic_subscription(topic_name, msg_type_str)
                    else:
                        # This is likely an input topic, create a publisher
                        self.create_topic_publisher(topic_name, msg_type_str)
                
        except Exception as e:
            logger.error(f"Error discovering topics: {str(e)}")
    
    def get_message_type(self, type_str):
        """Get a message type from its string representation"""
        try:
            # Parse the message type string (e.g., 'std_msgs/msg/String')
            parts = type_str.split('/')
            if len(parts) < 3:
                return None
                
            pkg_name = parts[0]
            msg_name = parts[-1]
            
            # Import the message module
            mod = importlib.import_module(f"{pkg_name}.msg")
            
            # Get the message class
            msg_class = getattr(mod, msg_name)
            return msg_class
            
        except (ImportError, AttributeError) as e:
            logger.error(f"Failed to import message type {type_str}: {e}")
            return None
    
    def create_topic_subscription(self, topic_name, msg_type_str):
        """Create a subscription for the given topic"""
        try:
            msg_type = self.get_message_type(msg_type_str)
            if not msg_type:
                logger.warning(f"Could not get message type for {topic_name}: {msg_type_str}")
                # Fall back to String if we can't get the type
                msg_type = String
            
            logger.info(f"Creating subscription for {topic_name} with type {msg_type.__name__}")
            
            # Create the subscription
            self._topic_subscriptions[topic_name] = self.create_subscription(
                msg_type,
                topic_name,
                lambda msg, tn=topic_name: self.handle_algorithm_output(msg, tn),
                10
            )
            self._topic_msg_types[topic_name] = msg_type
            
        except Exception as e:
            logger.error(f"Failed to create subscription for {topic_name}: {e}")
    
    def create_topic_publisher(self, topic_name, msg_type_str):
        """Create a publisher for the given topic"""
        try:
            msg_type = self.get_message_type(msg_type_str)
            if not msg_type:
                logger.warning(f"Could not get message type for {topic_name}: {msg_type_str}")
                # Fall back to String if we can't get the type
                msg_type = String
            
            logger.info(f"Creating publisher for {topic_name} with type {msg_type.__name__}")
            
            # Create the publisher
            self._topic_publishers[topic_name] = self.create_publisher(
                msg_type,
                topic_name,
                10
            )
            self._topic_msg_types[topic_name] = msg_type
            
        except Exception as e:
            logger.error(f"Failed to create publisher for {topic_name}: {e}")
    
    def poll_meta_simulator(self):
        """Poll meta simulator for messages"""
        while self.running:
            try:
                # Get topics to subscribe to
                topics = list(self._topic_publishers.keys())
                
                # Create request
                request = messages_pb2.PollRequest(
                    adaptor_id=self.adaptor_id,
                    topics=topics,
                    last_update_time=messages_pb2.TimeStamp(
                        seconds=self.current_time['seconds'],
                        nanoseconds=self.current_time['nanoseconds']
                    )
                )
                
                # Call gRPC Poll method
                response = self.stub.Poll(request)
                
                # Update current time
                self.current_time = {
                    'seconds': response.state.current_time.seconds,
                    'nanoseconds': response.state.current_time.nanoseconds
                }
                
                # Process messages
                for message in response.messages:
                    self.process_message(message)
                
            except Exception as e:
                logger.error(f"Error polling meta simulator: {str(e)}")
                
            # Wait for next poll
            time.sleep(self.poll_interval)
    
    def process_message(self, message):
        """Process incoming message from meta simulator"""
        try:
            # With the binary serialization approach, we don't need to check for specific message types
            # Just forward all messages from simulator adaptor to algorithm
            self.forward_state_to_algorithm(message)
            
        except Exception as e:
            logger.error(f"Error processing message: {str(e)}")
    
    def forward_state_to_algorithm(self, message):
        """Forward simulator state to algorithm"""
        try:
            # Extract topic from message
            topic = message.topic
            
            # Check if we have a publisher for this topic
            if topic not in self._topic_publishers:
                logger.warning(f"No publisher for topic: {topic}")
                return
            
            # Get message type for this topic
            msg_type = self._topic_msg_types.get(topic)
            if not msg_type:
                logger.warning(f"No message type for topic: {topic}")
                return
            
            try:
                # Deserialize binary data directly
                import pickle
                if message.data:
                    ros_msg = pickle.loads(message.data)
                    
                    # Publish to ROS topic
                    self._topic_publishers[topic].publish(ros_msg)
                    logger.info(f"Published binary message to topic: {topic}")
                else:
                    logger.warning(f"Received empty message data for topic: {topic}")
            except Exception as e:
                logger.error(f"Error deserializing message for {topic}: {e}")
                return
            
        except Exception as e:
            logger.error(f"Error forwarding state to algorithm: {str(e)}")
    
    def handle_algorithm_output(self, msg, topic_name):
        """Handle algorithm output from ROS topic"""
        try:
            # Serialize ROS message to binary
            import pickle
            serialized_data = pickle.dumps(msg)
            
            # Create protobuf message with binary data
            message = messages_pb2.Message(
                source_id=self.adaptor_id,
                topic=topic_name,
                message_type=msg.__class__.__module__ + '.' + msg.__class__.__name__,
                data=serialized_data
            )
            
            # Send the message to the meta simulator using SendMessage method
            try:
                response = self.stub.SendMessage(message)
                if response.success:
                    logger.info(f"Sent binary message to meta simulator for topic: {topic_name}")
                else:
                    logger.warning(f"Failed to send message: {response.message}")
            except Exception as e:
                logger.error(f"Error sending message to meta simulator: {e}")
            
            logger.info(f"Processed algorithm output from topic: {topic_name}")
            
        except Exception as e:
            logger.error(f"Error processing algorithm output: {str(e)}")
    
    def destroy_node(self):
        """Clean up resources"""
        self.running = False
        if self.poll_thread.is_alive():
            self.poll_thread.join(timeout=1.0)
        self.channel.close()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    adaptor = AlgorithmAdaptor()
    
    try:
        rclpy.spin(adaptor)
    except KeyboardInterrupt:
        pass
    finally:
        adaptor.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main() 