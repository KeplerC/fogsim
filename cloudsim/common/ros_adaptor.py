#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import importlib
import sys
import os
import logging
import time
import pickle
import base64

# Add the parent directory to the path for importing the base adaptor
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.adaptor_base import AdaptorBase

class ROSAdaptor(AdaptorBase, Node):
    """ROS-specific adaptor that bridges between ROS topics and meta simulator."""
    
    def __init__(self, 
                node_name='ros_adaptor',
                adaptor_id=None,
                meta_simulator_url=None,
                poll_interval=0.5,
                topic_discovery_interval=5.0,
                topic_prefix=''):
        """
        Initialize the ROS adaptor.
        
        Args:
            node_name: Name of the ROS node
            adaptor_id: Unique identifier for this adaptor
            meta_simulator_url: URL of the meta simulator
            poll_interval: Interval in seconds for polling the meta simulator
            topic_discovery_interval: Interval in seconds for discovering ROS topics
            topic_prefix: Prefix for ROS topics to consider
        """
        # Initialize ROS node
        rclpy.init()
        Node.__init__(self, node_name)
        
        # Initialize adaptor base
        AdaptorBase.__init__(
            self, 
            adaptor_id=adaptor_id, 
            meta_simulator_url=meta_simulator_url,
            poll_interval=poll_interval
        )
        
        # ROS-specific parameters
        self.topic_discovery_interval = float(topic_discovery_interval or 
                                            os.environ.get('TOPIC_DISCOVERY_INTERVAL', '5.0'))
        self.topic_prefix = topic_prefix or os.environ.get('TOPIC_PREFIX', '')
        
        # Track ROS subscriptions and publishers
        self._ros_subscriptions = {}  # topic_name -> subscription
        self._ros_publishers = {}     # topic_name -> publisher
        self._topic_msg_types = {}    # topic_name -> message_type
        
        # Start topic discovery
        self.discover_timer = self.create_timer(self.topic_discovery_interval, self.discover_topics)
        
        self.logger.info(f"ROS Adaptor {self.adaptor_id} initialized")
    
    def discover_topics(self):
        """Discover ROS topics and create subscriptions/publishers as needed."""
        try:
            # Get all available topics
            topics_and_types = self.get_topic_names_and_types()
            
            # Filter for topics with the specified prefix
            filtered_topics = [(topic, types) for topic, types in topics_and_types 
                              if topic.startswith(self.topic_prefix)]
            
            self.logger.info(f"Discovered topics with prefix '{self.topic_prefix}': {filtered_topics}")
            
            # Count publishers and subscribers for each topic
            pub_counts = {}
            sub_counts = {}
            
            # Get publisher and subscriber counts for each topic individually
            for topic_name, _ in filtered_topics:
                # Count publishers for this topic
                pub_counts[topic_name] = self.count_publishers(topic_name)
                
                # Count subscribers for this topic
                sub_counts[topic_name] = self.count_subscribers(topic_name)
            
            # Process each filtered topic
            for topic_name, topic_types in filtered_topics:
                if not topic_types:
                    continue
                    
                msg_type_str = topic_types[0]  # Use the first type
                
                # If we haven't handled this topic yet
                if topic_name not in self._ros_subscriptions and topic_name not in self._ros_publishers:
                    # Determine if this is an input or output topic based on publisher/subscriber counts
                    pub_count = pub_counts.get(topic_name, 0)
                    sub_count = sub_counts.get(topic_name, 0)
                    
                    self.logger.info(f"Topic {topic_name}: {pub_count} publishers, {sub_count} subscribers")
                    
                    if pub_count > sub_count:
                        # This is likely an output topic, create a subscription
                        self.create_ros_subscription(topic_name, msg_type_str)
                    else:
                        # This is likely an input topic, create a publisher
                        self.create_ros_publisher(topic_name, msg_type_str)
        
        except Exception as e:
            self.logger.error(f"Error discovering topics: {str(e)}")
    
    def get_message_type(self, type_str):
        """Get a message type from its string representation."""
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
            self.logger.error(f"Failed to import message type {type_str}: {e}")
            return None
    
    def create_ros_subscription(self, topic_name, msg_type_str):
        """Create a ROS subscription for the given topic."""
        try:
            msg_type = self.get_message_type(msg_type_str)
            if not msg_type:
                self.logger.warning(f"Could not get message type for {topic_name}: {msg_type_str}")
                from std_msgs.msg import String
                msg_type = String
            
            self.logger.info(f"Creating ROS subscription for {topic_name} with type {msg_type.__name__}")
            
            # Create the ROS subscription
            self._ros_subscriptions[topic_name] = self.create_subscription(
                msg_type,
                topic_name,
                lambda msg, tn=topic_name: self.handle_ros_message(msg, tn),
                10
            )
            self._topic_msg_types[topic_name] = msg_type
            
        except Exception as e:
            self.logger.error(f"Failed to create ROS subscription for {topic_name}: {e}")
    
    def create_ros_publisher(self, topic_name, msg_type_str):
        """Create a ROS publisher for the given topic."""
        try:
            msg_type = self.get_message_type(msg_type_str)
            if not msg_type:
                self.logger.warning(f"Could not get message type for {topic_name}: {msg_type_str}")
                from std_msgs.msg import String
                msg_type = String
            
            self.logger.info(f"Creating ROS publisher for {topic_name} with type {msg_type.__name__}")
            
            # Create the ROS publisher
            self._ros_publishers[topic_name] = self.create_publisher(
                msg_type,
                topic_name,
                10
            )
            self._topic_msg_types[topic_name] = msg_type
            
            # Subscribe to this topic in the meta simulator for receiving messages
            self.subscribe(topic_name, self.handle_meta_simulator_message)
            
        except Exception as e:
            self.logger.error(f"Failed to create ROS publisher for {topic_name}: {e}")
    
    def handle_ros_message(self, msg, topic_name):
        """Handle a message received from ROS and forward to meta simulator."""
        self.logger.info(f"Received ROS message on topic: {topic_name}")
        
        # Start timing for latency measurement
        start_time = time.time_ns()
        
        # Publish to meta simulator
        self.publish(topic_name, msg)
    
    def handle_meta_simulator_message(self, data, topic_name):
        """Handle a message received from meta simulator and forward to ROS."""
        self.logger.info(f"Received meta simulator message for topic: {topic_name}")
        
        if topic_name in self._ros_publishers:
            publisher = self._ros_publishers[topic_name]
            publisher.publish(data)
            self.logger.info(f"Published message to ROS topic: {topic_name}")
        else:
            self.logger.warning(f"No ROS publisher found for topic: {topic_name}")
    
    def count_publishers(self, topic_name):
        """Count the number of publishers for a ROS topic."""
        from rclpy.topic_endpoint_info import TopicEndpointTypeEnum
        publishers_info = self.get_publishers_info_by_topic(topic_name)
        return len(publishers_info)
    
    def count_subscribers(self, topic_name):
        """Count the number of subscribers for a ROS topic."""
        from rclpy.topic_endpoint_info import TopicEndpointTypeEnum
        subscribers_info = self.get_subscriptions_info_by_topic(topic_name)
        return len(subscribers_info)
    
    def destroy_node(self):
        """Clean up resources used by the node."""
        self.stop()
        Node.destroy_node(self)
        rclpy.shutdown()

def main(args=None):
    adaptor = ROSAdaptor()
    
    try:
        adaptor.start()
        rclpy.spin(adaptor)
    except KeyboardInterrupt:
        pass
    finally:
        adaptor.destroy_node()

if __name__ == '__main__':
    main() 