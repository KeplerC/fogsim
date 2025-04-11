#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import time
from std_msgs.msg import String, Float32
from geometry_msgs.msg import Twist, PoseStamped
import threading
import os

class TopicSetupNode(Node):
    def __init__(self):
        # Determine if this is simulator or algorithm based on ROS_DOMAIN_ID
        domain_id = int(os.environ.get('ROS_DOMAIN_ID', '1'))
        
        # Use different node names based on domain
        if domain_id == 1:
            super().__init__('simulator_test_node')
            self.is_simulator = True
        else:
            super().__init__('algorithm_test_node')
            self.is_simulator = False
            
        self.declare_parameter('pub_frequency', 1.0)  # Hz
        
        # Get parameters
        pub_frequency = self.get_parameter('pub_frequency').value
        
        # Set up topics for both simulator and algorithm nodes
        self.setup_topics()
        
        # Create timer for publishing simulated data
        self.pub_timer = self.create_timer(1.0 / pub_frequency, self.publish_data)
        
        self.get_logger().info(f"{'Simulator' if self.is_simulator else 'Algorithm'} test node setup complete")
    
    def setup_topics(self):
        """Set up topics for both simulator and algorithm"""
        if self.is_simulator:
            # Simulator publishes environment state and subscribes to actions
            self.env_state_pub = self.create_publisher(
                PoseStamped, 
                '/env_state',
                10
            )
            self.action_sub = self.create_subscription(
                Twist,
                '/action',
                self.action_callback,
                10
            )
            # Store received messages
            self.last_action = Twist()
        else:
            # Algorithm subscribes to environment state and publishes actions
            self.action_pub = self.create_publisher(
                Twist,
                '/action',
                10
            )
            self.env_state_sub = self.create_subscription(
                PoseStamped,
                '/env_state',
                self.env_state_callback,
                10
            )
            # Store received messages
            self.last_env_state = PoseStamped()
    
    def action_callback(self, msg):
        """Handle actions sent to simulator"""
        self.get_logger().info(f'Simulator received action: {msg}')
        self.last_action = msg
    
    def env_state_callback(self, msg):
        """Handle environment state sent to algorithm"""
        self.get_logger().info(f'Algorithm received environment state: {msg}')
        self.last_env_state = msg
        
        # Generate an algorithm output based on the input
        action_msg = Twist()
        # Simple algorithm: move toward origin
        action_msg.linear.x = -0.1 * msg.pose.position.x
        action_msg.linear.y = -0.1 * msg.pose.position.y
        action_msg.angular.z = 0.1
        self.action_pub.publish(action_msg)
    
    def publish_data(self):
        """Publish data based on node type"""
        if self.is_simulator:
            self.publish_env_state()
        # Algorithm does not need to publish on timer as it reacts to env_state messages
    
    def publish_env_state(self):
        """Publish environment state data"""
        # Publish simulated environment state
        env_state_msg = PoseStamped()
        env_state_msg.header.stamp = self.get_clock().now().to_msg()
        env_state_msg.header.frame_id = 'map'
        env_state_msg.pose.position.x = 1.0
        env_state_msg.pose.position.y = 2.0
        env_state_msg.pose.position.z = 0.0
        env_state_msg.pose.orientation.w = 1.0
        self.env_state_pub.publish(env_state_msg)
        
        self.get_logger().debug('Published environment state data')


def main(args=None):
    rclpy.init(args=args)
    node = TopicSetupNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main() 