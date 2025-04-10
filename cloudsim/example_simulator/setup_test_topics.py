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
        
        if self.is_simulator:
            self.setup_simulator_topics()
        else:
            self.setup_algorithm_topics()
        
        # Create timer for publishing simulated data
        self.pub_timer = self.create_timer(1.0 / pub_frequency, self.publish_data)
        
        self.get_logger().info(f"{'Simulator' if self.is_simulator else 'Algorithm'} test node setup complete")
    
    def setup_simulator_topics(self):
        """Set up simulator topics"""
        # Create topics for simulator
        self.sim_pose_pub = self.create_publisher(
            PoseStamped, 
            '/simulator/pose',
            10
        )
        self.sim_velocity_pub = self.create_publisher(
            Twist, 
            '/simulator/velocity',
            10
        )
        self.sim_status_pub = self.create_publisher(
            String, 
            '/simulator/status',
            10
        )
        self.sim_cmd_sub = self.create_subscription(
            Twist,
            '/simulator/cmd',
            self.sim_cmd_callback,
            10
        )
        
        # Store received messages
        self.last_cmd = Twist()
    
    def setup_algorithm_topics(self):
        """Set up algorithm topics"""
        # Create topics for algorithm
        self.alg_status_pub = self.create_publisher(
            String, 
            '/algorithm/status',
            10
        )
        self.alg_output_pub = self.create_publisher(
            Twist,
            '/algorithm/output',
            10
        )
        self.alg_input_sub = self.create_subscription(
            PoseStamped,
            '/algorithm/input',
            self.alg_input_callback,
            10
        )
        
        # Store received messages
        self.last_input = PoseStamped()
    
    def sim_cmd_callback(self, msg):
        """Handle commands sent to simulator"""
        self.get_logger().info(f'Received simulator command: {msg}')
        self.last_cmd = msg
        
        # Echo back status
        status_msg = String()
        status_msg.data = f"Executing command: linear={msg.linear.x:.2f}, angular={msg.angular.z:.2f}"
        self.sim_status_pub.publish(status_msg)
    
    def alg_input_callback(self, msg):
        """Handle input sent to algorithm"""
        self.get_logger().info(f'Received algorithm input: {msg}')
        self.last_input = msg
        
        # Echo back status and generate an output as if algorithm processed the input
        status_msg = String()
        status_msg.data = f"Processing input: position=({msg.pose.position.x:.2f}, {msg.pose.position.y:.2f})"
        self.alg_status_pub.publish(status_msg)
        
        # Generate an algorithm output based on the input
        output_msg = Twist()
        # Simple algorithm: move toward origin
        output_msg.linear.x = -0.1 * msg.pose.position.x
        output_msg.linear.y = -0.1 * msg.pose.position.y
        output_msg.angular.z = 0.1
        self.alg_output_pub.publish(output_msg)
    
    def publish_data(self):
        """Publish data based on node type"""
        if self.is_simulator:
            self.publish_simulator_data()
        else:
            self.publish_algorithm_data()
    
    def publish_simulator_data(self):
        """Publish simulator data"""
        # Publish simulated pose
        pose_msg = PoseStamped()
        pose_msg.header.stamp = self.get_clock().now().to_msg()
        pose_msg.header.frame_id = 'map'
        pose_msg.pose.position.x = 1.0
        pose_msg.pose.position.y = 2.0
        pose_msg.pose.position.z = 0.0
        pose_msg.pose.orientation.w = 1.0
        self.sim_pose_pub.publish(pose_msg)
        
        # Publish simulated velocity
        vel_msg = Twist()
        vel_msg.linear.x = 0.5
        vel_msg.angular.z = 0.1
        self.sim_velocity_pub.publish(vel_msg)
        
        # Publish status
        status_msg = String()
        status_msg.data = "Simulator running normally"
        self.sim_status_pub.publish(status_msg)
        
        self.get_logger().debug('Published simulator data')
    
    def publish_algorithm_data(self):
        """Publish algorithm status"""
        # Publish algorithm status
        status_msg = String()
        status_msg.data = "Algorithm waiting for input"
        self.alg_status_pub.publish(status_msg)
        
        self.get_logger().debug('Published algorithm status')


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