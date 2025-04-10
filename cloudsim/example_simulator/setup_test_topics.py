#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import time
from std_msgs.msg import String, Float32
from geometry_msgs.msg import Twist, PoseStamped
import threading

class TopicSetupNode(Node):
    def __init__(self):
        super().__init__('topic_setup_node')
        self.declare_parameter('sim_frequency', 1.0)  # Hz
        self.declare_parameter('alg_frequency', 1.0)  # Hz
        
        # Get parameters
        sim_frequency = self.get_parameter('sim_frequency').value
        alg_frequency = self.get_parameter('alg_frequency').value
        
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
        
        # Create topics for algorithm
        self.alg_input_pub = self.create_publisher(
            PoseStamped, 
            '/algorithm/input',
            10
        )
        self.alg_output_sub = self.create_subscription(
            Twist,
            '/algorithm/output',
            self.alg_output_callback,
            10
        )
        self.alg_status_pub = self.create_publisher(
            String, 
            '/algorithm/status',
            10
        )
        
        # Create timers for publishing simulated data
        self.sim_timer = self.create_timer(1.0 / sim_frequency, self.publish_simulator_data)
        self.alg_timer = self.create_timer(1.0 / alg_frequency, self.publish_algorithm_data)
        
        # Store received messages
        self.last_cmd = Twist()
        self.last_output = Twist()
        
        self.get_logger().info('Test topics setup complete')
    
    def sim_cmd_callback(self, msg):
        self.get_logger().info(f'Received simulator command: {msg}')
        self.last_cmd = msg
        
        # Echo back status
        status_msg = String()
        status_msg.data = f"Executing command: linear={msg.linear.x:.2f}, angular={msg.angular.z:.2f}"
        self.sim_status_pub.publish(status_msg)
    
    def alg_output_callback(self, msg):
        self.get_logger().info(f'Received algorithm output: {msg}')
        self.last_output = msg
        
        # Echo back status
        status_msg = String()
        status_msg.data = f"Algorithm produced output: linear={msg.linear.x:.2f}, angular={msg.angular.z:.2f}"
        self.alg_status_pub.publish(status_msg)
    
    def publish_simulator_data(self):
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
        # Publish test algorithm input
        pose_msg = PoseStamped()
        pose_msg.header.stamp = self.get_clock().now().to_msg()
        pose_msg.header.frame_id = 'map'
        pose_msg.pose.position.x = 3.0
        pose_msg.pose.position.y = 4.0
        pose_msg.pose.position.z = 0.0
        pose_msg.pose.orientation.w = 1.0
        self.alg_input_pub.publish(pose_msg)
        
        self.get_logger().debug('Published algorithm data')


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