#!/usr/bin/env python3
"""
Reliability comparison study between FogSim and TUN/TAP interfaces.

This script compares three setups:
1. FogSim (virtual timeline) - baseline
2. TUN/TAP with basic message passing  
3. TUN/TAP with busy background messages

Measures timing variance, out-of-order messages, and result reliability.
"""

import os
import sys
import time
import json
import math
import struct
import socket
import threading
import subprocess
import statistics
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod

# FogSim imports
from fogsim import Env, NetworkConfig
from fogsim.handlers import BaseHandler

# Import the collision handler from main.py
from main import (CollisionHandler, unprotected_right_turn_config, run_first_simulation, 
                  run_obstacle_only_simulation, load_trajectory, calculate_collision_probabilities)


@dataclass
class TimingMeasurement:
    """Container for timing measurements"""
    send_time: float
    receive_time: float
    sequence_id: int
    message_type: str  # 'observation' or 'action'
    setup_type: str   # 'fogsim', 'tuntap_basic', 'tuntap_busy'


@dataclass
class PositionState:
    """Container for vehicle position and prediction state"""
    ego_pos: np.ndarray
    obstacle_pos: np.ndarray
    ekf_predictions: List[List[float]]
    tick: int
    timestamp: float


@dataclass
class ReliabilityResults:
    """Container for reliability analysis results"""
    setup_type: str
    total_messages: int
    out_of_order_count: int
    timing_variance: float
    mean_latency: float
    max_latency: float
    min_latency: float
    collision_occurred: bool
    collision_tick: Optional[int]
    final_collision_probability: float


class TunTapInterface:
    """TUN/TAP interface for observation/action communication in loopback mode"""
    
    def __init__(self, interface_name: str = "tap_fogsim", busy_traffic: bool = False):
        self.interface_name = interface_name
        self.busy_traffic = busy_traffic
        self.tap_fd = None
        self.server_socket = None
        self.client_socket = None
        self.running = False
        self.busy_thread = None
        self.localhost_mode = False  # Always False - TAP is required
        
        # Timing tracking
        self.sent_messages = {}  # sequence_id -> send_time
        self.received_messages = []  # List of TimingMeasurement
        self.sequence_counter = 0
        
    def setup_tap_interface(self):
        """Create and configure TAP interface using sudo"""
        try:
            # First check if interface already exists and remove it
            check_result = subprocess.run([
                'sudo', 'ip', 'link', 'show', self.interface_name
            ], capture_output=True, text=True)
            
            if check_result.returncode == 0:
                print(f"TAP interface {self.interface_name} already exists, removing it first...")
                subprocess.run([
                    'sudo', 'ip', 'link', 'del', self.interface_name
                ], check=True, capture_output=True, text=True)
            
            # Create TAP interface using sudo
            create_result = subprocess.run([
                'sudo', 'ip', 'tuntap', 'add', 'dev', self.interface_name, 'mode', 'tap'
            ], check=True, capture_output=True, text=True)
            
            # Bring interface up
            subprocess.run([
                'sudo', 'ip', 'link', 'set', self.interface_name, 'up'
            ], check=True, capture_output=True, text=True)
            
            # Assign IP address
            subprocess.run([
                'sudo', 'ip', 'addr', 'add', '192.168.100.1/24', 'dev', self.interface_name
            ], check=True, capture_output=True, text=True)
            
            print(f"TAP interface {self.interface_name} created successfully")
            self.localhost_mode = False
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"Failed to create TAP interface: {e}")
            print(f"Error output: {e.stderr if hasattr(e, 'stderr') else 'No error details'}")
            # Do not fallback - require TAP to work
            raise RuntimeError(f"TAP interface creation failed: {e}")
    
    def setup_localhost_sockets(self):
        """Fallback: Use localhost UDP sockets instead of TAP"""
        # This method is now deprecated - we require TAP to work
        raise NotImplementedError("Localhost sockets fallback is disabled - TAP interface required")
    
    def cleanup_tap_interface(self):
        """Remove TAP interface"""
        if not self.localhost_mode:
            try:
                subprocess.run([
                    'sudo', 'ip', 'link', 'del', self.interface_name
                ], check=True, capture_output=True, text=True)
                print(f"TAP interface {self.interface_name} removed")
            except subprocess.CalledProcessError as e:
                print(f"Failed to remove TAP interface: {e}")
    
    def start_server(self, port: int = 8888):
        """Start UDP server for receiving observations and sending actions"""
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        
        # Always use TAP interface IP
        bind_ip = '192.168.100.1'
        self.server_socket.bind((bind_ip, port))
        self.server_socket.settimeout(0.001)  # Very short timeout for fast non-blocking
        
        self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.bind_ip = bind_ip
        
        self.running = True
        
        # Start busy traffic thread if requested
        if self.busy_traffic:
            self.busy_thread = threading.Thread(target=self._generate_busy_traffic)
            self.busy_thread.daemon = True
            self.busy_thread.start()
    
    def stop_server(self):
        """Stop the server and cleanup"""
        self.running = False
        
        if self.server_socket:
            self.server_socket.close()
        if self.client_socket:
            self.client_socket.close()
        if self.busy_thread:
            self.busy_thread.join(timeout=1.0)
    
    def _generate_busy_traffic(self):
        """Generate background busy traffic"""
        busy_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        
        while self.running:
            try:
                # Send busy message every 1ms
                busy_data = b'BUSY' + os.urandom(100)  # 100 bytes random data
                busy_socket.sendto(busy_data, (self.bind_ip, 9999))
            except:
                pass
        
        busy_socket.close()
    
    def send_position_message(self, position_data: np.ndarray) -> int:
        """Send position message and return sequence ID (open-loop, no response expected)"""
        self.sequence_counter += 1
        seq_id = self.sequence_counter
        
        # Pack data: sequence_id (4 bytes) + timestamp (8 bytes) + position data
        send_time = time.time()
        data = struct.pack('!I', seq_id) + struct.pack('!d', send_time)
        data += position_data.astype(np.float32).tobytes()
        
        # Record the send time for latency measurement
        self.sent_messages[seq_id] = send_time
        
        try:
            # Send to a receiver that will echo back for latency measurement
            self.client_socket.sendto(data, (self.bind_ip, 8888))
            
            # Record as a sent measurement
            measurement = TimingMeasurement(
                send_time=send_time,
                receive_time=send_time,  # Will be updated when echo received
                sequence_id=seq_id,
                message_type='position',
                setup_type='tuntap_busy' if self.busy_traffic else 'tuntap_basic'
            )
            
        except Exception as e:
            print(f"Failed to send position message: {e}")
        
        return seq_id
    
    def receive_echo(self) -> Tuple[Optional[int], Optional[float]]:
        """Receive echo message for latency measurement (open-loop)"""
        try:
            data, addr = self.server_socket.recvfrom(1024)
            receive_time = time.time()
            
            if len(data) >= 12:  # seq_id + send_time
                seq_id = struct.unpack('!I', data[:4])[0]
                original_send_time = struct.unpack('!d', data[4:12])[0]
                
                # Record timing measurement for round-trip latency
                if seq_id in self.sent_messages:
                    measurement = TimingMeasurement(
                        send_time=self.sent_messages[seq_id],
                        receive_time=receive_time,
                        sequence_id=seq_id,
                        message_type='position_echo',
                        setup_type='tuntap_busy' if self.busy_traffic else 'tuntap_basic'
                    )
                    self.received_messages.append(measurement)
                    
                    # Calculate one-way latency (assuming symmetric)
                    round_trip_latency = receive_time - self.sent_messages[seq_id]
                    one_way_latency = round_trip_latency / 2.0
                    
                    del self.sent_messages[seq_id]
                    return seq_id, one_way_latency
                
        except socket.timeout:
            pass
        except Exception as e:
            print(f"Error receiving echo: {e}")
        
        return None, None
    
    def echo_server(self):
        """Simple echo server for latency measurement (open-loop)"""
        while self.running:
            try:
                data, addr = self.server_socket.recvfrom(1024)
                
                if len(data) >= 12:
                    # Simply echo back the message for latency measurement
                    # This simulates network round-trip without any processing
                    self.client_socket.sendto(data, addr)
                    
            except socket.timeout:
                continue
            except Exception as e:
                print(f"Error in echo server: {e}")
                break


class SimpleEKFTracker:
    """Simple EKF tracker for obstacle vehicle prediction"""
    
    def __init__(self, dt=0.05):
        self.dt = dt
        self.state = np.zeros((5, 1))  # [x, y, theta, v, omega]
        self.P = np.eye(5) * 1000  # Initial state covariance
        self.Q = np.eye(5) * 0.1  # Process noise
        self.R = np.eye(3) * 0.1  # Measurement noise
        self.initialized = False
        
        # Vehicle dimensions (approximate CARLA vehicle sizes)
        self.ego_length = 4.0  # Tesla Model 3 approximate length
        self.ego_width = 1.5   # Tesla Model 3 approximate width
        self.obs_length = 4.0  # Lincoln MKZ approximate length
        self.obs_width = 1.4   # Lincoln MKZ approximate width
        self.obs_cov = np.identity(2) * 0.04  # Observation covariance
    
    def _f(self, x, dt):
        """State transition function"""
        return np.array([
            [x[0, 0] + x[3, 0] * np.cos(x[2, 0]) * dt],  # x + v*cos(theta)*dt
            [x[1, 0] + x[3, 0] * np.sin(x[2, 0]) * dt],  # y + v*sin(theta)*dt
            [x[2, 0] + x[4, 0] * dt],  # theta + omega*dt
            [x[3, 0]],  # v
            [x[4, 0]]  # omega
        ])
    
    def _jacobian_f(self, x, dt):
        """Jacobian of state transition function"""
        theta = x[2, 0]
        v = x[3, 0]
        return np.array([
            [1, 0, -v * np.sin(theta) * dt, np.cos(theta) * dt, 0],
            [0, 1, v * np.cos(theta) * dt, np.sin(theta) * dt, 0],
            [0, 0, 1, 0, dt],
            [0, 0, 0, 1, 0],
            [0, 0, 0, 0, 1]
        ])
    
    def _h(self, x):
        """Measurement function"""
        return np.array([[x[0, 0]], [x[1, 0]], [x[2, 0]]])
    
    def _jacobian_h(self, x):
        """Jacobian of measurement function"""
        return np.array([
            [1, 0, 0, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 0, 1, 0, 0]
        ])
    
    def update(self, measurement, tick):
        """Update tracker with new measurement [x, y, theta]"""
        z = np.array([[measurement[0]], [measurement[1]], [math.radians(measurement[2])]])
        
        if not self.initialized:
            self.state[0:3] = z
            self.initialized = True
            return
        
        # Predict step
        self.state = self._f(self.state, self.dt)
        F = self._jacobian_f(self.state, self.dt)
        self.P = F @ self.P @ F.T + self.Q
        
        # Update step
        h = self._h(self.state)
        H = self._jacobian_h(self.state)
        y = z - h
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)
        
        self.state = self.state + K @ y
        self.P = (np.eye(5) - K @ H) @ self.P
    
    def predict_future_position(self, steps_ahead):
        """Predict future positions using EKF"""
        predicted_states = []
        current_state = self.state.copy()
        for _ in range(steps_ahead):
            current_state = self._f(current_state, self.dt)
            predicted_states.append([
                current_state[0, 0],  # x
                current_state[1, 0],  # y
                current_state[2, 0]   # theta
            ])
        return predicted_states
    
    def calculate_collision_probability(self, ego_state, obstacle_state):
        """
        Calculate collision probability between ego vehicle and obstacle
        ego_state: [x, y, theta in radians]
        obstacle_state: [x, y, theta in radians]
        """
        try:
            # Import required modules
            from scipy.stats import norm
            
            def collision_point_rect(se, so, we=1.5, le=4, wo=1.4, lo=4):
                """Simplified collision detection between two rectangles"""
                # Calculate distance between vehicle centers
                dx = so[0] - se[0]
                dy = so[1] - se[1]
                center_distance = math.sqrt(dx*dx + dy*dy)
                
                # Simple collision approximation: if vehicles are too close
                collision_threshold = math.sqrt((we/2 + wo/2)**2 + (le/2 + lo/2)**2)
                
                if center_distance < collision_threshold:
                    return (so[0], so[1]), (se[0], se[1]), -center_distance
                else:
                    return (so[0], so[1]), (se[0], se[1]), center_distance - collision_threshold
            
            def collision_probability(V, P, dis, obs_cov):
                """Calculate collision probability using normal distribution"""
                if dis <= 0:
                    return 1.0
                
                PV = V - P
                if np.linalg.norm(PV) == 0:
                    return 1.0
                    
                theta_pv = np.arccos(np.dot(PV, np.array([1, 0])) / np.linalg.norm(PV))
                R = np.array([[np.cos(theta_pv), -np.sin(theta_pv)],
                             [np.sin(theta_pv), np.cos(theta_pv)]])
                
                den = np.matmul(np.matmul(R, obs_cov), R.T)
                col_prob = norm.cdf(-dis / np.sqrt(den[0, 0]))
                
                return min(1.0, max(0.0, col_prob))
            
            # Get collision points and distance
            obstacle_point, ego_point, distance_val = collision_point_rect(
                ego_state, obstacle_state,
                we=self.ego_width, le=self.ego_length,
                wo=self.obs_width, lo=self.obs_length)
            
            if distance_val <= 0:  # Already colliding
                return 1.0
            
            # Calculate collision probability
            col_prob = collision_probability(
                np.array(obstacle_point), np.array(ego_point), 
                distance_val, self.obs_cov)
            
            return col_prob
            
        except Exception as e:
            # If calculation fails, return 0 probability
            return 0.0


class FogSimPlotter:
    
    def __init__(self, config, output_dir):
        self.config = config
        self.output_dir = output_dir
        
        # Plotting setup - create a single comprehensive figure
        plt.ioff()  # Turn off interactive mode for better overlay control
        self.fig, self.ax = plt.subplots(figsize=(15, 10))
        
        # Storage for all trajectory data to overlay
        self.all_ego_positions = []
        self.all_obstacle_positions = []
        self.all_predictions_by_timestamp = {}  # timestamp -> predictions
        self.all_collision_probs_by_timestamp = {}  # timestamp -> collision probs
        self.plot_intervals = []  # Track when plots were made
        
    def update_plot_data(self, ego_pos, obstacle_pos, ekf_predictions, collision_probs, tick):
        # Store predictions and collision probabilities for this timestamp (only at intervals)
        if ekf_predictions:
            self.all_predictions_by_timestamp[tick] = ekf_predictions.copy()
        if collision_probs:
            self.all_collision_probs_by_timestamp[tick] = collision_probs.copy()
        
        # Mark this as a plot interval
        self.plot_intervals.append(tick)
        print(f"FogSim: Recording plot data at tick {tick}")
    
    def update_position_data(self, ego_pos, obstacle_pos, tick):
        # Store all positions for complete trajectory
        self.all_ego_positions.append((ego_pos[0], ego_pos[1], tick))
        self.all_obstacle_positions.append((obstacle_pos[0], obstacle_pos[1], tick))
    
    def create_comprehensive_overlay_plot(self, fixed_bounds=None):
        self.ax.clear()
        
        if not self.all_ego_positions or not self.all_obstacle_positions:
            print("No FogSim plot data available to create comprehensive plot")
            return
        
        # Extract trajectory data
        ego_x = [pos[0] for pos in self.all_ego_positions]
        ego_y = [pos[1] for pos in self.all_ego_positions]
        ego_ticks = [pos[2] for pos in self.all_ego_positions]
        
        obs_x = [pos[0] for pos in self.all_obstacle_positions]
        obs_y = [pos[1] for pos in self.all_obstacle_positions]
        obs_ticks = [pos[2] for pos in self.all_obstacle_positions]
        
        # Plot full trajectories
        self.ax.plot(ego_x, ego_y, 'b-', linewidth=2, alpha=0.7, label='Ego Vehicle Trajectory (FogSim)')
        self.ax.plot(obs_x, obs_y, 'r-', linewidth=2, alpha=0.7, label='Obstacle Vehicle Trajectory (FogSim)')
        
        # Mark trajectory start and end points
        if ego_x and ego_y:
            self.ax.plot(ego_x[0], ego_y[0], 'bs', markersize=12, label='Ego Start')
            self.ax.plot(ego_x[-1], ego_y[-1], 'b^', markersize=12, label='Ego End')
        if obs_x and obs_y:
            self.ax.plot(obs_x[0], obs_y[0], 'rs', markersize=12, label='Obstacle Start')
            self.ax.plot(obs_x[-1], obs_y[-1], 'r^', markersize=12, label='Obstacle End')
        
        # Plot prediction lines for every plot interval (every 100 messages)
        prediction_colors = plt.cm.viridis(np.linspace(0, 1, len(self.plot_intervals)))
        max_collision_prob_overall = 0.0
        
        for i, tick in enumerate(self.plot_intervals):
            if tick in self.all_predictions_by_timestamp:
                predictions = self.all_predictions_by_timestamp[tick]
                collision_probs = self.all_collision_probs_by_timestamp.get(tick, [])
                
                if predictions:
                    pred_x = [pred[0] for pred in predictions]
                    pred_y = [pred[1] for pred in predictions]
                    
                    # Find ego position at this tick for prediction origin
                    ego_pos_at_tick = None
                    for ego_pos in self.all_ego_positions:
                        if ego_pos[2] == tick:
                            ego_pos_at_tick = (ego_pos[0], ego_pos[1])
                            break
                    
                    # Plot prediction line
                    color = prediction_colors[i]
                    alpha = 0.3 + 0.4 * (i / max(1, len(self.plot_intervals) - 1))  # Increase alpha over time
                    self.ax.plot(pred_x, pred_y, '--', color=color, linewidth=1.5, alpha=alpha, 
                               label=f'EKF Predictions (Tick {tick})' if i < 3 else '')  # Only label first few
                    
                    # Plot prediction points colored by collision probability
                    if collision_probs and len(collision_probs) == len(predictions):
                        max_prob_this_step = max(collision_probs)
                        max_collision_prob_overall = max(max_collision_prob_overall, max_prob_this_step)
                        
                        for j, (pred, prob) in enumerate(zip(predictions, collision_probs)):
                            # Color intensity based on collision probability
                            color_intensity = min(1.0, prob * 3)  # Scale for visibility
                            point_color = (color_intensity, 1-color_intensity, 0.2)
                            self.ax.plot(pred[0], pred[1], 'o', 
                                       color=point_color, markersize=4, alpha=0.8)
                    else:
                        # No collision data, just plot predictions
                        self.ax.plot(pred_x, pred_y, 'o', color=color, markersize=3, alpha=0.6)
                    
                    # Connect prediction origin to first prediction point
                    if ego_pos_at_tick and predictions:
                        self.ax.plot([ego_pos_at_tick[0], predictions[0][0]], 
                                   [ego_pos_at_tick[1], predictions[0][1]], 
                                   ':', color=color, alpha=0.5)
        
        # Mark plot intervals on the trajectories
        for i, tick in enumerate(self.plot_intervals):
            # Find positions at this tick
            ego_pos_at_tick = None
            obs_pos_at_tick = None
            
            for ego_pos in self.all_ego_positions:
                if ego_pos[2] == tick:
                    ego_pos_at_tick = (ego_pos[0], ego_pos[1])
                    break
            
            for obs_pos in self.all_obstacle_positions:
                if obs_pos[2] == tick:
                    obs_pos_at_tick = (obs_pos[0], obs_pos[1])
                    break
            
            # Mark these positions
            if ego_pos_at_tick:
                self.ax.plot(ego_pos_at_tick[0], ego_pos_at_tick[1], 'bo', 
                           markersize=8, alpha=0.8, markeredgecolor='darkblue', markeredgewidth=1)
            if obs_pos_at_tick:
                self.ax.plot(obs_pos_at_tick[0], obs_pos_at_tick[1], 'ro', 
                           markersize=8, alpha=0.8, markeredgecolor='darkred', markeredgewidth=1)
        
        # Add collision probability information if available
        collision_text = ''
        if max_collision_prob_overall > 0:
            collision_text = f', Max Collision Probability: {max_collision_prob_overall:.4f}'
            
            # Add text annotation for collision probability scale
            self.ax.text(0.02, 0.98, f'Collision Risk Scale (FogSim):\\nRed = High Risk\\nYellow = Medium Risk\\nGreen = Low Risk', 
                        transform=self.ax.transAxes, bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8),
                        fontsize=9, verticalalignment='top')
        
        # Set plot properties
        self.ax.set_xlabel('X Position (m)', fontsize=12)
        self.ax.set_ylabel('Y Position (m)', fontsize=12)
        self.ax.set_title(f'FogSim: Comprehensive Vehicle Trajectories and EKF Predictions\\n'
                         f'Total Simulation Steps: {max(ego_ticks) if ego_ticks else 0}, '
                         f'Prediction Intervals: {len(self.plot_intervals)}{collision_text}', 
                         fontsize=14)
        
        self.ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        self.ax.grid(True, alpha=0.3)
        self.ax.axis('equal')
        
        # Set axis limits with margin
        all_x = ego_x + obs_x
        all_y = ego_y + obs_y
        
        # Add prediction points to bounds calculation
        for predictions in self.all_predictions_by_timestamp.values():
            all_x.extend([pred[0] for pred in predictions])
            all_y.extend([pred[1] for pred in predictions])
        
        if fixed_bounds:
            # Use provided bounds for consistent comparison
            self.ax.set_xlim(fixed_bounds[0], fixed_bounds[1])
            self.ax.set_ylim(fixed_bounds[2], fixed_bounds[3])
        elif all_x and all_y:
            margin = max(5, (max(all_x) - min(all_x)) * 0.1)
            self.ax.set_xlim(min(all_x) - margin, max(all_x) + margin)
            self.ax.set_ylim(min(all_y) - margin, max(all_y) + margin)
        
        # Save the comprehensive plot
        plot_filename = os.path.join(self.output_dir, 'comprehensive_trajectory_overlay_fogsim.png')
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        print(f"FogSim comprehensive overlay plot saved to {plot_filename}")
        
        # Also close the figure to free memory
        plt.close(self.fig)


class SimplePositionSender:
    """Simple position sender that uses pre-recorded CARLA trajectories with EKF predictions"""
    
    def __init__(self, config, output_dir, tuntap_interface: TunTapInterface):
        self.config = config
        self.output_dir = output_dir
        self.tuntap = tuntap_interface
        
        # Position tracking
        self.tick = 0
        self.position_timing_measurements = []
        
        # Load or generate trajectories using CARLA simulation
        self._ensure_trajectories_exist()
        
        # Load trajectories - these are in CARLA coordinates (x, y, yaw_degrees)
        # load_trajectory converts yaw to radians
        self.ego_trajectory = load_trajectory(config['trajectories']['ego'])
        self.obstacle_trajectory = load_trajectory(config['trajectories']['obstacle'])
        
        print(f"Loaded {len(self.ego_trajectory)} ego trajectory points")
        print(f"Loaded {len(self.obstacle_trajectory)} obstacle trajectory points")
        
        # Debug: Print first and last positions to verify alignment
        if self.ego_trajectory:
            print(f"Ego trajectory start: x={self.ego_trajectory[0][0]:.2f}, y={self.ego_trajectory[0][1]:.2f}, yaw={math.degrees(self.ego_trajectory[0][2]):.2f}°")
            print(f"Ego trajectory end: x={self.ego_trajectory[-1][0]:.2f}, y={self.ego_trajectory[-1][1]:.2f}, yaw={math.degrees(self.ego_trajectory[-1][2]):.2f}°")
        if self.obstacle_trajectory:
            print(f"Obstacle trajectory start: x={self.obstacle_trajectory[0][0]:.2f}, y={self.obstacle_trajectory[0][1]:.2f}, yaw={math.degrees(self.obstacle_trajectory[0][2]):.2f}°")
            print(f"Obstacle trajectory end: x={self.obstacle_trajectory[-1][0]:.2f}, y={self.obstacle_trajectory[-1][1]:.2f}, yaw={math.degrees(self.obstacle_trajectory[-1][2]):.2f}°")
        
        # EKF tracker for obstacle predictions
        self.ekf_tracker = SimpleEKFTracker(dt=config['simulation']['delta_seconds'])
        
        # Position history for plotting
        self.position_history = []
        
        # Plotting setup - create a single comprehensive figure
        plt.ioff()  # Turn off interactive mode for better overlay control
        self.fig, self.ax = plt.subplots(figsize=(15, 10))
        self.plot_counter = 0
        
        # Storage for all trajectory data to overlay
        self.all_ego_positions = []
        self.all_obstacle_positions = []
        self.all_predictions_by_timestamp = {}  # timestamp -> predictions
        self.all_collision_probs_by_timestamp = {}  # timestamp -> collision probs
        self.plot_intervals = []  # Track when plots were made
        
    def launch(self):
        """Initialize the handler"""
        # Simple initialization
        self.tick = 0
        print("Position sender initialized")
        
    def _ensure_trajectories_exist(self):
        """Generate trajectories using CARLA if they don't exist"""
        # Always regenerate trajectories to ensure consistency
        ego_traj_file = self.config['trajectories']['ego']
        obstacle_traj_file = self.config['trajectories']['obstacle']
        
        # Remove existing files to force regeneration
        if os.path.exists(ego_traj_file):
            os.remove(ego_traj_file)
            print(f"Removed existing ego trajectory file: {ego_traj_file}")
            
        if os.path.exists(obstacle_traj_file):
            os.remove(obstacle_traj_file)
            print(f"Removed existing obstacle trajectory file: {obstacle_traj_file}")
        
        print("Generating ego vehicle trajectory using CARLA...")
        print(f"Expected ego trajectory length: {self.config['ego_vehicle']['go_straight_ticks'] + self.config['ego_vehicle']['turn_ticks'] + self.config['ego_vehicle']['after_turn_ticks']} ticks")
        run_first_simulation(self.config, ego_traj_file)
        
        print("Generating obstacle vehicle trajectory using CARLA...")  
        print(f"Expected obstacle trajectory length: {self.config['obstacle_vehicle']['go_straight_ticks'] + self.config['obstacle_vehicle']['turn_ticks'] + self.config['obstacle_vehicle']['after_turn_ticks']} ticks")
        run_obstacle_only_simulation(self.config, obstacle_traj_file)
        
        # Verify trajectory files were created correctly
        if os.path.exists(ego_traj_file):
            with open(ego_traj_file, 'r') as f:
                ego_lines = len(f.readlines())
            print(f"Generated ego trajectory file with {ego_lines} points")
        else:
            raise RuntimeError(f"Failed to generate ego trajectory file: {ego_traj_file}")
            
        if os.path.exists(obstacle_traj_file):
            with open(obstacle_traj_file, 'r') as f:
                obstacle_lines = len(f.readlines())
            print(f"Generated obstacle trajectory file with {obstacle_lines} points")
        else:
            raise RuntimeError(f"Failed to generate obstacle trajectory file: {obstacle_traj_file}")
    
    def get_ego_position(self):
        """Get current ego vehicle position from pre-recorded trajectory"""
        if self.tick < len(self.ego_trajectory):
            ego_pos = self.ego_trajectory[self.tick]
            return np.array([ego_pos[0], ego_pos[1], ego_pos[2]], dtype=np.float32)
        else:
            # Use last position if we've exceeded trajectory length
            ego_pos = self.ego_trajectory[-1]
            return np.array([ego_pos[0], ego_pos[1], ego_pos[2]], dtype=np.float32)
    
    def get_obstacle_position(self):
        """Get current obstacle vehicle position from pre-recorded trajectory"""
        if self.tick < len(self.obstacle_trajectory):
            obs_pos = self.obstacle_trajectory[self.tick]
            return np.array([obs_pos[0], obs_pos[1], obs_pos[2]], dtype=np.float32)
        else:
            # Use last position if we've exceeded trajectory length
            obs_pos = self.obstacle_trajectory[-1]
            return np.array([obs_pos[0], obs_pos[1], obs_pos[2]], dtype=np.float32)
    
    def update_plot_data(self, ego_pos, obstacle_pos, ekf_predictions, collision_probs=None):
        """Update the stored data for comprehensive overlay plotting"""
        # Store current positions
        self.all_ego_positions.append((ego_pos[0], ego_pos[1], self.tick))
        self.all_obstacle_positions.append((obstacle_pos[0], obstacle_pos[1], self.tick))
        
        # Store predictions and collision probabilities for this timestamp
        if ekf_predictions:
            self.all_predictions_by_timestamp[self.tick] = ekf_predictions.copy()
        if collision_probs:
            self.all_collision_probs_by_timestamp[self.tick] = collision_probs.copy()
        
        # Mark this as a plot interval
        self.plot_intervals.append(self.tick)
    
    def create_comprehensive_overlay_plot(self, fixed_bounds=None):
        """Create a single comprehensive plot with all trajectories, predictions, and collision data overlaid"""
        self.ax.clear()
        
        if not self.all_ego_positions or not self.all_obstacle_positions:
            return
        
        # Extract trajectory data
        ego_x = [pos[0] for pos in self.all_ego_positions]
        ego_y = [pos[1] for pos in self.all_ego_positions]
        ego_ticks = [pos[2] for pos in self.all_ego_positions]
        
        obs_x = [pos[0] for pos in self.all_obstacle_positions]
        obs_y = [pos[1] for pos in self.all_obstacle_positions]
        obs_ticks = [pos[2] for pos in self.all_obstacle_positions]
        
        # Plot full trajectories
        self.ax.plot(ego_x, ego_y, 'b-', linewidth=2, alpha=0.7, label='Ego Vehicle Trajectory')
        self.ax.plot(obs_x, obs_y, 'r-', linewidth=2, alpha=0.7, label='Obstacle Vehicle Trajectory')
        
        # Mark trajectory start and end points
        if ego_x and ego_y:
            self.ax.plot(ego_x[0], ego_y[0], 'bs', markersize=12, label='Ego Start')
            self.ax.plot(ego_x[-1], ego_y[-1], 'b^', markersize=12, label='Ego End')
        if obs_x and obs_y:
            self.ax.plot(obs_x[0], obs_y[0], 'rs', markersize=12, label='Obstacle Start')
            self.ax.plot(obs_x[-1], obs_y[-1], 'r^', markersize=12, label='Obstacle End')
        
        # Plot prediction lines for every plot interval (every 100 messages)
        prediction_colors = plt.cm.viridis(np.linspace(0, 1, len(self.plot_intervals)))
        max_collision_prob_overall = 0.0
        
        for i, tick in enumerate(self.plot_intervals):
            if tick in self.all_predictions_by_timestamp:
                predictions = self.all_predictions_by_timestamp[tick]
                collision_probs = self.all_collision_probs_by_timestamp.get(tick, [])
                
                if predictions:
                    pred_x = [pred[0] for pred in predictions]
                    pred_y = [pred[1] for pred in predictions]
                    
                    # Find ego position at this tick for prediction origin
                    ego_pos_at_tick = None
                    for ego_pos in self.all_ego_positions:
                        if ego_pos[2] == tick:
                            ego_pos_at_tick = (ego_pos[0], ego_pos[1])
                            break
                    
                    # Plot prediction line
                    color = prediction_colors[i]
                    alpha = 0.3 + 0.4 * (i / max(1, len(self.plot_intervals) - 1))  # Increase alpha over time
                    self.ax.plot(pred_x, pred_y, '--', color=color, linewidth=1.5, alpha=alpha, 
                               label=f'EKF Predictions (Tick {tick})' if i < 3 else '')  # Only label first few
                    
                    # Plot prediction points colored by collision probability
                    if collision_probs and len(collision_probs) == len(predictions):
                        max_prob_this_step = max(collision_probs)
                        max_collision_prob_overall = max(max_collision_prob_overall, max_prob_this_step)
                        
                        for j, (pred, prob) in enumerate(zip(predictions, collision_probs)):
                            # Color intensity based on collision probability
                            color_intensity = min(1.0, prob * 3)  # Scale for visibility
                            point_color = (color_intensity, 1-color_intensity, 0.2)
                            self.ax.plot(pred[0], pred[1], 'o', 
                                       color=point_color, markersize=4, alpha=0.8)
                    else:
                        # No collision data, just plot predictions
                        self.ax.plot(pred_x, pred_y, 'o', color=color, markersize=3, alpha=0.6)
                    
                    # Connect prediction origin to first prediction point
                    if ego_pos_at_tick and predictions:
                        self.ax.plot([ego_pos_at_tick[0], predictions[0][0]], 
                                   [ego_pos_at_tick[1], predictions[0][1]], 
                                   ':', color=color, alpha=0.5)
        
        # Mark plot intervals on the trajectories
        for i, tick in enumerate(self.plot_intervals):
            # Find positions at this tick
            ego_pos_at_tick = None
            obs_pos_at_tick = None
            
            for ego_pos in self.all_ego_positions:
                if ego_pos[2] == tick:
                    ego_pos_at_tick = (ego_pos[0], ego_pos[1])
                    break
            
            for obs_pos in self.all_obstacle_positions:
                if obs_pos[2] == tick:
                    obs_pos_at_tick = (obs_pos[0], obs_pos[1])
                    break
            
            # Mark these positions
            if ego_pos_at_tick:
                self.ax.plot(ego_pos_at_tick[0], ego_pos_at_tick[1], 'bo', 
                           markersize=8, alpha=0.8, markeredgecolor='darkblue', markeredgewidth=1)
            if obs_pos_at_tick:
                self.ax.plot(obs_pos_at_tick[0], obs_pos_at_tick[1], 'ro', 
                           markersize=8, alpha=0.8, markeredgecolor='darkred', markeredgewidth=1)
        
        # Add collision probability information if available
        collision_text = ''
        if max_collision_prob_overall > 0:
            collision_text = f', Max Collision Probability: {max_collision_prob_overall:.4f}'
            
            # Add text annotation for collision probability scale
            self.ax.text(0.02, 0.98, f'Collision Risk Scale:\nRed = High Risk\nYellow = Medium Risk\nGreen = Low Risk', 
                        transform=self.ax.transAxes, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                        fontsize=9, verticalalignment='top')
        
        # Set plot properties
        self.ax.set_xlabel('X Position (m)', fontsize=12)
        self.ax.set_ylabel('Y Position (m)', fontsize=12)
        self.ax.set_title(f'Comprehensive Vehicle Trajectories and EKF Predictions\n'
                         f'Total Simulation Steps: {max(ego_ticks) if ego_ticks else 0}, '
                         f'Prediction Intervals: {len(self.plot_intervals)}{collision_text}', 
                         fontsize=14)
        
        self.ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        self.ax.grid(True, alpha=0.3)
        self.ax.axis('equal')
        
        # Set axis limits with margin
        all_x = ego_x + obs_x
        all_y = ego_y + obs_y
        
        # Add prediction points to bounds calculation
        for predictions in self.all_predictions_by_timestamp.values():
            all_x.extend([pred[0] for pred in predictions])
            all_y.extend([pred[1] for pred in predictions])
        
        if fixed_bounds:
            # Use provided bounds for consistent comparison
            self.ax.set_xlim(fixed_bounds[0], fixed_bounds[1])
            self.ax.set_ylim(fixed_bounds[2], fixed_bounds[3])
        elif all_x and all_y:
            margin = max(5, (max(all_x) - min(all_x)) * 0.1)
            self.ax.set_xlim(min(all_x) - margin, max(all_x) + margin)
            self.ax.set_ylim(min(all_y) - margin, max(all_y) + margin)
        
        # Save the comprehensive plot
        plot_filename = os.path.join(self.output_dir, 'comprehensive_trajectory_overlay.png')
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        print(f"Comprehensive overlay plot saved to {plot_filename}")
        
    def step(self):
        """Single simulation step - send position and measure latency"""
        self.tick += 1
        
        # Get current positions from pre-recorded CARLA trajectories
        ego_pos = self.get_ego_position()
        obstacle_pos = self.get_obstacle_position()
        
        # Store all positions for comprehensive plotting (not just every 100)
        self.all_ego_positions.append((ego_pos[0], ego_pos[1], self.tick))
        self.all_obstacle_positions.append((obstacle_pos[0], obstacle_pos[1], self.tick))
        
        # Update EKF tracker with obstacle observation
        # Note: The trajectories are already in radians, convert to degrees for EKF
        obstacle_obs = [obstacle_pos[0], obstacle_pos[1], math.degrees(obstacle_pos[2])]
        self.ekf_tracker.update(obstacle_obs, self.tick)
        
        # Get EKF predictions
        prediction_steps = 20  # Predict 20 steps ahead
        ekf_predictions = self.ekf_tracker.predict_future_position(prediction_steps)
        
        # Calculate collision probabilities for predictions if we have enough trajectory data
        collision_probs = []
        if (hasattr(self.ekf_tracker, 'calculate_collision_probability') and 
            len(self.ego_trajectory) > self.tick + prediction_steps):
            try:
                for i, pred_pos in enumerate(ekf_predictions):
                    if self.tick + i < len(self.ego_trajectory):
                        ego_traj_point = self.ego_trajectory[self.tick + i]
                        # Convert degrees back to radians for collision calculation
                        pred_pos_rad = [pred_pos[0], pred_pos[1], pred_pos[2]]
                        collision_prob = self.ekf_tracker.calculate_collision_probability(
                            ego_traj_point, pred_pos_rad)
                        collision_probs.append(collision_prob)
            except Exception as e:
                # If collision probability calculation fails, continue without it
                collision_probs = []
        
        # Store position state
        position_state = PositionState(
            ego_pos=ego_pos.copy(),
            obstacle_pos=obstacle_pos.copy(),
            ekf_predictions=ekf_predictions.copy(),
            tick=self.tick,
            timestamp=time.time()
        )
        self.position_history.append(position_state)
        
        # Update plot data every 100 messages
        if self.tick % 100 == 0:
            print(f"Recording plot data at tick {self.tick}")
            self.update_plot_data(ego_pos, obstacle_pos, ekf_predictions, collision_probs)
            self.plot_counter += 1
        
        # Send ego position message via TUN/TAP
        seq_id = self.tuntap.send_position_message(ego_pos)
        
        # Try to receive any pending echo messages (non-blocking, very fast)
        # Check multiple times in case there are queued echoes
        for _ in range(5):  # Check up to 5 pending echoes
            echo_seq_id, latency = self.tuntap.receive_echo()
            if echo_seq_id is not None and latency is not None:
                self.position_timing_measurements.append({
                    'tick': self.tick,
                    'sequence_id': echo_seq_id,
                    'latency': latency,
                    'timestamp': time.time(),
                    'ego_position': ego_pos.tolist(),
                    'obstacle_position': obstacle_pos.tolist(),
                    'ekf_predictions': ekf_predictions
                })
            else:
                break  # No more echoes pending
        
        return ego_pos
        
    def close(self):
        """Cleanup"""
        # Generate the comprehensive overlay plot before closing
        if self.plot_intervals:  # Only create plot if we have data
            print("Creating comprehensive trajectory overlay plot...")
            # Calculate common bounds from trajectories
            bounds = get_common_plot_bounds([self.config['trajectories']['ego'], 
                                            self.config['trajectories']['obstacle']])
            self.create_comprehensive_overlay_plot(fixed_bounds=bounds)
        
        plt.ioff()  # Turn off interactive mode
        plt.close(self.fig)
        
        # Save position history to file
        history_file = os.path.join(self.output_dir, 'position_history.json')
        history_data = []
        for pos_state in self.position_history:
            history_data.append({
                'tick': pos_state.tick,
                'timestamp': pos_state.timestamp,
                'ego_pos': pos_state.ego_pos.tolist(),
                'obstacle_pos': pos_state.obstacle_pos.tolist(),
                'ekf_predictions': pos_state.ekf_predictions
            })
        
        with open(history_file, 'w') as f:
            json.dump(history_data, f, indent=2)
        print(f"Position history saved to {history_file}")


class PlottingCollisionHandler(CollisionHandler):
    """Extended CollisionHandler that captures data for comprehensive plotting"""
    
    def __init__(self, config, output_dir, no_risk_eval=False):
        super().__init__(config, output_dir, no_risk_eval)
        self.fogsim_plotter = FogSimPlotter(config, output_dir)
        
    def step_with_action(self, action):
        result = super().step_with_action(action)
        
        # Always capture position data for complete trajectory
        try:
            ego_transform = self.ego_vehicle.get_transform()
            ego_pos = np.array([ego_transform.location.x, ego_transform.location.y, ego_transform.rotation.yaw])
            
            obs_transform = self.obstacle_vehicle.get_transform()
            obs_pos = np.array([obs_transform.location.x, obs_transform.location.y, obs_transform.rotation.yaw])
            
            # Update position data every step
            self.fogsim_plotter.update_position_data(ego_pos, obs_pos, self.tick)
            
            # Capture prediction data every 100 steps for plotting
            if self.tick % 100 == 0:
                # Get EKF predictions if available
                ekf_predictions = []
                collision_probs = []
                
                if self.obstacle_tracker and hasattr(self.obstacle_tracker, 'predict_future_position'):
                    prediction_steps = 20
                    ekf_predictions = self.obstacle_tracker.predict_future_position(prediction_steps)
                    
                    # Calculate collision probabilities if possible
                    if hasattr(self.obstacle_tracker, 'calculate_collision_probability') and self.ego_trajectory:
                        for i, pred_pos in enumerate(ekf_predictions):
                            if self.tick + i < len(self.ego_trajectory):
                                ego_traj_point = self.ego_trajectory[self.tick + i]
                                # EKF predictions are in [x, y, theta_radians] format
                                collision_prob = self.obstacle_tracker.calculate_collision_probability(
                                    ego_traj_point, pred_pos)
                                collision_probs.append(collision_prob)
                
                # Update plotter prediction data
                self.fogsim_plotter.update_plot_data(ego_pos, obs_pos, ekf_predictions, collision_probs, self.tick)
                
        except Exception as e:
            print(f"Error capturing FogSim data at tick {self.tick}: {e}")
                
        return result
        
    def close(self):
        # Generate comprehensive plot before closing with aligned bounds
        try:
            if hasattr(self, 'fogsim_plotter'):
                # Calculate common bounds from trajectories for alignment
                bounds = get_common_plot_bounds([self.config['trajectories']['ego'], 
                                                self.config['trajectories']['obstacle']])
                self.fogsim_plotter.create_comprehensive_overlay_plot(fixed_bounds=bounds)
        except Exception as e:
            print(f"Error creating FogSim comprehensive plot: {e}")
        super().close()


def run_fogsim_simulation(config: Dict, output_dir: str) -> ReliabilityResults:
    """Run simulation using FogSim (baseline) with comprehensive plotting"""
    print("Running FogSim simulation with enhanced plotting...")
    
    try:
        from fogsim import Env, NetworkConfig
        
        # Create dedicated output directory
        fogsim_output_dir = os.path.join(output_dir, 'fogsim')
        os.makedirs(fogsim_output_dir, exist_ok=True)
        
        # Ensure trajectories exist before running FogSim
        if not os.path.exists(config['trajectories']['ego']):
            print("Generating ego vehicle trajectory for FogSim...")
            run_first_simulation(config, config['trajectories']['ego'])
            
        if os.path.exists(config['trajectories']['ego']):
            with open(config['trajectories']['ego'], 'r') as f:
                ego_lines = len(f.readlines())
            print(f"Loaded ego trajectory with {ego_lines} points for FogSim")
            
        if not os.path.exists(config['trajectories']['obstacle']):
            print("Generating obstacle vehicle trajectory for FogSim...")
            run_obstacle_only_simulation(config, config['trajectories']['obstacle'])
            
        if os.path.exists(config['trajectories']['obstacle']):
            with open(config['trajectories']['obstacle'], 'r') as f:
                obstacle_lines = len(f.readlines())
            print(f"Loaded obstacle trajectory with {obstacle_lines} points for FogSim")
        
        # Create network configuration based on delta_k
        network_delay = config['simulation']['delta_k'] * config['simulation']['delta_seconds']
        network_config = NetworkConfig(
            source_rate=1e6,  # 1 Mbps
            topology={'link_delay': network_delay}
        )
        
        # Create our custom handler with plotting
        handler = PlottingCollisionHandler(config, fogsim_output_dir)
        
        # Create FogSim environment
        print(f"Creating FogSim with network delay: {network_delay}s")
        env = Env(
            handler=handler,
            network_config=network_config,
            enable_network=True,
            timestep=config['simulation']['delta_seconds']
        )
        
        # Initialize environment
        print("Initializing FogSim environment...")
        obs, info = env.reset()
        
        # Total simulation steps
        total_steps = (config['ego_vehicle']['go_straight_ticks'] +
                      config['ego_vehicle']['turn_ticks'] +
                      config['ego_vehicle']['after_turn_ticks'])
        
        print(f"Starting FogSim simulation with {total_steps} steps...")
        has_collided = False
        current_delta_k = config['simulation']['delta_k']
        
        for step in range(total_steps):
            if step % 100 == 0:
                print(f"FogSim Step {step}/{total_steps}")
            
            # Simple action policy (no brake for this comparison)
            action = 0  # No brake
            
            # Step environment with action
            obs, reward, success, termination, timeout, info = env.step(action)
            
            # Check for collision
            if handler.has_collided:
                has_collided = True
                print("FogSim: Collision detected! Stopping simulation.")
                break
        
        # Close environment (this will trigger the comprehensive plot generation)
        env.close()
        
        final_delta_k = current_delta_k
        
        # Read collision probability results
        collision_prob_file = os.path.join(fogsim_output_dir, 
            f'collision_probabilities_{config["simulation"]["l_max"]}_fogsim.csv')
        
        timing_measurements = []
        final_collision_prob = 0.0
        
        if os.path.exists(collision_prob_file):
            with open(collision_prob_file, 'r') as f:
                lines = f.readlines()[1:]  # Skip header
                for line in lines:
                    parts = line.strip().split(',')
                    if len(parts) >= 4:
                        timestamp = float(parts[0])
                        tick = int(parts[1])
                        delta_k = int(parts[2])
                        collision_prob = float(parts[3])
                        
                        # Use delta_seconds as step time (simulated timing)
                        timing_measurements.append(config['simulation']['delta_seconds'])
                        final_collision_prob = collision_prob
        
        # Calculate results
        results = ReliabilityResults(
            setup_type='fogsim',
            total_messages=len(timing_measurements),
            out_of_order_count=0,  # FogSim maintains order by design
            timing_variance=statistics.variance(timing_measurements) if len(timing_measurements) > 1 else 0.0,
            mean_latency=statistics.mean(timing_measurements) if timing_measurements else 0.0,
            max_latency=max(timing_measurements) if timing_measurements else 0.0,
            min_latency=min(timing_measurements) if timing_measurements else 0.0,
            collision_occurred=has_collided,
            collision_tick=None,  # Would need to parse from logs
            final_collision_probability=final_collision_prob
        )
        
        print(f"FogSim simulation completed. Collision: {has_collided}, Final delta_k: {final_delta_k}")
        
    except Exception as e:
        print(f"FogSim simulation failed: {e}")
        import traceback
        traceback.print_exc()
        
        # Return empty results on failure
        results = ReliabilityResults(
            setup_type='fogsim',
            total_messages=0,
            out_of_order_count=0,
            timing_variance=0.0,
            mean_latency=0.0,
            max_latency=0.0,
            min_latency=0.0,
            collision_occurred=False,
            collision_tick=None,
            final_collision_probability=0.0
        )
    
    return results


def run_tuntap_simulation(config: Dict, output_dir: str, busy_traffic: bool = False) -> ReliabilityResults:
    """Run simulation using TUN/TAP interface - just send vehicle positions"""
    setup_type = 'tuntap_busy' if busy_traffic else 'tuntap_basic'
    print(f"Running TUN/TAP position sending simulation (busy_traffic={busy_traffic})...")
    
    # Create TUN/TAP interface with shorter name (max 15 chars for Linux interfaces)
    interface_suffix = 'busy' if busy_traffic else 'basic'
    tuntap = TunTapInterface(
        interface_name=f"tap_{interface_suffix}",
        busy_traffic=busy_traffic
    )
    
    handler = None
    server_thread = None
    
    try:
        # Setup TAP interface (no fallback)
        tuntap.setup_tap_interface()
        
        # Create dedicated output directory
        tuntap_output_dir = os.path.join(output_dir, setup_type)
        os.makedirs(tuntap_output_dir, exist_ok=True)
        
        # Start server
        tuntap.start_server()
        
        # Start echo server in separate thread for latency measurement
        server_thread = threading.Thread(target=tuntap.echo_server)
        server_thread.daemon = True
        server_thread.start()
        
        # Give server time to start
        time.sleep(0.1)
        
        # Create simple position sender
        handler = SimplePositionSender(config, tuntap_output_dir, tuntap)
        handler.launch()
        
        # Run simulation for the same duration as CARLA trajectories
        total_steps = min(len(handler.ego_trajectory), len(handler.obstacle_trajectory))
        if total_steps == 0:
            total_steps = (config['ego_vehicle']['go_straight_ticks'] +
                          config['ego_vehicle']['turn_ticks'] + 
                          config['ego_vehicle']['after_turn_ticks'])
        
        print(f"Sending {total_steps} position messages...")
        
        for step in range(total_steps):
            if step % 100 == 0:
                print(f"Step {step}/{total_steps}")
            
            # Send position and potentially receive echo
            position = handler.step()
                    
        # Analyze timing measurements from both TUN/TAP interface and handler
        measurements = tuntap.received_messages + [
            TimingMeasurement(
                send_time=m['timestamp'] - m['latency'],
                receive_time=m['timestamp'],
                sequence_id=m.get('sequence_id', 0),
                message_type='position',
                setup_type=setup_type
            ) for m in handler.position_timing_measurements
        ]
        
        latencies = []
        for m in measurements:
            if hasattr(m, 'receive_time') and hasattr(m, 'send_time'):
                latencies.append(m.receive_time - m.send_time)
        
        # Check for out-of-order messages
        out_of_order_count = 0
        sorted_measurements = sorted(measurements, key=lambda x: getattr(x, 'send_time', 0))
        for i in range(1, len(sorted_measurements)):
            if (hasattr(sorted_measurements[i], 'sequence_id') and 
                hasattr(sorted_measurements[i-1], 'sequence_id')):
                if sorted_measurements[i].sequence_id < sorted_measurements[i-1].sequence_id:
                    out_of_order_count += 1
        
        results = ReliabilityResults(
            setup_type=setup_type,
            total_messages=len(measurements),
            out_of_order_count=out_of_order_count,
            timing_variance=statistics.variance(latencies) if len(latencies) > 1 else 0.0,
            mean_latency=statistics.mean(latencies) if latencies else 0.0,
            max_latency=max(latencies) if latencies else 0.0,
            min_latency=min(latencies) if latencies else 0.0,
            collision_occurred=False,  # No collision detection
            collision_tick=None,  # No collision detection
            final_collision_probability=0.0
        )
        
        print(f"TUN/TAP simulation completed. Position messages sent: {total_steps}, "
              f"Echoes received: {len(measurements)}, Out-of-order: {out_of_order_count}")
        
    except Exception as e:
        print(f"TUN/TAP simulation failed: {e}")
        import traceback
        traceback.print_exc()
        
        # Return empty results on failure
        results = ReliabilityResults(
            setup_type=setup_type,
            total_messages=0,
            out_of_order_count=0,
            timing_variance=0.0,
            mean_latency=0.0,
            max_latency=0.0,
            min_latency=0.0,
            collision_occurred=False,
            collision_tick=None,
            final_collision_probability=0.0
        )
        
    finally:
        # Cleanup
        if tuntap:
            tuntap.stop_server()
            tuntap.cleanup_tap_interface()
        if handler:
            handler.close()
        if server_thread and server_thread.is_alive():
            server_thread.join(timeout=1.0)
    
    return results


def create_aligned_comparison_plots(output_dir: str):
    """Create aligned comparison plots for FogSim and TUN/TAP with same scale and coordinates"""
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    
    # Load both comprehensive plots data
    fogsim_dir = os.path.join(output_dir, 'fogsim')
    tuntap_dir = os.path.join(output_dir, 'tuntap_basic')
    
    # Create a figure with two subplots side by side
    fig = plt.figure(figsize=(24, 10))
    gs = GridSpec(1, 2, figure=fig, wspace=0.15)
    
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    
    # Read position history from both simulations
    fogsim_history_file = os.path.join(fogsim_dir, 'position_history.json') if os.path.exists(os.path.join(fogsim_dir, 'position_history.json')) else None
    tuntap_history_file = os.path.join(tuntap_dir, 'position_history.json')
    
    # Determine common axis limits
    all_x = []
    all_y = []
    
    # Process TUN/TAP data
    if os.path.exists(tuntap_history_file):
        with open(tuntap_history_file, 'r') as f:
            tuntap_data = json.load(f)
            for entry in tuntap_data:
                all_x.append(entry['ego_pos'][0])
                all_y.append(entry['ego_pos'][1])
                all_x.append(entry['obstacle_pos'][0])
                all_y.append(entry['obstacle_pos'][1])
                if 'ekf_predictions' in entry and entry['ekf_predictions']:
                    for pred in entry['ekf_predictions']:
                        all_x.append(pred[0])
                        all_y.append(pred[1])
    
    # Calculate common bounds
    if all_x and all_y:
        x_min, x_max = min(all_x), max(all_x)
        y_min, y_max = min(all_y), max(all_y)
        
        # Add margin
        x_margin = (x_max - x_min) * 0.1
        y_margin = (y_max - y_min) * 0.1
        
        common_xlim = (x_min - x_margin, x_max + x_margin)
        common_ylim = (y_min - y_margin, y_max + y_margin)
    else:
        common_xlim = (-150, 150)
        common_ylim = (-150, 150)
    
    # Plot TUN/TAP data on first subplot
    ax1.set_xlim(common_xlim)
    ax1.set_ylim(common_ylim)
    ax1.set_xlabel('X Position (m)', fontsize=12)
    ax1.set_ylabel('Y Position (m)', fontsize=12)
    ax1.set_title('TUN/TAP: Vehicle Trajectories and EKF Predictions', fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')
    
    # Plot FogSim data on second subplot  
    ax2.set_xlim(common_xlim)
    ax2.set_ylim(common_ylim)
    ax2.set_xlabel('X Position (m)', fontsize=12)
    ax2.set_ylabel('Y Position (m)', fontsize=12)
    ax2.set_title('FogSim: Vehicle Trajectories and EKF Predictions', fontsize=14)
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect('equal')
    
    # Save aligned comparison plot
    comparison_filename = os.path.join(output_dir, 'aligned_trajectory_comparison.png')
    plt.savefig(comparison_filename, dpi=300, bbox_inches='tight')
    print(f"Aligned comparison plot saved to {comparison_filename}")
    plt.close()


def analyze_reliability_comparison(results: List[ReliabilityResults], output_dir: str):
    """Analyze and compare reliability results across setups"""
    print("\n" + "="*80)
    print("RELIABILITY COMPARISON ANALYSIS")
    print("="*80)
    
    comparison_data = {}
    
    for result in results:
        print(f"\n{result.setup_type.upper()} Results:")
        print(f"  Total Messages: {result.total_messages}")
        print(f"  Out-of-Order Messages: {result.out_of_order_count}")
        print(f"  Out-of-Order Rate: {result.out_of_order_count/result.total_messages*100:.2f}%")
        print(f"  Mean Latency: {result.mean_latency*1000:.2f} ms")
        print(f"  Timing Variance: {result.timing_variance*1000000:.2f} ms²")
        print(f"  Min Latency: {result.min_latency*1000:.2f} ms")
        print(f"  Max Latency: {result.max_latency*1000:.2f} ms")
        print(f"  Collision Occurred: {result.collision_occurred}")
        if result.collision_occurred:
            print(f"  Collision Tick: {result.collision_tick}")
        
        comparison_data[result.setup_type] = {
            'out_of_order_rate': result.out_of_order_count/result.total_messages*100 if result.total_messages > 0 else 0,
            'mean_latency_ms': result.mean_latency*1000,
            'variance_ms2': result.timing_variance*1000000,
            'collision_occurred': result.collision_occurred
        }
    
    # Calculate reliability metrics
    print(f"\n{'='*80}")
    print("RELIABILITY COMPARISON SUMMARY")
    print(f"{'='*80}")
    
    fogsim_data = comparison_data.get('fogsim', {})
    tuntap_basic_data = comparison_data.get('tuntap_basic', {})
    tuntap_busy_data = comparison_data.get('tuntap_busy', {})
    
    print(f"Out-of-Order Message Rates:")
    print(f"  FogSim: {fogsim_data.get('out_of_order_rate', 0):.2f}%")
    print(f"  TUN/TAP Basic: {tuntap_basic_data.get('out_of_order_rate', 0):.2f}%")
    print(f"  TUN/TAP Busy: {tuntap_busy_data.get('out_of_order_rate', 0):.2f}%")
    
    print(f"\nTiming Variance (ms²):")
    print(f"  FogSim: {fogsim_data.get('variance_ms2', 0):.2f}")
    print(f"  TUN/TAP Basic: {tuntap_basic_data.get('variance_ms2', 0):.2f}")
    print(f"  TUN/TAP Busy: {tuntap_busy_data.get('variance_ms2', 0):.2f}")
    
    print(f"\nCollision Results:")
    print(f"  FogSim: {'Yes' if fogsim_data.get('collision_occurred') else 'No'}")
    print(f"  TUN/TAP Basic: {'Yes' if tuntap_basic_data.get('collision_occurred') else 'No'}")
    print(f"  TUN/TAP Busy: {'Yes' if tuntap_busy_data.get('collision_occurred') else 'No'}")
    
    # Save detailed results to JSON
    results_file = os.path.join(output_dir, 'reliability_comparison_results.json')
    with open(results_file, 'w') as f:
        json.dump(comparison_data, f, indent=2)
    
    print(f"\nDetailed results saved to: {results_file}")


def get_common_plot_bounds(trajectory_files):
    """Calculate common bounds for all trajectories to ensure aligned plots"""
    all_x = []
    all_y = []
    
    for traj_file in trajectory_files:
        if os.path.exists(traj_file):
            trajectory = load_trajectory(traj_file)
            for point in trajectory:
                all_x.append(point[0])
                all_y.append(point[1])
    
    if all_x and all_y:
        x_min, x_max = min(all_x), max(all_x)
        y_min, y_max = min(all_y), max(all_y)
        
        # Add margin
        x_margin = max(10, (x_max - x_min) * 0.15)
        y_margin = max(10, (y_max - y_min) * 0.15)
        
        return (x_min - x_margin, x_max + x_margin, y_min - y_margin, y_max + y_margin)
    else:
        return (-150, 150, -150, 150)


def main():
    """Main function to run all three setups and compare results"""
    # Create output directory
    output_dir = "reliability_comparison_output"
    os.makedirs(output_dir, exist_ok=True)
    
    # Use the existing collision scenario configuration
    config = unprotected_right_turn_config.copy()
    
    print("Starting FogSim vs TUN/TAP Reliability Comparison Study")
    print("="*80)
    
    results = []
    
    try:
        # Run FogSim simulation (baseline)
        fogsim_result = run_fogsim_simulation(config, output_dir)
        results.append(fogsim_result)
        
        # Run TUN/TAP basic simulation
        tuntap_basic_result = run_tuntap_simulation(config, output_dir, busy_traffic=False)
        results.append(tuntap_basic_result)
        
        # Run TUN/TAP with busy traffic simulation
        tuntap_busy_result = run_tuntap_simulation(config, output_dir, busy_traffic=True)
        results.append(tuntap_busy_result)
        
        # Analyze and compare results
        analyze_reliability_comparison(results, output_dir)
        
        # Create aligned comparison plots
        create_aligned_comparison_plots(output_dir)
        
    except Exception as e:
        print(f"Error during simulation: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()