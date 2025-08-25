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

# FogSim imports
from fogsim import Env, NetworkConfig
from fogsim.handlers import BaseHandler

# Import the collision handler from main.py
from main import CollisionHandler, unprotected_right_turn_config


@dataclass
class TimingMeasurement:
    """Container for timing measurements"""
    send_time: float
    receive_time: float
    sequence_id: int
    message_type: str  # 'observation' or 'action'
    setup_type: str   # 'fogsim', 'tuntap_basic', 'tuntap_busy'


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
        self.localhost_mode = False
        
        # Timing tracking
        self.sent_messages = {}  # sequence_id -> send_time
        self.received_messages = []  # List of TimingMeasurement
        self.sequence_counter = 0
        
    def setup_tap_interface(self):
        """Create and configure TAP interface using sudo"""
        try:
            # Create TAP interface using sudo
            subprocess.run([
                'sudo', 'ip', 'tuntap', 'add', 'dev', self.interface_name, 'mode', 'tap'
            ], check=True, capture_output=True)
            
            # Bring interface up
            subprocess.run([
                'sudo', 'ip', 'link', 'set', self.interface_name, 'up'
            ], check=True, capture_output=True)
            
            # Assign IP address
            subprocess.run([
                'sudo', 'ip', 'addr', 'add', '192.168.100.1/24', 'dev', self.interface_name
            ], check=True, capture_output=True)
            
            print(f"TAP interface {self.interface_name} created successfully with sudo")
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"Failed to create TAP interface with sudo: {e}")
            print("Falling back to localhost sockets...")
            return self.setup_localhost_sockets()
    
    def setup_localhost_sockets(self):
        """Fallback: Use localhost UDP sockets instead of TAP"""
        print("Using localhost UDP sockets (fallback mode)")
        self.localhost_mode = True
        return True
    
    def cleanup_tap_interface(self):
        """Remove TAP interface"""
        if not self.localhost_mode:
            try:
                subprocess.run([
                    'sudo', 'ip', 'tuntap', 'del', 'dev', self.interface_name, 'mode', 'tap'
                ], check=True, capture_output=True)
                print(f"TAP interface {self.interface_name} removed with sudo")
            except subprocess.CalledProcessError as e:
                print(f"Failed to remove TAP interface: {e}")
    
    def start_server(self, port: int = 8888):
        """Start UDP server for receiving observations and sending actions"""
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        
        # Use localhost if in fallback mode
        bind_ip = '127.0.0.1' if self.localhost_mode else '192.168.100.1'
        self.server_socket.bind((bind_ip, port))
        self.server_socket.settimeout(1.0)  # Non-blocking with timeout
        
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
                time.sleep(0.001)
            except:
                pass
        
        busy_socket.close()
    
    def send_observation(self, obs: np.ndarray) -> int:
        """Send observation and return sequence ID"""
        self.sequence_counter += 1
        seq_id = self.sequence_counter
        
        # Pack data: sequence_id (4 bytes) + timestamp (8 bytes) + observation data
        send_time = time.time()
        data = struct.pack('!I', seq_id) + struct.pack('!d', send_time)
        data += obs.astype(np.float32).tobytes()
        
        self.sent_messages[seq_id] = send_time
        
        try:
            self.client_socket.sendto(data, (self.bind_ip, 8888))
        except Exception as e:
            print(f"Failed to send observation: {e}")
        
        return seq_id
    
    def receive_action(self) -> Tuple[Optional[int], Optional[float]]:
        """Receive action response, return (action, latency) or (None, None)"""
        try:
            data, addr = self.server_socket.recvfrom(1024)
            receive_time = time.time()
            
            if len(data) >= 12:  # seq_id + send_time + action
                seq_id = struct.unpack('!I', data[:4])[0]
                send_time = struct.unpack('!d', data[4:12])[0]
                action = struct.unpack('!I', data[12:16])[0] if len(data) >= 16 else 0
                
                # Record timing measurement
                if seq_id in self.sent_messages:
                    measurement = TimingMeasurement(
                        send_time=self.sent_messages[seq_id],
                        receive_time=receive_time,
                        sequence_id=seq_id,
                        message_type='action',
                        setup_type='tuntap_busy' if self.busy_traffic else 'tuntap_basic'
                    )
                    self.received_messages.append(measurement)
                    
                    latency = receive_time - self.sent_messages[seq_id]
                    del self.sent_messages[seq_id]
                    return action, latency
                
        except socket.timeout:
            pass
        except Exception as e:
            print(f"Error receiving action: {e}")
        
        return None, None
    
    def mock_decision_server(self):
        """Mock server that processes observations and returns actions"""
        while self.running:
            try:
                data, addr = self.server_socket.recvfrom(1024)
                receive_time = time.time()
                
                if len(data) >= 12:
                    seq_id = struct.unpack('!I', data[:4])[0]
                    send_time = struct.unpack('!d', data[4:12])[0]
                    
                    # Extract observation data
                    obs_data = data[12:]
                    obs = np.frombuffer(obs_data, dtype=np.float32)
                    
                    # Simple collision avoidance logic (mock)
                    # If obstacle is close (distance < threshold), brake
                    if len(obs) >= 2:
                        distance = np.sqrt(obs[0]**2 + obs[1]**2)
                        action = 1 if distance < 10.0 else 0  # Brake if close
                    else:
                        action = 0
                    
                    # Add artificial processing delay (simulate computation)
                    time.sleep(0.001)  # 1ms processing delay
                    
                    # Send response
                    response_data = data[:12] + struct.pack('!I', action)  # Echo seq_id + send_time + action
                    self.client_socket.sendto(response_data, addr)
                    
            except socket.timeout:
                continue
            except Exception as e:
                print(f"Error in decision server: {e}")
                break


class TunTapCollisionHandler(CollisionHandler):
    """Extended CollisionHandler that uses TUN/TAP interface for communication"""
    
    def __init__(self, config, output_dir, tuntap_interface: TunTapInterface):
        # Initialize the parent CollisionHandler
        super().__init__(config, output_dir)
        self.tuntap = tuntap_interface
        
        # Additional timing measurements for TUN/TAP
        self.tuntap_timing_measurements = []
        self.received_actions_queue = deque()
        
    def get_states(self):
        """Override get_states to use TUN/TAP communication"""
        # Get the normal observation from parent class
        states = super().get_states()
        observation = states['observation']
        
        # Send observation via TUN/TAP and measure timing
        seq_id = self.tuntap.send_observation(observation)
        
        # Store the original observation for later use
        self.last_observation = observation
        self.last_seq_id = seq_id
        
        return states
    
    def set_states(self, states=None, action=None):
        """Override set_states to get action from TUN/TAP"""
        # Try to receive action from TUN/TAP
        received_action, latency = self.tuntap.receive_action()
        
        if received_action is not None and latency is not None:
            # Record timing measurement
            self.tuntap_timing_measurements.append({
                'tick': self.tick,
                'sequence_id': getattr(self, 'last_seq_id', 0),
                'latency': latency,
                'timestamp': time.time(),
                'observation': getattr(self, 'last_observation', np.array([0,0,0])).tolist(),
                'action': received_action
            })
            
            # Use the received action
            action = received_action
        
        # Call parent set_states with the action (from TUN/TAP or parameter)
        super().set_states(states, action)
    
    def step_with_action(self, action):
        """Override step_with_action to handle TUN/TAP timing"""
        # Send current observation via TUN/TAP first
        states = self.get_states()
        
        # Try to get action from TUN/TAP
        received_action, latency = self.tuntap.receive_action()
        
        if received_action is not None:
            action = received_action
            
            if latency is not None:
                self.tuntap_timing_measurements.append({
                    'tick': self.tick,
                    'latency': latency,
                    'timestamp': time.time(),
                    'action': received_action
                })
        
        # Use the parent's step_with_action
        return super().step_with_action(action)


def run_fogsim_simulation(config: Dict, output_dir: str) -> ReliabilityResults:
    """Run simulation using FogSim (baseline)"""
    print("Running FogSim simulation...")
    
    try:
        # Use the existing FogSim function from main.py
        from main import run_adaptive_simulation_fogsim
        
        # Create dedicated output directory
        fogsim_output_dir = os.path.join(output_dir, 'fogsim')
        os.makedirs(fogsim_output_dir, exist_ok=True)
        
        # Run the FogSim simulation
        has_collided, final_delta_k = run_adaptive_simulation_fogsim(config, fogsim_output_dir)
        
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
    """Run simulation using TUN/TAP interface"""
    setup_type = 'tuntap_busy' if busy_traffic else 'tuntap_basic'
    print(f"Running TUN/TAP simulation (busy_traffic={busy_traffic})...")
    
    # Create TUN/TAP interface
    tuntap = TunTapInterface(
        interface_name=f"tap_fogsim_{setup_type}",
        busy_traffic=busy_traffic
    )
    
    handler = None
    server_thread = None
    
    try:
        # Setup TAP interface (this will fallback to localhost if needed)
        if not tuntap.setup_tap_interface():
            print("Failed to setup TAP interface, aborting TUN/TAP test")
            return ReliabilityResults(
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
        
        # Create dedicated output directory
        tuntap_output_dir = os.path.join(output_dir, setup_type)
        os.makedirs(tuntap_output_dir, exist_ok=True)
        
        # Start server
        tuntap.start_server()
        
        # Start decision server in separate thread
        server_thread = threading.Thread(target=tuntap.mock_decision_server)
        server_thread.daemon = True
        server_thread.start()
        
        # Give server time to start
        time.sleep(0.1)
        
        # Create extended collision handler
        handler = TunTapCollisionHandler(config, tuntap_output_dir, tuntap)
        handler.launch()
        
        # Run simulation similar to CollisionHandler step_with_action pattern
        total_steps = (config['ego_vehicle']['go_straight_ticks'] +
                      config['ego_vehicle']['turn_ticks'] +
                      config['ego_vehicle']['after_turn_ticks'])
        
        print(f"Running {total_steps} simulation steps...")
        collision_occurred = False
        collision_tick = None
        
        for step in range(total_steps):
            if step % 100 == 0:
                print(f"Step {step}/{total_steps}")
            
            # Use the step_with_action method which handles TUN/TAP communication
            obs, reward, success, termination, timeout, info = handler.step_with_action(0)
            
            # Check for collision
            if handler.has_collided:
                collision_occurred = True
                collision_tick = handler.tick
                print(f"Collision detected at step {step}, tick {handler.tick}")
                break
        
        # Analyze timing measurements from both TUN/TAP interface and handler
        measurements = tuntap.received_messages + [
            TimingMeasurement(
                send_time=m['timestamp'] - m['latency'],
                receive_time=m['timestamp'],
                sequence_id=m.get('sequence_id', 0),
                message_type='action',
                setup_type=setup_type
            ) for m in handler.tuntap_timing_measurements
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
            collision_occurred=collision_occurred,
            collision_tick=collision_tick,
            final_collision_probability=0.0
        )
        
        print(f"TUN/TAP simulation completed. Messages: {len(measurements)}, "
              f"Out-of-order: {out_of_order_count}, Collision: {collision_occurred}")
        
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
            if not tuntap.localhost_mode:
                tuntap.cleanup_tap_interface()
        if handler:
            handler.close()
        if server_thread and server_thread.is_alive():
            server_thread.join(timeout=1.0)
    
    return results


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
        
    except Exception as e:
        print(f"Error during simulation: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()