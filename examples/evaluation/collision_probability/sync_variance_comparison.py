#!/usr/bin/env python3
"""
Clean script to compare the variance in collision tick numbers between 
synchronous and asynchronous CARLA modes using the configurable sync_mode parameter.
"""

import os
import subprocess
import statistics
import json
import time
import csv
from pathlib import Path
import threading
import queue
from concurrent.futures import ThreadPoolExecutor, as_completed
import docker

class CarlaDockerManager:
    """Manages CARLA Docker containers for parallel execution"""
    
    def __init__(self, base_port=2000, max_instances=4):  # Support multiple parallel containers
        self.base_port = base_port
        self.max_instances = max_instances
        self.docker_client = docker.from_env()
        self.active_containers = {}
        self.port_lock = threading.Lock()
        self.used_ports = set()
        self.available_ports = list(range(base_port, base_port + max_instances * 10, 10))  # Use ports with spacing
        
        # Clean up any existing CARLA containers before starting
        print("Performing initial cleanup of existing CARLA containers...")
        self.cleanup_all_carla_containers()
    
    def start_carla_container(self, port):
        """Start a CARLA docker container on specified port"""
        try:
            # Check if port is available
            if not self.check_port_availability(port):
                print(f"Port {port} is already in use")
                return False
            
            container_name = f"carla_sim_{port}_{int(time.time())}"  # Add timestamp for uniqueness
            
            # Clean up any existing containers with similar names
            self._cleanup_existing_containers(port)
            
            # Start new container with host networking (no port mapping needed)
            print(f"Creating CARLA container {container_name}...")
            container = self.docker_client.containers.run(
                "carlasim/carla:0.9.15",
                name=container_name,
                detach=True,
                auto_remove=False,  # Keep container for debugging
                privileged=True,
                volumes={'/tmp/.X11-unix': {'bind': '/tmp/.X11-unix', 'mode': 'rw'}},
                network_mode="host",  # Use host networking - no port mapping needed
                command=f"/bin/bash ./CarlaUE4.sh -world-port={port} -RenderOffScreen -carla-rpc-port={port} -carla-streaming-port={port+1} -carla-secondary-port={port+2}",
                # environment={"DISPLAY": ":0"},  # Set display for rendering
                restart_policy={"Name": "no"}  # Don't restart automatically
            )
            
            self.active_containers[port] = container
            print(f"Started CARLA container {container_name} on port {port}")
            
            # Wait for CARLA to be ready and verify it's running
            print(f"Waiting for CARLA on port {port} to be ready...")
            
            # Check container status periodically during startup
            for i in range(4):  # Check 4 times over 20 seconds
                time.sleep(5)
                try:
                    container.reload()
                    print(f"Container {container_name} status: {container.status}")
                    
                    if container.status == 'exited':
                        # Container exited, get logs for debugging
                        logs = container.logs().decode('utf-8')[-500:]  # Last 500 chars
                        print(f"Container {container_name} exited. Last logs: {logs}")
                        return False
                    elif container.status != 'running':
                        print(f"Container {container_name} in unexpected state: {container.status}")
                        return False
                        
                except docker.errors.NotFound:
                    print(f"Container {container_name} disappeared during startup")
                    return False
                except Exception as e:
                    print(f"Error checking container {container_name}: {e}")
                    return False
            
            # Final status check
            try:
                container.reload()
                if container.status == 'running':
                    print(f"Container {container_name} is running successfully on port {port}")
                    return True
                else:
                    print(f"Container {container_name} final status: {container.status}")
                    return False
            except Exception as e:
                print(f"Final check failed for container {container_name}: {e}")
                return False
            
        except Exception as e:
            print(f"Failed to start CARLA container on port {port}: {e}")
            return False
    
    def _cleanup_existing_containers(self, port):
        """Clean up existing containers that might conflict"""
        try:
            # List all containers (running and stopped) that match our naming pattern
            all_containers = self.docker_client.containers.list(all=True)
            for container in all_containers:
                if f"carla_sim_{port}" in container.name:
                    try:
                        print(f"Cleaning up existing container: {container.name}")
                        container.stop()
                        container.remove()
                    except Exception as e:
                        print(f"Error cleaning up container {container.name}: {e}")
            time.sleep(2)
        except Exception as e:
            print(f"Error during container cleanup: {e}")
    
    def stop_carla_container(self, port):
        """Stop and remove CARLA container"""
        if port in self.active_containers:
            try:
                container = self.active_containers[port]
                
                # Check if container still exists
                try:
                    container.reload()
                    if container.status == 'running':
                        print(f"Stopping CARLA container on port {port}")
                        container.stop(timeout=10)
                    
                    # Always try to remove the container
                    container.remove(force=True)
                    print(f"Removed CARLA container on port {port}")
                    
                except docker.errors.NotFound:
                    print(f"Container on port {port} already removed")
                except Exception as e:
                    print(f"Error stopping/removing container on port {port}: {e}")
                    
                del self.active_containers[port]
                
            except Exception as e:
                print(f"Error accessing container on port {port}: {e}")
                # Still remove from active containers
                del self.active_containers[port]
        
        # Clean up any lingering containers
        self._cleanup_existing_containers(port)
    
    def get_available_port(self):
        """Get an available port for CARLA"""
        with self.port_lock:
            # Find an available port with proper spacing
            for port in self.available_ports:
                if port not in self.used_ports and port not in self.active_containers:
                    self.used_ports.add(port)
                    return port
            raise RuntimeError(f"No available ports. Used: {self.used_ports}, Active: {list(self.active_containers.keys())}")
    
    def return_port(self, port):
        """Return port to available pool"""
        with self.port_lock:
            self.used_ports.discard(port)
    
    def cleanup_all(self):
        """Stop all active containers and clean up everything"""
        print("Cleaning up all CARLA containers...")
        
        # Stop active containers
        for port in list(self.active_containers.keys()):
            self.stop_carla_container(port)
        
        # Clean up any remaining CARLA containers
        try:
            all_containers = self.docker_client.containers.list(all=True)
            for container in all_containers:
                if "carla_sim" in container.name:
                    try:
                        print(f"Force cleaning container: {container.name}")
                        container.stop(timeout=5)
                        container.remove()
                    except Exception as e:
                        print(f"Error force cleaning container {container.name}: {e}")
        except Exception as e:
            print(f"Error during final cleanup: {e}")
        
        self.active_containers.clear()
        print("Cleanup completed")
    
    def cleanup_all_carla_containers(self):
        """Clean up ALL CARLA containers system-wide"""
        try:
            # Get all containers (running and stopped)
            all_containers = self.docker_client.containers.list(all=True)
            carla_containers = []
            
            for container in all_containers:
                # Check if container is related to CARLA
                if ("carla" in container.name.lower() or 
                    "carlasim" in str(container.attrs.get('Config', {}).get('Image', '')).lower()):
                    carla_containers.append(container)
            
            if carla_containers:
                print(f"Found {len(carla_containers)} existing CARLA containers to clean up")
                
                for container in carla_containers:
                    try:
                        print(f"Cleaning up container: {container.name} (status: {container.status})")
                        if container.status == 'running':
                            container.stop(timeout=5)
                        container.remove(force=True)
                    except Exception as e:
                        print(f"Error cleaning container {container.name}: {e}")
                        # Try force remove
                        try:
                            container.remove(force=True)
                        except:
                            pass
                
                print("Waiting for cleanup to complete...")
                time.sleep(5)
            else:
                print("No existing CARLA containers found")
                
        except Exception as e:
            print(f"Error during CARLA container cleanup: {e}")
    
    def check_port_availability(self, port):
        """Check if a port is available for use"""
        import socket
        import subprocess
        
        # First check if any running containers are using this port
        try:
            running_containers = self.docker_client.containers.list()
            for container in running_containers:
                if f"carla" in container.name.lower():
                    # Check if this container might be using the port
                    if f"{port}" in str(container.attrs.get('Config', {}).get('Cmd', [])):
                        print(f"Port {port} is being used by running container: {container.name}")
                        print(f"Attempting to stop container {container.name}")
                        try:
                            container.stop(timeout=5)
                            container.remove(force=True)
                            print(f"Stopped and removed container {container.name}")
                        except Exception as cleanup_error:
                            print(f"Error cleaning up container {container.name}: {cleanup_error}")
                        return False
        except Exception as e:
            print(f"Error checking running containers: {e}")
        
        # Check for any CARLA processes using this port and kill them
        try:
            result = subprocess.run(['netstat', '-tlnp'], capture_output=True, text=True)
            for line in result.stdout.split('\n'):
                if f':{port}' in line and 'CarlaUE4' in line:
                    print(f"Found CARLA process using port {port}, attempting to kill it")
                    # Extract PID from netstat output
                    parts = line.split()
                    if len(parts) > 6:
                        pid_part = parts[6].split('/')
                        if len(pid_part) > 0 and pid_part[0].isdigit():
                            pid = pid_part[0]
                            try:
                                subprocess.run(['kill', '-9', pid], check=True)
                                print(f"Killed CARLA process {pid}")
                                time.sleep(2)  # Give it time to die
                            except Exception as kill_error:
                                print(f"Error killing process {pid}: {kill_error}")
        except Exception as e:
            print(f"Error checking for CARLA processes: {e}")
        
        # Then check if the port is actually bound
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                s.bind(('localhost', port))
                return True
        except OSError as e:
            print(f"Port {port} socket check failed: {e}")
            return False

def run_simulation(sync_mode=False, config_type='merge', run_id=0, carla_port=2000):
    """
    Run a single simulation with specified parameters.
    
    Args:
        sync_mode (bool): Whether to use synchronous mode
        config_type (str): Configuration type ('right_turn', 'left_turn', 'merge')
        run_id (int): Run identifier for output directory
        carla_port (int): CARLA server port
        
    Returns:
        dict: Results including collision tick if collision occurred
    """
    output_dir = f'./sync_comparison/{"sync" if sync_mode else "async"}/port_{carla_port}/run_{run_id}'
    
    # Prepare command
    # Use FogSim to ensure we're using CollisionHandler which has the sync_mode setting
    cmd = [
        'python', 'main.py',
        '--config_type', config_type,
        '--output_dir', output_dir,
        '--no_risk_eval',  # Disable risk evaluation for cleaner comparison
        '--use_fogsim',    # This ensures CollisionHandler is used
        # '--cautious_delta_k', '20',
        '--carla_port', str(carla_port),  # Add port specification
    ]
    
    if sync_mode:
        cmd.append('--sync_mode')
    
    try:
        # Run simulation
        print(f"  Running {'synchronous' if sync_mode else 'asynchronous'} mode, run {run_id}...")
        print(f"  Command: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)  # Reduced timeout
        
        if result.returncode != 0:
            print(f"    Error: {result.stderr[:100]}")
            return {'success': False, 'error': result.stderr[:100]}
        
        # Check for collision and extract tick
        collision_tick = extract_collision_tick(output_dir)
        
        return {
            'success': True,
            'collision_occurred': collision_tick is not None,
            'collision_tick': collision_tick,
            'output_dir': output_dir
        }
        
    except subprocess.TimeoutExpired as e:
        print(f"    Timeout after 60 seconds")
        print(f"    Partial stdout: {e.stdout[:200] if e.stdout else 'None'}")
        print(f"    Partial stderr: {e.stderr[:200] if e.stderr else 'None'}")
        return {'success': False, 'error': 'timeout'}
    except Exception as e:
        print(f"    Exception: {e}")
        return {'success': False, 'error': str(e)}

def run_parallel_simulations(simulations, docker_manager, max_workers=4):
    """
    Run multiple simulations in parallel using multiple CARLA instances
    
    Args:
        simulations (list): List of simulation parameters (sync_mode, config_type, run_id)
        docker_manager (CarlaDockerManager): Docker container manager
        max_workers (int): Maximum number of parallel workers
        
    Returns:
        list: Results from all simulations
    """
    results = []
    
    def run_single_simulation(sim_params):
        """Run a single simulation with container management"""
        sync_mode, config_type, run_id = sim_params
        carla_port = None
        max_retries = 3
        
        print(f"\n--- Starting simulation {run_id} ({'sync' if sync_mode else 'async'}) ---")
        
        for attempt in range(max_retries):
            try:
                # Get available port
                carla_port = docker_manager.get_available_port()
                
                print(f"Attempt {attempt + 1}/{max_retries}: Starting CARLA container on port {carla_port}")
                
                # Start CARLA container with retry
                if not docker_manager.start_carla_container(carla_port):
                    if attempt < max_retries - 1:
                        print(f"Container start failed, retrying in 10 seconds...")
                        if carla_port is not None:
                            docker_manager.return_port(carla_port)
                        time.sleep(10)
                        continue
                    return {'success': False, 'error': 'Failed to start CARLA container after retries', 'run_id': run_id}
                
                # Run simulation
                print(f"Running simulation {run_id} on port {carla_port}")
                result = run_simulation(sync_mode, config_type, run_id, carla_port)
                result['run_id'] = run_id
                result['carla_port'] = carla_port
                print(f"Completed simulation {run_id}: {'Success' if result.get('success', False) else 'Failed'}")
                return result
                
            except Exception as e:
                print(f"Attempt {attempt + 1} failed: {e}")
                
                if attempt < max_retries - 1:
                    if carla_port is not None:
                        docker_manager.return_port(carla_port)
                    time.sleep(10)  # Wait before retry
                else:
                    return {'success': False, 'error': f'Failed after {max_retries} attempts: {str(e)}', 'run_id': run_id}
            
            finally:
                # Always cleanup after each simulation
                if carla_port is not None:
                    docker_manager.stop_carla_container(carla_port)
                    docker_manager.return_port(carla_port)
                    time.sleep(2)  # Give container time to fully stop
        
        return {'success': False, 'error': 'Unknown error', 'run_id': run_id}
    
    # Use ThreadPoolExecutor for parallel execution
    with ThreadPoolExecutor(max_workers=min(max_workers, len(simulations))) as executor:
        print(f"Starting {len(simulations)} simulations in parallel with {min(max_workers, len(simulations))} workers...")
        
        # Submit all tasks
        futures = {executor.submit(run_single_simulation, sim): sim for sim in simulations}
        
        # Collect results as they complete
        for future in as_completed(futures):
            sim_params = futures[future]
            try:
                result = future.result()
                results.append(result)
                print(f"Simulation {result['run_id']} completed: {'Success' if result.get('success', False) else 'Failed'}")
            except Exception as e:
                sync_mode, config_type, run_id = sim_params
                print(f"Simulation {run_id} failed with exception: {e}")
                results.append({'success': False, 'error': str(e), 'run_id': run_id})
    
    return results

def run_sequential_simulations(simulations, docker_manager):
    """
    Run multiple simulations sequentially using a single CARLA instance
    
    Args:
        simulations (list): List of simulation parameters (sync_mode, config_type, run_id)
        docker_manager (CarlaDockerManager): Docker container manager
        
    Returns:
        list: Results from all simulations
    """
    results = []
    
    for sim_params in simulations:
        sync_mode, config_type, run_id = sim_params
        carla_port = None
        max_retries = 3
        
        print(f"\n--- Running simulation {run_id} ({'sync' if sync_mode else 'async'}) ---")
        
        for attempt in range(max_retries):
            try:
                # Get available port
                carla_port = docker_manager.get_available_port()
                
                print(f"Attempt {attempt + 1}/{max_retries}: Starting CARLA container on port {carla_port}")
                
                # Start CARLA container with retry
                if not docker_manager.start_carla_container(carla_port):
                    if attempt < max_retries - 1:
                        print(f"Container start failed, retrying in 10 seconds...")
                        time.sleep(10)
                        continue
                    results.append({'success': False, 'error': 'Failed to start CARLA container after retries', 'run_id': run_id})
                    break
                
                # Run simulation
                print(f"Running simulation {run_id} on port {carla_port}")
                result = run_simulation(sync_mode, config_type, run_id, carla_port)
                result['run_id'] = run_id
                result['carla_port'] = carla_port
                results.append(result)
                print(f"Completed simulation {run_id}: {'Success' if result.get('success', False) else 'Failed'}")
                break
                
            except Exception as e:
                print(f"Attempt {attempt + 1} failed: {e}")
                
                if attempt < max_retries - 1:
                    time.sleep(10)  # Wait before retry
                else:
                    results.append({'success': False, 'error': f'Failed after {max_retries} attempts: {str(e)}', 'run_id': run_id})
            
            finally:
                # Always cleanup after each simulation
                if carla_port is not None:
                    docker_manager.stop_carla_container(carla_port)
                    time.sleep(2)  # Give container time to fully stop
    
    return results

def extract_collision_tick(output_dir):
    """
    Extract collision tick from simulation output files.
    
    Args:
        output_dir (str): Directory containing simulation output
        
    Returns:
        int or None: Collision tick if found, None otherwise
    """
    # Check statistics file first
    stats_file = Path(output_dir) / 'monte_carlo_results' / 'statistics.csv'
    if stats_file.exists():
        try:
            with open(stats_file, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if 'collision' in row and row['collision'].lower() == 'true':
                        # Found collision, but need to get tick from elsewhere
                        pass
        except Exception:
            pass
    
    # Check collision probabilities file
    for file in Path(output_dir).glob('collision_probabilities*.csv'):
        try:
            with open(file, 'r') as f:
                lines = f.readlines()
                if len(lines) > 1:  # Has data beyond header
                    # Get last line before potential collision
                    last_line = lines[-1].strip()
                    if last_line:
                        parts = last_line.split(',')
                        if len(parts) >= 2:
                            tick = int(float(parts[1]))
                            # Heuristic: if simulation stopped early, collision likely occurred
                            # You may need to adjust this based on your simulation length
                            if tick < 900:  # Assuming normal simulation is ~1000 ticks
                                return tick + 1  # Collision likely at next tick
        except Exception:
            pass
    
    # Check simulation output text file if exists
    stats_txt = Path(output_dir) / 'monte_carlo_results' / 'statistics.txt'
    if stats_txt.exists():
        try:
            with open(stats_txt, 'r') as f:
                content = f.read()
                if 'Number of collisions: 1' in content:
                    # Collision occurred but couldn't extract exact tick
                    return -1  # Sentinel value for "collision occurred but tick unknown"
        except Exception:
            pass
    
    return None

def run_comparison(num_runs=10, config_types=['merge'], max_parallel=4):
    """
    Run comparison between synchronous and asynchronous modes.
    
    Args:
        num_runs (int): Number of runs for each mode
        config_types (list): List of configuration types to test
        max_parallel (int): Maximum number of parallel CARLA instances
        
    Returns:
        dict: Comparison results and statistics for all configurations
    """
    print(f"Starting Synchronous vs Asynchronous Mode Comparison")
    print(f"Configurations: {', '.join(config_types)}")
    print(f"Number of runs per mode: {num_runs}")
    print("=" * 60)
    
    # Clean up previous results
    os.system('rm -rf ./sync_comparison')
    
    # Initialize Docker manager
    docker_manager = CarlaDockerManager(base_port=2000, max_instances=max_parallel)
    
    try:
        all_results = {}
        
        for config_type in config_types:
            print(f"\n{'='*60}")
            print(f"Testing configuration: {config_type}")
            print(f"{'='*60}")
            
            os.makedirs(f'./sync_comparison/{config_type}/sync', exist_ok=True)
            os.makedirs(f'./sync_comparison/{config_type}/async', exist_ok=True)
            
            # Prepare simulation tasks for parallel execution
            async_simulations = [(False, config_type, i) for i in range(num_runs)]
            sync_simulations = [(True, config_type, i) for i in range(num_runs)]
            
            # Run asynchronous simulations in parallel
            print(f"\nRunning ASYNCHRONOUS mode simulations for {config_type} in parallel...")
            async_results = run_parallel_simulations(async_simulations, docker_manager, max_parallel)
            
            # Run synchronous simulations in parallel  
            print(f"\nRunning SYNCHRONOUS mode simulations for {config_type} in parallel...")
            sync_results = run_parallel_simulations(sync_simulations, docker_manager, max_parallel)
            
            # Extract collision ticks
            async_ticks = [r['collision_tick'] for r in async_results 
                           if r['success'] and r['collision_occurred'] and r['collision_tick'] > 0]
            sync_ticks = [r['collision_tick'] for r in sync_results 
                          if r['success'] and r['collision_occurred'] and r['collision_tick'] > 0]
            
            # Calculate statistics for this configuration
            print(f"\n{'-'*60}")
            print(f"RESULTS SUMMARY for {config_type}")
            print(f"{'-'*60}")
            
            config_results = {
                'config_type': config_type,
                'num_runs': num_runs,
                'async': {
                    'collision_ticks': async_ticks,
                    'num_collisions': len(async_ticks),
                    'success_runs': sum(1 for r in async_results if r['success'])
                },
                'sync': {
                    'collision_ticks': sync_ticks,
                    'num_collisions': len(sync_ticks),
                    'success_runs': sum(1 for r in sync_results if r['success'])
                }
            }
            
            print(f"\nAsynchronous Mode:")
            print(f"  Successful runs: {config_results['async']['success_runs']}/{num_runs}")
            print(f"  Collisions: {config_results['async']['num_collisions']}")
            if len(async_ticks) > 0:
                print(f"  Collision ticks: {async_ticks}")
            if len(async_ticks) > 1:
                config_results['async']['mean'] = statistics.mean(async_ticks)
                config_results['async']['stdev'] = statistics.stdev(async_ticks)
                config_results['async']['variance'] = statistics.variance(async_ticks)
                print(f"  Mean: {config_results['async']['mean']:.2f}")
                print(f"  Std Dev: {config_results['async']['stdev']:.2f}")
                print(f"  Variance: {config_results['async']['variance']:.2f}")
            
            print(f"\nSynchronous Mode:")
            print(f"  Successful runs: {config_results['sync']['success_runs']}/{num_runs}")
            print(f"  Collisions: {config_results['sync']['num_collisions']}")
            if len(sync_ticks) > 0:
                print(f"  Collision ticks: {sync_ticks}")
            if len(sync_ticks) > 1:
                config_results['sync']['mean'] = statistics.mean(sync_ticks)
                config_results['sync']['stdev'] = statistics.stdev(sync_ticks)
                config_results['sync']['variance'] = statistics.variance(sync_ticks)
                print(f"  Mean: {config_results['sync']['mean']:.2f}")
                print(f"  Std Dev: {config_results['sync']['stdev']:.2f}")
                print(f"  Variance: {config_results['sync']['variance']:.2f}")
            
            # Compare variances
            print(f"\nVariance Comparison:")
            if len(async_ticks) > 1 and len(sync_ticks) > 1:
                async_variance = config_results['async']['variance']
                sync_variance = config_results['sync']['variance']
                
                if sync_variance == 0.0:
                    # Handle zero variance case
                    if async_variance == 0.0:
                        print(f"  → Both modes have zero variance (all collisions at same tick)")
                        config_results['variance_ratio'] = 1.0
                    else:
                        print(f"  → Synchronous mode has zero variance (all collisions at same tick: {sync_ticks[0]})")
                        print(f"  → Asynchronous mode has variance: {async_variance:.2f}")
                        print(f"  → Synchronous mode is MORE CONSISTENT (zero variance vs {async_variance:.2f})")
                        config_results['variance_ratio'] = float('inf')  # Infinite ratio
                elif async_variance == 0.0:
                    print(f"  → Asynchronous mode has zero variance (all collisions at same tick: {async_ticks[0]})")
                    print(f"  → Synchronous mode has variance: {sync_variance:.2f}")
                    print(f"  → Asynchronous mode is MORE CONSISTENT (zero variance vs {sync_variance:.2f})")
                    config_results['variance_ratio'] = 0.0  # Zero ratio
                else:
                    # Normal case - both have non-zero variance
                    variance_ratio = async_variance / sync_variance
                    config_results['variance_ratio'] = variance_ratio
                    print(f"  Variance Ratio (Async/Sync): {variance_ratio:.3f}")
                    
                    if variance_ratio > 1.5:
                        print(f"  → Asynchronous mode shows SIGNIFICANTLY higher variance ({variance_ratio:.2f}x)")
                    elif variance_ratio > 1.1:
                        print(f"  → Asynchronous mode shows moderately higher variance ({variance_ratio:.2f}x)")
                    elif variance_ratio < 0.67:
                        print(f"  → Synchronous mode shows SIGNIFICANTLY higher variance ({1/variance_ratio:.2f}x)")
                    elif variance_ratio < 0.9:
                        print(f"  → Synchronous mode shows moderately higher variance ({1/variance_ratio:.2f}x)")
                    else:
                        print(f"  → Variances are similar (ratio ≈ 1)")
            else:
                print(f"  Insufficient data for variance comparison")
                print(f"  (Need at least 2 collision measurements for each mode)")
            
            all_results[config_type] = config_results
    
    finally:
        # Clean up all Docker containers
        docker_manager.cleanup_all()
    
    # Save all results to JSON
    results_file = './sync_comparison/comparison_results.json'
    final_results = {
        'num_runs': num_runs,
        'configurations': all_results
    }
    with open(results_file, 'w') as f:
        json.dump(final_results, f, indent=2)
    print(f"\nDetailed results saved to: {results_file}")
    
    return final_results

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Compare collision tick variance between sync and async CARLA modes')
    parser.add_argument('--num_runs', type=int, default=20,
                        help='Number of simulation runs per mode (default: 1)')
    parser.add_argument('--config_types', type=str, nargs='+',
                        choices=['right_turn', 'left_turn', 'merge'],
                        default=['merge'],
                        help='Scenario configuration types to test (default: merge)')
    parser.add_argument('--max_parallel', type=int, default=10,
                        help='Maximum number of parallel CARLA instances (default: 4)')
    
    args = parser.parse_args()
    
    try:
        results = run_comparison(args.num_runs, args.config_types, args.max_parallel)
        
        # Print final conclusion for all configurations
        print("\n" + "=" * 60)
        print("OVERALL CONCLUSION")
        print("=" * 60)
        
        for config_type, config_results in results['configurations'].items():
            print(f"\n{config_type.upper()}:")
            if 'variance_ratio' in config_results:
                ratio = config_results['variance_ratio']
                if ratio == float('inf'):
                    print(f"  → Synchronous mode provides PERFECT CONSISTENCY (zero variance)")
                    print(f"  → All synchronous collisions occur at the same tick")
                elif ratio == 0.0:
                    print(f"  → Asynchronous mode provides PERFECT CONSISTENCY (zero variance)")
                    print(f"  → All asynchronous collisions occur at the same tick")
                elif ratio > 1.5:
                    print(f"  → Synchronous mode provides MORE CONSISTENT collision timing ({ratio:.2f}x less variance)")
                elif ratio < 0.67:
                    print(f"  → Asynchronous mode provides MORE CONSISTENT collision timing ({1/ratio:.2f}x less variance)") 
                    print(f"     (This is unexpected - please verify simulation settings)")
                else:
                    print(f"  → Both modes show similar consistency (ratio: {ratio:.2f})")
            else:
                print(f"  → Unable to draw conclusions - insufficient collision data")
            
    except KeyboardInterrupt:
        print("\n\nComparison interrupted by user")
    except Exception as e:
        print(f"\n\nError during comparison: {e}")
        import traceback
        traceback.print_exc()