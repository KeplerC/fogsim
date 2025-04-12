from flask import Flask, render_template, jsonify, request
import os
import threading
import time
import logging
import numpy as np

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Global state to store simulation data
simulation_state = {
    'current_observation': None,
    'round_trip_latency': 0,
    'action_latencies': [],
    'observation_latencies': [],
    'simulation_frame': None,
    'current_risk': 0.15,
    'settings': {
        'road': {
            'weather': 'Clear',
            'traffic_density': 0.5
        },
        'network': {
            'latency': 50,
            'packet_loss': 2.0
        },
        'computation': {
            'processing_power': 0.6
        },
        'algorithms': {
            'perception': 'Algorithm A',
            'planning': 'Algorithm B'
        }
    }
}

# Simulator instance will be set by the runner
simulator = None
simulation_thread = None
running = False

@app.route('/')
def index():
    """Serve the main dashboard page."""
    return render_template('index.html')

@app.route('/api/state')
def get_state():
    """Return current simulation state as JSON."""
    return jsonify(simulation_state)

@app.route('/api/settings', methods=['POST'])
def update_settings():
    """Update simulation settings."""
    if request.method == 'POST':
        data = request.json
        
        # Update simulation settings
        if 'road' in data:
            simulation_state['settings']['road'].update(data['road'])
        if 'network' in data:
            simulation_state['settings']['network'].update(data['network'])
        if 'computation' in data:
            simulation_state['settings']['computation'].update(data['computation'])
        if 'algorithms' in data:
            simulation_state['settings']['algorithms'].update(data['algorithms'])
            
        # TODO: Apply settings to actual simulator
        if simulator:
            # Apply network settings
            if hasattr(simulator, 'network_simulator'):
                network_sim = simulator.network_simulator
                if hasattr(network_sim, 'set_latency'):
                    network_sim.set_latency(simulation_state['settings']['network']['latency'] / 1000.0)
                if hasattr(network_sim, 'set_loss_rate'):
                    network_sim.set_loss_rate(simulation_state['settings']['network']['packet_loss'] / 100.0)
                    
        return jsonify({'status': 'success'})
    
    return jsonify({'status': 'error', 'message': 'Invalid request'})

@app.route('/api/control', methods=['POST'])
def control_simulation():
    """Start/stop/reset the simulation."""
    global running, simulation_thread
    
    if request.method == 'POST':
        data = request.json
        command = data.get('command')
        
        if command == 'start' and simulator and not running:
            running = True
            simulation_thread = threading.Thread(target=run_simulation)
            simulation_thread.daemon = True
            simulation_thread.start()
            return jsonify({'status': 'success', 'message': 'Simulation started'})
            
        elif command == 'stop' and running:
            running = False
            if simulation_thread:
                simulation_thread.join(timeout=1.0)
            return jsonify({'status': 'success', 'message': 'Simulation stopped'})
            
        elif command == 'reset' and simulator:
            # Reset simulator
            running = False
            if simulation_thread:
                simulation_thread.join(timeout=1.0)
            
            # Reset the simulator
            simulator.reset()
            simulation_state['current_observation'] = None
            simulation_state['round_trip_latency'] = 0
            simulation_state['action_latencies'] = []
            simulation_state['observation_latencies'] = []
            simulation_state['simulation_frame'] = None
            
            return jsonify({'status': 'success', 'message': 'Simulation reset'})
    
    return jsonify({'status': 'error', 'message': 'Invalid request'})

def convert_frame_to_base64(frame):
    """Convert numpy frame to base64 encoded jpeg for display."""
    if frame is None:
        return None
    
    import cv2
    import base64
    
    # Convert the frame to jpeg format
    _, buffer = cv2.imencode('.jpg', cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    # Convert to base64 string
    return base64.b64encode(buffer).decode('utf-8')

def run_simulation():
    """Run the simulation loop in a separate thread."""
    global running, simulator, simulation_state
    
    if not simulator:
        logger.error("No simulator instance available")
        running = False
        return
    
    # Get initial observation
    observation = simulator.current_observation
    if observation is None:
        observation = simulator.reset()
        # Handle different return types from reset()
        if isinstance(observation, tuple):
            observation = observation[0]  # Extract observation from tuple
    
    step_count = 0
    
    while step_count < 1000:  # Limit steps as a safety measure
        try:
            # Simple policy for CartPole: move cart in direction of pole tilt
            # Adjust based on your environment
            if isinstance(observation, np.ndarray) and len(observation) >= 3:
                # For CartPole
                pole_angle = observation[2]
                action = 1 if pole_angle > 0 else 0
            else:
                # Default random action if environment is unknown
                action = simulator.robotics_simulator.action_space.sample()
            
            # Step the simulator
            step_result = simulator.step(action)
            
            # Handle different return formats
            if len(step_result) == 4:
                observation, reward, done, info = step_result
            else:
                observation, reward, terminated, truncated, info = step_result
                done = terminated or truncated
            
            # Update simulation state
            simulation_state['current_observation'] = observation.tolist() if isinstance(observation, np.ndarray) else observation
            
            # Get latency information
            if 'round_trip_latency' in info:
                simulation_state['round_trip_latency'] = info['round_trip_latency']
            if 'action_latencies' in info:
                simulation_state['action_latencies'] = info['action_latencies']
            if 'observation_latencies' in info:
                simulation_state['observation_latencies'] = info['observation_latencies']
            
            # Render frame
            frame = simulator.render()
            if frame is not None:
                simulation_state['simulation_frame'] = convert_frame_to_base64(frame)
            
            # Calculate risk (this is a placeholder, implement your risk calculation)
            simulation_state['current_risk'] = 0.15 + (0.05 * np.sin(step_count / 10.0))
            
            step_count += 1
            
            if done:
                logger.info(f"Simulation completed after {step_count} steps")
                running = False
                break
                
            # Add a small delay to prevent high CPU usage
            time.sleep(0.05)
            
        except Exception as e:
            logger.error(f"Error in simulation loop: {str(e)}")
            running = False
            break
    
    logger.info(f"Simulation thread exiting after {step_count} steps")
    running = False

def set_simulator(sim_instance):
    """Set the simulator instance for the web interface."""
    global simulator
    simulator = sim_instance
    logger.info(f"Simulator instance set: {type(simulator).__name__}")

def start_dashboard(sim_instance=None, host='0.0.0.0', port=5000, debug=False):
    """Start the dashboard Flask app."""
    global simulator
    
    if sim_instance:
        simulator = sim_instance
    
    app.run(host=host, port=port, debug=debug)

if __name__ == '__main__':
    # When run directly, start the app without a simulator (for development)
    app.run(debug=True) 