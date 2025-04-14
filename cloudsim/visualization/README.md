# CloudSim Visualization

A server and client system for visualizing co-simulations between robotics and network simulators.

## Features

- Web-based visualization dashboard
- Real-time display of simulation frames and metrics
- Support for multiple simultaneous simulations
- Interactive controls to adjust network parameters (latency, packet loss, bandwidth)
- Ability to enable/disable rendering
- Reset functionality
- Cross-platform support

## Components

The visualization system consists of three main components:

1. **Visualization Server**: A Flask and Socket.IO based server that receives frames and metrics from simulations and serves the web interface.
2. **Client Adapter**: A library that simulations use to connect to the visualization server and send data.
3. **Simulator Wrapper**: A wrapper for co-simulators that adds visualization capabilities.

## Installation

### Requirements

- Python 3.7 or higher
- Flask
- Flask-SocketIO
- python-socketio
- PIL (Pillow)
- NumPy

You can install the required packages with pip:

```bash
pip install flask flask-socketio python-socketio pillow numpy
```

## Usage

### Starting the Visualization Server

```python
from cloudsim.visualization.visualization_server import VisualizationServer

# Create and start the server
server = VisualizationServer(host='0.0.0.0', port=5000)
server.run()

# Or start it in a background thread
server_thread = server.run_in_thread()
```

### Using the Visualization Wrapper

The easiest way to use the visualization system is to wrap your co-simulator with the `VisualizationCoSimulator` class:

```python
from cloudsim.visualization.simulator_wrapper import VisualizationCoSimulator

# Assuming you already have a co-simulator instance
co_sim = GymCoSimulator(...)  # or CarlaCoSimulator

# Wrap it with visualization capabilities
viz_sim = VisualizationCoSimulator(
    co_simulator=co_sim,
    server_url='http://localhost:5000',
    simulation_id='my_simulation',
    auto_connect=True
)

# Use the wrapped simulator just like the original
observation = viz_sim.reset()
observation, reward, done, info = viz_sim.step(action)
```

### Using the Client Adapter Directly

If you want more control over the visualization process, you can use the client adapter directly:

```python
from cloudsim.visualization.client_adapter import VisualizationClientAdapter

# Create a client adapter
viz_client = VisualizationClientAdapter(
    server_url='http://localhost:5000',
    simulation_id='my_simulation',
    simulator_type='custom'
)

# Connect to the server
viz_client.connect()

# Register command handlers
viz_client.register_command_handler('my_command', my_handler_function)

# Send frames and metrics
viz_client.send_frame(my_frame)
viz_client.send_metrics({'metric1': value1, 'metric2': value2})

# Disconnect when done
viz_client.disconnect()
```

## Web Interface

The visualization web interface is available at `http://server_address:port/` (e.g., `http://localhost:5000/`).

### Interface Features

- **Simulation Selector**: Choose which simulation to view
- **Frame Display**: View the current frame from the selected simulation
- **Metrics Panel**: View real-time metrics for the selected simulation
- **Network Controls**: Adjust network parameters (latency, packet loss, bandwidth)
- **Rendering Toggle**: Enable or disable rendering
- **Reset Button**: Reset the selected simulation

## Examples

See the example scripts in the `examples/` directory:

- `visualization_example.py`: Example using a Gym environment
- `carla_visualization_example.py`: Example using a Carla environment

## Customization

### Custom Command Handlers

You can add custom command handlers to respond to commands from the visualization server:

```python
def my_custom_handler(params):
    # Process the command parameters
    value = params.get('some_value')
    
    # Do something with the value
    # ...
    
    # Return a result
    return {'status': 'success', 'result': some_result}

# Register the handler
viz_client.register_command_handler('my_custom_command', my_custom_handler)
```

### Custom Metrics

You can send any custom metrics to the visualization server:

```python
viz_client.send_metrics({
    'fps': 30.5,
    'reward': 10.0,
    'latency': 50.2,
    'custom_metric': my_value
})
```

## Troubleshooting

- If the connection fails, check that the server is running and accessible from the client.
- If frames aren't showing up, make sure your render function returns a valid image format (numpy array, PIL Image, or base64 string).
- Check the server logs for error messages.
- Make sure you're using compatible versions of Socket.IO on the client and server.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 