# Co-Simulator

A Python package for co-simulation between robotics and network simulation environments.

## Features

- Integration with popular robotics simulators (Gym, Carla)
- Network simulation support via NS3
- Message scheduling and latency simulation
- Extensible architecture for adding new simulators

## Installation

```bash
# Install the package
pip install cosimulator

# For development
pip install -e ".[dev]"
```

## Usage

### Gym Environment Example

```python
import gym
from cosimulator import GymCoSimulator
from cosimulator.network import NS3NetworkSimulator

# Create network simulator
network_sim = NS3NetworkSimulator()

# Create Gym environment
env = gym.make('CartPole-v1')

# Create co-simulator
co_sim = GymCoSimulator(network_sim, env)

# Run simulation
observation = co_sim.reset()
done = False

while not done:
    action = env.action_space.sample()
    observation, reward, done, info = co_sim.step(action)
    co_sim.render()

co_sim.close()
```

### Carla Environment Example

```python
import carla
from cosimulator import CarlaCoSimulator
from cosimulator.network import NS3NetworkSimulator

# Create network simulator
network_sim = NS3NetworkSimulator()

# Create Carla environment
client = carla.Client('localhost', 2000)
world = client.get_world()

# Create co-simulator
co_sim = CarlaCoSimulator(network_sim, world)

# Run simulation
observation = co_sim.reset()
done = False

while not done:
    action = np.array([0.0, 0.5])  # Example action
    observation, reward, done, info = co_sim.step(action)
    co_sim.render()

co_sim.close()
```

## Visualization Features

The visualization system provides real-time monitoring of the simulation through a web interface. Key features include:

### Real-time Metrics Display
- Collision probability monitoring
- Current simulation step
- Vehicle positions (ego and obstacle)
- Metrics update in real-time as the simulation runs

### Frame Visualization
- Real-time display of simulation frames
- Support for multiple simultaneous simulations
- Frame rate monitoring
- Automatic frame format conversion and optimization

### Control Panel
- Network parameter adjustment
  - Latency control
  - Packet loss simulation
  - Bandwidth limitation
- Simulation controls
  - Enable/disable rendering
  - Reset simulation
  - Emergency braking based on collision probability

### Multi-simulation Support
- View multiple simulations simultaneously
- Switch between different simulations
- Independent metrics tracking for each simulation

### Emergency Braking System
- Real-time collision probability calculation
- Automatic emergency braking when collision probability exceeds threshold (1.1)
- Visual feedback of collision risk in metrics display

### Usage
1. Start the visualization server:
```bash
python -m cloudsim.visualization.visualization_server
```

2. Run a simulation with visualization:
```bash
python examples/carla_visualization_example.py
```

3. Access the visualization interface at `http://localhost:5000`

The visualization interface will automatically connect to running simulations and display their metrics and frames in real-time.

## Development

### Running Tests

```bash
pytest tests/
```

### Code Style

```bash
# Format code
black .

# Check code style
flake8

# Type checking
mypy .
```

## License

MIT License 