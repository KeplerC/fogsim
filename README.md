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