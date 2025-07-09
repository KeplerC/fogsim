# FogSim

FogSim is a co-simulation framework that enables integration between robotics simulation environments (Gym, CARLA, Mujoco, etc.) and network simulation to study the effects of network latency and bandwidth constraints on robotic systems.

## Key Features

- **Handler-based Architecture**: Unified interface supporting Gym, CARLA, and Mujoco simulators
- **Rich Network Simulation**: Powered by ns.py with configurable topologies, congestion control, and scheduling
- **Mujoco/Roboverse Compatible**: Standard interface familiar to robotics researchers  
- **Network Configuration**: Easy-to-use configuration system exposing advanced network simulation features
- **Comprehensive Testing**: Full test suite with unit and integration tests

## Installation

```bash
# Basic installation
pip install fogsim

# With optional dependencies
pip install 'fogsim[gym]'      # For OpenAI Gym support
pip install 'fogsim[carla]'    # For CARLA support  
pip install 'fogsim[all]'      # All optional dependencies

# For development
pip install -e ".[dev]"
```

## Quick Start

### Basic Usage (No Network)

```python
import fogsim
import numpy as np

# Create a Gym handler
handler = fogsim.GymHandler(env_name="CartPole-v1")

# Create FogSim environment
env = fogsim.Env(handler, enable_network=False)

# Standard gym-like interface
observation, extra_info = env.reset()

for step in range(100):
    action = np.random.choice([0, 1])  # Random action
    observation, reward, success, termination, timeout, extra_info = env.step(action)
    
    if termination or timeout:
        break

env.close()
```

### With Network Simulation

```python
import fogsim

# Create handler and network configuration
handler = fogsim.GymHandler(env_name="CartPole-v1")
network_config = fogsim.get_low_latency_config()  # Pre-configured for 5G/edge

# Create environment with network simulation
env = fogsim.Env(handler, network_config, enable_network=True)

# Reset and run with network effects
observation, extra_info = env.reset()

for step in range(100):
    action = np.random.choice([0, 1])
    observation, reward, success, termination, timeout, extra_info = env.step(action)
    
    # Check network latencies
    if extra_info.get('network_latencies'):
        print(f"Network latencies: {extra_info['network_latencies']}")
    
    if termination or timeout:
        break

env.close()
```

### Custom Network Configuration

```python
import fogsim

# Create custom network configuration
network_config = fogsim.NetworkConfig()
network_config.source_rate = 1e6  # 1 Mbps
network_config.topology.link_delay = 0.1  # 100ms delay
network_config.packet_loss_rate = 0.01  # 1% packet loss

# Enable traffic shaping
network_config.enable_token_bucket_shaping(rate=500e3, size=10e3)

# Use different congestion control
network_config.congestion_control = fogsim.CongestionControl.BBR

handler = fogsim.GymHandler(env_name="CartPole-v1")
env = fogsim.Env(handler, network_config, enable_network=True)
```

## Supported Simulators

### OpenAI Gym

```python
# Basic Gym environment
handler = fogsim.GymHandler(env_name="CartPole-v1")

# With custom render mode
handler = fogsim.GymHandler(env_name="CartPole-v1", render_mode="rgb_array")

# With pre-created environment
import gym
gym_env = gym.make("CartPole-v1")
handler = fogsim.GymHandler(env=gym_env)
```

### CARLA

```python
# CARLA with default settings
handler = fogsim.CarlaHandler()

# Custom CARLA configuration
handler = fogsim.CarlaHandler(
    host='localhost',
    port=2000,
    synchronous=True,
    fixed_delta_seconds=0.05,
    render_mode='camera'
)
```

### Mujoco

```python
# Mujoco with XML file
handler = fogsim.MujocoHandler(model_path="path/to/model.xml")

# Mujoco with XML string
xml_model = '''<mujoco>...</mujoco>'''
handler = fogsim.MujocoHandler(model_xml=xml_model)

# With rendering
handler = fogsim.MujocoHandler(
    model_path="model.xml",
    render_mode="rgb_array",
    render_width=640,
    render_height=480
)
```

## Network Configuration Options

### Pre-configured Networks

```python
# Low latency (5G/edge computing)
config = fogsim.get_low_latency_config()

# Satellite network (high latency)
config = fogsim.get_satellite_config()

# IoT network (low bandwidth, variable latency)
config = fogsim.get_iot_config()
```

### Custom Configurations

```python
config = fogsim.NetworkConfig()

# Topology
config.set_fattree_topology(k=4)  # Fat-tree topology
config.set_internet_topology("Abilene")  # Real-world topology

# Scheduling algorithms
config.scheduler = fogsim.SchedulerType.WFQ  # Weighted Fair Queuing
config.scheduler = fogsim.SchedulerType.DRR  # Deficit Round Robin

# Congestion control
config.congestion_control = fogsim.CongestionControl.BBR
config.congestion_control = fogsim.CongestionControl.CUBIC

# Traffic shaping
config.enable_token_bucket_shaping(rate=1e6, size=10e3)

# Per-flow configuration
config.add_flow(flow_id=0, weight=2, priority=1)
config.add_flow(flow_id=1, weight=1, priority=0)
```

## Examples

See the `examples/` directory for complete examples:

- `fogsim_basic_example.py` - Basic usage with Gym
- `carla_monte_carlo/` - Advanced CARLA simulation
- `network_config_example.py` - Network configuration showcase

## Legacy Compatibility

FogSim maintains backward compatibility with the original co-simulator interface:

```python
from fogsim import GymCoSimulator, NSPyNetworkSimulator

# Legacy interface still works
network_sim = NSPyNetworkSimulator()
co_sim = GymCoSimulator(network_sim, gym_env)
```

## Development

### Running Tests

```bash
# Run all tests
pytest tests/

# Run specific test categories
pytest tests/unit/
pytest tests/integration/

# With coverage
pytest tests/ --cov=fogsim --cov-report=html
```

### Code Quality

```bash
# Format code
black fogsim/ tests/ examples/

# Type checking
mypy fogsim/

# Linting
flake8 fogsim/ tests/ examples/
```

## Architecture

FogSim follows a modular architecture:

- **Env**: Main environment interface (Mujoco/Roboverse compatible)
- **Handlers**: Simulator-specific implementations (Gym, CARLA, Mujoco)
- **Network**: Network simulation and configuration
- **Legacy**: Backward compatibility layer

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality  
4. Ensure all tests pass
5. Submit a pull request

## License

MIT License - see LICENSE file for details.

## Citation

If you use FogSim in your research, please cite:

```bibtex
@misc{fogsim2024,
  title={FogSim: A Co-Simulation Framework for Robotics and Network Simulation},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/fogsim}
}
```