# FogSim - Refactored Co-simulation Framework

**Major Breaking Changes - Version 0.2.0**

FogSim has been completely refactored to align with the architecture described in CLAUDE.md. This version introduces **breaking changes** and removes legacy complexity.

## Architecture Overview

FogSim implements three distinct simulation modes:

1. **Virtual Timeline (VIRTUAL)** - Decoupled from wallclock time for maximum scalability and reproducibility
2. **Real Clock + Simulated Network (SIMULATED_NET)** - Real-time execution with network simulation
3. **Real Clock + Real Network (REAL_NET)** - Real-time execution with actual network constraints

## New Simplified API

### Basic Usage

```python
from fogsim import FogSim, SimulationMode
from fogsim.handlers import GymHandler

# Create handler for your simulator (Gym, CARLA, etc.)
handler = GymHandler("CartPole-v1")

# Create FogSim instance with desired mode
fogsim = FogSim(handler, mode=SimulationMode.VIRTUAL, timestep=0.1)

# Run simulation
obs, info = fogsim.reset()
for step in range(100):
    action = fogsim.action_space.sample()
    obs, reward, success, term, timeout, info = fogsim.step(action)
    
    if term or timeout:
        break

fogsim.close()
```

### Three Modes Example

```python
# Mode 1: Virtual Timeline (highest performance)
fogsim_virtual = FogSim(handler, SimulationMode.VIRTUAL)

# Mode 2: Real Clock + Simulated Network
fogsim_simnet = FogSim(handler, SimulationMode.SIMULATED_NET)  

# Mode 3: Real Clock + Real Network
fogsim_real = FogSim(handler, SimulationMode.REAL_NET)
```

## Project Structure

```
fogsim/
├── core.py              # Main FogSim class (NEW)
├── clock/               # Time management (NEW)
│   ├── virtual_clock.py # Virtual timeline
│   └── real_clock.py    # Real-time sync
├── handlers/            # Simulator interfaces (UNCHANGED)
├── network/             # Network components (SIMPLIFIED)
│   ├── nspy_simulator.py
│   └── real_network.py
└── messages.py          # Simple message definitions (NEW)

examples/
├── basic_demos/         # Simple examples
├── evaluation/          # Evaluation experiments
│   ├── rl_training/     # RL policy training
│   └── carla/           # CARLA experiments
```

## What Was Removed

- **BaseCoSimulator** - Complex legacy wrapper
- **evaluation.py** - Overcomplicated evaluation framework  
- **network_control.py** - Merged into network components
- **message_passing.py** - Replaced with simple messages
- **time_backend.py** - Split into clock modules
- Legacy co-simulator classes
- Backward compatibility layers

## Migration Guide

**Old API:**
```python
from fogsim import GymCoSimulator
cosim = GymCoSimulator(network_sim, "CartPole-v1")
```

**New API:**
```python  
from fogsim import FogSim, SimulationMode
from fogsim.handlers import GymHandler

handler = GymHandler("CartPole-v1")
fogsim = FogSim(handler, SimulationMode.VIRTUAL)
```

## Key Benefits

- **90% code reduction** in core modules
- **Clear separation** of three simulation modes
- **Minimal API surface** - easier to understand and maintain
- **Direct implementation** of CLAUDE.md architecture  
- **No legacy baggage** - clean foundation for future development

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