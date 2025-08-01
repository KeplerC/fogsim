# FogSim Examples

This directory contains examples demonstrating FogSim's capabilities as outlined in CLAUDE.md.

## Core Demonstrations

### 1. Three Modes Demo (`three_modes_demo.py`)
Demonstrates FogSim's three operational modes:
- **Virtual Timeline (FogSIM)**: Pure virtual time for highest performance
- **Real Clock + Simulated Network**: Wallclock with ns.py network simulation
- **Real Clock + Real Network**: Wallclock with Linux tc for real network control

```bash
# Run all modes comparison
python examples/three_modes_demo.py

# Run specific mode
python examples/three_modes_demo.py --mode virtual
python examples/three_modes_demo.py --mode simulated
sudo python examples/three_modes_demo.py --mode real  # Requires root for tc
```

### 2. FogSim Evaluation Demo (`fogsim_evaluation_demo.py`)
Comprehensive evaluation demonstrating the three key hypotheses:
1. **High frame rate**: Virtual mode achieves maximum simulation speed
2. **Reproducibility**: Perfect determinism in virtual mode
3. **Sim-to-real correlation**: Network conditions affect performance similarly

```bash
python examples/fogsim_evaluation_demo.py
```

### 3. Car Braking Experiment (`car_braking_experiment.py`)
Demonstrates reproducibility issues when depending on wallclock time:
- Simple car braking scenario
- Shows variance in outcomes with wallclock-based timing
- Perfect reproducibility with virtual timeline

```bash
python examples/car_braking_experiment.py
```

### 4. Training Convergence Demo (`training_convergence_demo.py`)
Shows benefits of high frame rate for RL training:
- Trains Q-learning agent for fixed wallclock time
- Compares episodes completed in different modes
- Demonstrates faster convergence with virtual mode

```bash
python examples/training_convergence_demo.py
```

## Key Results

### Virtual Mode (FogSIM) Benefits:
- **10-100x higher frame rate** compared to wallclock-synced modes
- **Perfect reproducibility** - identical results across runs
- **More training iterations** in same wallclock time

### Mode Selection Guidelines:
- **Virtual Mode**: Use for training, parameter tuning, and reproducible research
- **Simulated Network Mode**: Use when wallclock synchronization is needed with network simulation
- **Real Network Mode**: Use for final validation and sim-to-real transfer studies

## Real Network Mode

FogSim's real network mode uses an actual network server to forward messages,
measuring real network latency and demonstrating the sim-to-real gap.

### Architecture
1. **FogSim Server**: Runs on local or remote machine, handles simulation messages
2. **Real Network Client**: Forwards messages through actual network to server
3. **Latency Measurement**: Measures actual round-trip time for each message

### Running Real Network Mode

#### 1. Start the FogSim Server
```bash
# Local server
python -m fogsim.real_network_server --host 127.0.0.1 --port 8765

# Remote server (on remote machine)
python -m fogsim.real_network_server --host 0.0.0.0 --port 8765
```

#### 2. Run Sim-to-Real Gap Experiment
```bash
# With local server
python examples/sim_real_gap_demo.py --episodes 5

# With remote server
python examples/sim_real_gap_demo.py --server-host REMOTE_IP --remote --episodes 5
```

#### 3. Test Server Connectivity
```bash
# Test connection to server
python -m fogsim.real_network_client SERVER_IP
```

### Complete Test Script
```bash
# Run comprehensive real network test
./examples/complete_real_network_test.sh
```

This script:
- Starts a local FogSim server
- Tests connectivity
- Runs sim-to-real gap experiments
- Compares all three modes
- Generates comparison plots

## Network Configuration

Examples use predefined network configurations:
- `low_latency`: 1ms delay, 1Gbps bandwidth
- `edge_cloud`: 10ms delay, 100Mbps bandwidth  
- `satellite`: 600ms delay, 10Mbps bandwidth

For real network mode:
```bash
# First, start the FogSim server
python -m fogsim.real_network_server &

# Then run experiments
python examples/three_modes_demo.py --mode real

# Or run the sim-to-real gap experiment
python examples/sim_real_gap_demo.py
```

The real network mode:
- Forwards all messages through the network server
- Measures actual network latency
- Demonstrates sim-to-real gap
- Works with both local and remote servers

## Requirements

- Python 3.7+
- FogSim and dependencies installed
- Linux with tc (traffic control) for real network mode
- Root privileges for real network mode
- Matplotlib for visualization (optional)

## Troubleshooting

1. **Import errors**: Ensure FogSim is installed: `pip install -e .`
2. **Real network server issues**: 
   - Check server is running: `nc -z localhost 8765`
   - Check firewall allows port 8765
   - For remote servers, ensure network connectivity
3. **Low frame rates**: Virtual mode should be 10-100x faster than other modes
4. **Non-reproducible results**: Only virtual mode guarantees perfect reproducibility
5. **Connection refused**: Start the FogSim server before running real network mode
6. **High latency**: Expected for remote servers; use local server for low latency tests