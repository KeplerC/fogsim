# CloudSim: Network and Robotics Co-Simulation Framework

CloudSim is a co-simulation framework that enables simulation of robotic systems with realistic network effects. It integrates robotics simulators (via ROS2) with network simulators to provide a comprehensive testing environment.

## Architecture

The framework consists of the following components:

- **Meta Simulator**: Central coordination server that manages the simulation clock and message passing between components
- **Simulator Adaptor**: ROS2 node that connects to robotic simulators and forwards state to algorithms
- **Algorithm Adaptor**: ROS2 node that sends inputs to algorithms and measures response latencies
- **Network Simulator Adaptor**: Calculates network delays between components

All components communicate using Protocol Buffers (protobuf) messages across process boundaries.

## Requirements

- Docker and Docker Compose
- ROS2 Humble (for local development)
- Python 3.8+

## Getting Started

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd cloudsim
   ```

2. Build and start the containers:
   ```bash
   docker-compose up --build
   ```

3. Access the Meta Simulator API at http://localhost:5000/state

## Extending the Framework

### Adding a Simulator

To add a robotic simulator:

1. Create a ROS2 wrapper for your simulator that publishes simulator state
2. Configure the Simulator Adaptor to subscribe to your simulator's topics
3. Update the `docker-compose.yml` file to include your simulator container

### Adding an Algorithm

To add an algorithm:

1. Create a ROS2 node for your algorithm or wrap an existing algorithm with ROS2 interfaces
2. Configure the Algorithm Adaptor to communicate with your algorithm
3. Update the `docker-compose.yml` file to include your algorithm container

### Using a Different Network Simulator

The framework currently supports a simple network model and NS3 (placeholder). To use a different network simulator:

1. Implement a new class in `network_simulator_adaptor.py` that provides the `calculate_latency` method
2. Set the `NETWORK_SIMULATOR_TYPE` environment variable in `docker-compose.yml`

## API Reference

### Meta Simulator API

- `GET /state`: Get current simulation state
- `POST /register`: Register a new adaptor
- `POST /send`: Send a message
- `POST /poll`: Poll for messages
- `POST /advance`: Advance simulation time

## License

[MIT License](LICENSE) 