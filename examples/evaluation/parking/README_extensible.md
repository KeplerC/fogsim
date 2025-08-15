# Extensible Cloud Parking Simulation

This extensible framework simulates autonomous parking with different cloud computing architectures and network delays using FogSim.

## Cloud Computing Scenarios

The system supports four different cloud architectures:

### 1. **Baseline** (`baseline`)
- **Perception**: Local (vehicle)
- **Planning**: Local (vehicle)  
- **Control**: Local (vehicle)
- **Description**: All processing on vehicle (no cloud, no delays)

### 2. **Cloud Perception** (`cloud_perception`)
- **Perception**: Cloud (with network delay)
- **Planning**: Local (vehicle)
- **Control**: Local (vehicle)
- **Description**: Only perception processing is offloaded to cloud

### 3. **Cloud Planning** (`cloud_planning`)
- **Perception**: Local (vehicle)
- **Planning**: Cloud (with network delay)
- **Control**: Local (vehicle)
- **Description**: Only planning processing is offloaded to cloud

### 4. **Full Cloud** (`full_cloud`)
- **Perception**: Cloud
- **Planning**: Cloud  
- **Control**: Cloud (with network delay)
- **Description**: All processing offloaded to cloud

## Network Scenarios

Pre-configured network conditions:

- **`no_latency`**: 0ms delay, 0% loss, high bandwidth
- **`low_latency`**: 5ms delay, 0.1% loss, 1 Mbps
- **`medium_latency`**: 20ms delay, 1% loss, 500 Kbps  
- **`high_latency`**: 50ms delay, 3% loss, 200 Kbps

## Usage Examples

### Basic Usage
```bash
# Test baseline vs cloud perception with low latency
python3 parking_experiment_extensible.py --clouds baseline cloud_perception --network low_latency

# Test all cloud scenarios with medium latency
python3 parking_experiment_extensible.py --clouds baseline cloud_perception cloud_planning full_cloud --network medium_latency

# Test with custom latency
python3 parking_experiment_extensible.py --clouds baseline cloud_planning --latency 15
```

### Advanced Usage
```bash
# Test multiple FogSim modes with video recording
python3 parking_experiment_extensible.py --modes virtual simulated --clouds baseline cloud_perception --video

# Test specific scenarios with custom configuration
python3 parking_experiment_extensible.py --config custom_network.json --clouds full_cloud

# Save current configuration for later use
python3 parking_experiment_extensible.py --network medium_latency --save-config medium_config.json
```

### Video Recording
```bash
# Enable video recording for visual analysis
python3 parking_experiment_extensible.py --clouds baseline cloud_perception --video
```

## Architecture Overview

### Data Flow for Cloud Perception
```
Vehicle Sensors → [Network Delay] → Cloud Perception → Local Planning → Local Control → Vehicle
```

### Data Flow for Cloud Planning  
```
Vehicle Sensors → Local Perception → [Network Delay] → Cloud Planning → Local Control → Vehicle
```

### Data Flow for Full Cloud
```
Vehicle Sensors → [Network] → Cloud Perception → Cloud Planning → [Network] → Vehicle Control
```

## FogSim Integration

The system uses FogSim to accurately simulate:
- **Network delays** for cloud communication
- **Packet loss** effects on cloud reliability
- **Bandwidth limitations** affecting data transmission
- **Different simulation modes** (virtual timeline, real clock, real network)

## Output and Analysis

The experiment generates:
- **Performance metrics**: IOU (parking accuracy), parking time
- **Network statistics**: Messages sent/received, average latency
- **Comparison plots**: IOU and time comparisons across cloud architectures
- **Video recordings**: Visual analysis of parking behavior (optional)

## Files Structure

- `cloud_components.py`: Base classes and component implementations
- `extensible_parking_handler.py`: FogSim-integrated handler
- `parking_experiment_extensible.py`: Main experiment runner
- `experiment_utils.py`: CARLA parking utilities (existing)

## Research Applications

This framework enables research into:
- **Cloud computing trade-offs** in autonomous vehicles
- **Network delay impacts** on autonomous driving performance
- **Component placement optimization** (what to run locally vs cloud)
- **Reliability analysis** with packet loss and variable latency