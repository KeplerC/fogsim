"""
Configuration for CARLA synchronous vs asynchronous mode comparison
Crossroad scenario: ego vehicle approaches stationary obstacle at intersection
"""

# Simple scenario - ego approaches stationary obstacle on straight road
CROSSROAD_CONFIG = {
    'host': 'localhost',
    'port': 2000,
    'delta_seconds': 0.01,  # 100 FPS
    'max_steps': 1200,  # 12 seconds at 100 FPS
    
    # Ego vehicle configuration - on straight road
    'ego_spawn_offset': {
        'x': 4.0,
        'y': -90.0,
        'z': 0,
        'yaw': 0.0  # Facing west
    },
    'throttle': 0.5,
    
    # Obstacle vehicle configuration - stationary on same road
    'obstacle_spawn_offset': {
        'x': 3,
        'y': -30.0,
        'z': 0.0,
        'yaw': 0.0  # Same orientation
    },
    'obstacle_throttle': 0.0,  # Stationary
    
    # Collision detection
    'collision_threshold': 30.0,  # meters - distance to start braking
    
    # Camera settings for viewing scenario (offset from base spawn point)
    'camera_offset': {
        'x': 8.0,      # Offset in x direction
        'y': -70.0,     # Offset in y direction
        'z': 50.0,      # Height above ground
        'fov': '90'
    }
}

EXPERIMENT_CONFIGS = {
    'sync': {
        **CROSSROAD_CONFIG,
        'synchronous_mode': True,
    },
    
    'async': {
        **CROSSROAD_CONFIG,
        'synchronous_mode': False,
    }
}

# Network bandwidth configurations for different experiments
BANDWIDTH_CONFIGS = {
    'high': 100.0,    # 100 Mbps - minimal latency
    'medium': 10.0,   # 10 Mbps - moderate latency
    'low': 1.0,       # 1 Mbps - high latency
    'very_low': 0.1   # 0.1 Mbps - very high latency
}