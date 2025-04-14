
unprotected_right_turn_config = {
    'simulation': {
        'host': 'localhost',
        'port': 2000,
        'fps': 100,
        'delta_seconds': 0.01,  # 100fps
        'prediction_steps': 8000,  # 100 frames, need to be divided to delta_k 
        'l_max': 40,
        'delta_k': 40,
        'emergency_brake_threshold': 1.1,
        'cautious_threshold': 0.0,
        'cautious_delta_k': 15,
        'tracker_type': 'ekf'
    },
    'trajectories': {
        'ego': './ego_trajectory.csv',
        'obstacle': './obstacle_trajectory.csv'
    },
    'video': {
        'filename': './bev_images/unprotected_right_turn_collision.mp4',
        'fps': 100,
        'width': 800,
        'height': 800,
    },
    'ego_vehicle': {
        'model': 'vehicle.tesla.model3',
        'spawn_offset': {
            'x': 4,
            'y': -90,
            'yaw': 0
        },
        'go_straight_ticks': 500,  # * 10ms = 5s
        'turn_ticks': 250,  # * 10ms = 2.5s
        'after_turn_ticks': 200,  # Add this new parameter for post-turn straight driving
        'throttle': {
            'straight': 0.4,
            'turn': 0.4,
            'after_turn': 0.4  # Add throttle for after turn
        },
        'steer': {
            'turn': 0.3
        }
    },
    'obstacle_vehicle': {
        'model': 'vehicle.lincoln.mkz_2020',
        'spawn_offset': {
            'x': 19,
            'y': 28,
            'yaw': 90
        },
        'go_straight_ticks': 400,  # * 10ms = 4s
        'turn_ticks': 200,  # * 10ms = 2s
        'after_turn_ticks': 350,  # * 10ms = 3.5s
        'throttle': {
            'straight': 0.52,
            'turn': 0.4,
            'after_turn': 0.5
        },
        'steer': {
            'straight': 0.0,
            'turn': 0.0,
            'after_turn': 0.0
        }
    },
    'camera': {
        'height': 50,
        'offset': {
            'x': 10,
            'y': 35
        },
        'fov': '90'
    },
    'save_options': {
        'save_video': False,
        'save_images': True,
    }
}

unprotected_left_turn_config = {
    'simulation': {
        'host': 'localhost',
        'port': 2000,
        'fps': 100,
        'delta_seconds': 0.01,  # 100fps
        'prediction_steps': 8000,  # 100 frames, need to be divided to delta_k 
        'l_max': 40,
        'delta_k': 40,
        'emergency_brake_threshold': 1.1,
        'cautious_threshold': 0.0,
        'cautious_delta_k': 20,
        'tracker_type': 'ekf'
    },
    'trajectories': {
        'ego': './ego_trajectory.csv',
        'obstacle': './obstacle_trajectory.csv'
    },
    'video': {
        'filename': './bev_images/unprotected_left_turn_collision.mp4',
        'fps': 100,
        'width': 800,
        'height': 800,
    },
    'ego_vehicle': {
        'model': 'vehicle.tesla.model3',
        'spawn_offset': {
            'x': 7.5,
            'y': -90,
            'yaw': 0
        },
        'go_straight_ticks': 500,  # * 10ms = 5s
        'turn_ticks': 250,  # * 10ms = 2.5s
        'after_turn_ticks': 200,  # Add this new parameter for post-turn straight driving
        'throttle': {
            'straight': 0.4,
            'turn': 0.4,
            'after_turn': 0.4  # Add throttle for after turn
        },
        'steer': {
            'turn': -0.3
        }
    },
    'obstacle_vehicle': {
        'model': 'vehicle.lincoln.mkz_2020',
        'spawn_offset': {
            'x': 3.5,
            'y': 55,
            'yaw': 180
        },
        'go_straight_ticks': 400,
        'turn_ticks': 200,
        'after_turn_ticks': 350,
        'throttle': {
            'straight': 0.53,
            'turn': 0.4,
            'after_turn': 0.52
        },
        'steer': {
            'straight': 0.0,
            'turn': 0.0,
            'after_turn': 0.0
        }
    },
    'camera': {
        'height': 50,
        'offset': {
            'x': 10,
            'y': 35
        },
        'fov': '90'
    },
    'save_options': {
        'save_video': False,
        'save_images': True,
    }
}

opposite_direction_merge_config = {
    'simulation': {
        'host': 'localhost',
        'port': 2000,
        'fps': 100,
        'delta_seconds': 0.01,  # 100fps
        'prediction_steps': 8000,  # 100 frames, need to be divided to delta_k 
        'l_max': 40,
        'delta_k': 40,
        'emergency_brake_threshold': 1.1,
        'cautious_threshold': 0.0,
        'cautious_delta_k': 20,
        'tracker_type': 'ekf'
    },
    'trajectories': {
        'ego': './ego_trajectory.csv',
        'obstacle': './obstacle_trajectory.csv'
    },
    'video': {
        'filename': './bev_images/opposite_direction_merge_collision.mp4',
        'fps': 100,
        'width': 800,
        'height': 800,
    },
    'ego_vehicle': {
        'model': 'vehicle.tesla.model3',
        'spawn_offset': {
            'x': 4,
            'y': -90,
            'yaw': 0
        },
        'go_straight_ticks': 500,  # * 10ms = 5s
        'turn_ticks': 250,  # * 10ms = 2.5s
        'after_turn_ticks': 200,  # Add this new parameter for post-turn straight driving
        'throttle': {
            'straight': 0.4,
            'turn': 0.4,
            'after_turn': 0.4  # Add throttle for after turn
        },
        'steer': {
            'turn': 0.3
        }
    },
    'obstacle_vehicle': {
        'model': 'vehicle.lincoln.mkz_2020',
        'spawn_offset': {
            'x': 8,
            'y': 35,
            'yaw': 180
        },
        'go_straight_ticks': 100,
        'turn_ticks': 400,
        'after_turn_ticks': 350,
        'throttle': {
            'straight': 0.52,
            'turn': 0.45,
            'after_turn': 0.5
        },
        'steer': {
            'straight': 0.0,
            'turn': -0.4,
            'after_turn': 0.0
        }
    },
    'camera': {
        'height': 50,
        'offset': {
            'x': 10,
            'y': 35
        },
        'fov': '90'
    },
    'save_options': {
        'save_video': False,
        'save_images': True,
    }
}

