"""
Simplified utilities for collision detection in CARLA simulation
"""

import numpy as np

class CollisionDetector:
    """Simple collision detection based on distance threshold"""
    
    def __init__(self, threshold=3.0):
        self.threshold = threshold
    
    def check_collision(self, ego_pos, obstacle_pos):
        """
        Check if collision occurs based on distance threshold
        
        Args:
            ego_pos: [x, y] position of ego vehicle
            obstacle_pos: [x, y] position of obstacle vehicle
            
        Returns:
            bool: True if collision detected, False otherwise
        """
        distance = np.sqrt(
            (ego_pos[0] - obstacle_pos[0])**2 + 
            (ego_pos[1] - obstacle_pos[1])**2
        )
        return distance < self.threshold
    
    def get_distance(self, ego_pos, obstacle_pos):
        """
        Get distance between ego vehicle and obstacle vehicle
        
        Args:
            ego_pos: [x, y] position of ego vehicle
            obstacle_pos: [x, y] position of obstacle vehicle
            
        Returns:
            float: Distance in meters
        """
        return np.sqrt(
            (ego_pos[0] - obstacle_pos[0])**2 + 
            (ego_pos[1] - obstacle_pos[1])**2
        )