#!/usr/bin/env python3
"""
Simple test to verify sync_mode is working in CollisionHandler
"""

import main
import tempfile
import os

def test_collision_handler_sync_mode():
    """Test that CollisionHandler respects sync_mode setting"""
    
    # Test with sync_mode = True
    config = main.opposite_direction_merge_config.copy()
    config['simulation']['sync_mode'] = True
    
    print("Testing CollisionHandler with sync_mode=True")
    print(f"Config sync_mode: {config['simulation']['sync_mode']}")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            handler = main.CollisionHandler(config, temp_dir, no_risk_eval=True)
            handler.launch()
            
            # Check the CARLA world settings
            world_settings = handler.world.get_settings()
            print(f"CARLA synchronous_mode setting: {world_settings.synchronous_mode}")
            
            handler.close()
            
            assert world_settings.synchronous_mode == True, "Expected synchronous_mode=True"
            print("✓ sync_mode=True working correctly")
            
        except Exception as e:
            print(f"Error with sync_mode=True: {e}")
            return False
    
    # Test with sync_mode = False  
    config['simulation']['sync_mode'] = False
    
    print("\nTesting CollisionHandler with sync_mode=False")
    print(f"Config sync_mode: {config['simulation']['sync_mode']}")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            handler = main.CollisionHandler(config, temp_dir, no_risk_eval=True)
            handler.launch()
            
            # Check the CARLA world settings
            world_settings = handler.world.get_settings()
            print(f"CARLA synchronous_mode setting: {world_settings.synchronous_mode}")
            
            handler.close()
            
            assert world_settings.synchronous_mode == False, "Expected synchronous_mode=False"
            print("✓ sync_mode=False working correctly")
            
        except Exception as e:
            print(f"Error with sync_mode=False: {e}")
            return False
    
    return True

if __name__ == "__main__":
    success = test_collision_handler_sync_mode()
    if success:
        print("\n🎉 CollisionHandler sync_mode configuration is working correctly!")
    else:
        print("\n❌ CollisionHandler sync_mode configuration has issues!")