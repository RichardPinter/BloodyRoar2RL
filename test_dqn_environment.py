#!/usr/bin/env python3
"""
Test DQN Environment in Isolation

Tests the DQN slow environment with mock data to ensure it works correctly
without requiring a game connection. This isolates potential issues to the
DQN environment before testing the full factory pattern.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import time
from typing import Tuple

def test_dqn_environment_creation():
    """Test basic DQN environment creation without game connection"""
    print("🧪 Testing DQN Environment Creation...")
    
    try:
        from core.dqn_slow_rl_environment import DQNSlowRLEnvironment
        
        # Create environment with small parameters for testing
        env = DQNSlowRLEnvironment(
            frame_stack_size=4,
            img_size=(84, 84),  # Smaller for testing
            health_history_length=4,
            window_title="NonExistentWindow"  # Intentionally use fake window
        )
        
        print(f"✅ DQN Environment created successfully")
        print(f"   Frame stack: {env.frame_stack_size}")
        print(f"   Image size: {env.img_size}")
        print(f"   Health history: {env.health_history_length}")
        print(f"   Environment type: DQN")
        print(f"   Window capture: {'Found' if env.window_capture else 'Not found (expected)'}")
        
        # Test basic properties
        print(f"   Action space size: {env.get_action_space_size()}")
        print(f"   Actions: {env.get_actions()}")
        print(f"   Observation space size: {env.get_observation_space_size()}")
        
        env.close()
        return True
        
    except Exception as e:
        print(f"❌ DQN Environment creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_hybrid_state_manager_integration():
    """Test that HybridStateManager works with mock data"""
    print("\n🧪 Testing HybridStateManager Integration...")
    
    try:
        from core.dqn_slow_rl_environment import DQNSlowRLEnvironment
        from detection.health_detector import HealthState
        
        env = DQNSlowRLEnvironment(
            frame_stack_size=4,
            img_size=(64, 64),  # Even smaller for testing
            health_history_length=4,
            window_title="NonExistentWindow"
        )
        
        # Test adding mock frames to hybrid state manager
        print("   Adding mock frames to HybridStateManager...")
        
        for i in range(6):  # Add more than frame stack size
            # Create mock screenshot
            mock_screenshot = np.random.randint(0, 255, (100, 150, 3), dtype=np.uint8)
            
            # Create mock health state
            mock_health = HealthState(
                p1_health=100.0 - i*5,
                p2_health=100.0 - i*3,
                p1_pixels=i*10,
                p2_pixels=i*6
            )
            
            # Add frame to hybrid state manager
            env.hybrid_state_manager.add_frame(mock_screenshot, mock_health)
            
            print(f"     Frame {i+1}: P1={mock_health.p1_health:.1f}% P2={mock_health.p2_health:.1f}%")
        
        # Test getting current state
        screenshots, health_history = env.hybrid_state_manager.get_current_state()
        
        print(f"\n   ✅ Hybrid State Output:")
        print(f"     Screenshots: {screenshots.shape} dtype={screenshots.dtype}")
        print(f"     Health history: {health_history.shape} dtype={health_history.dtype}")
        print(f"     Screenshot range: [{screenshots.min():.3f}, {screenshots.max():.3f}]")
        print(f"     Latest health: P1={health_history[-1, 0]:.1f}% P2={health_history[-1, 1]:.1f}%")
        print(f"     Latest deltas: P1={health_history[-1, 2]:+.1f}% P2={health_history[-1, 3]:+.1f}%")
        
        # Verify output format
        assert screenshots.shape == (4, 64, 64), f"Wrong screenshot shape: {screenshots.shape}"
        assert health_history.shape == (4, 4), f"Wrong health shape: {health_history.shape}"
        assert screenshots.dtype == np.float32, f"Wrong screenshot dtype: {screenshots.dtype}"
        assert health_history.dtype == np.float32, f"Wrong health dtype: {health_history.dtype}"
        assert 0.0 <= screenshots.min() <= screenshots.max() <= 1.0, "Screenshots not normalized [0,1]"
        
        print("   ✅ Output format validation passed!")
        
        # Test readiness
        print(f"   Manager ready: {env.hybrid_state_manager.is_ready()}")
        print(f"   Manager full: {env.hybrid_state_manager.is_full()}")
        
        env.close()
        return True
        
    except Exception as e:
        print(f"❌ HybridStateManager integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_observation_hook():
    """Test the _on_observation_collected hook with mock data"""
    print("\n🧪 Testing Observation Collection Hook...")
    
    try:
        from core.dqn_slow_rl_environment import DQNSlowRLEnvironment
        from core.round_sub_episode import GameState
        from detection.health_detector import HealthState
        
        env = DQNSlowRLEnvironment(
            frame_stack_size=4,
            img_size=(48, 48),  # Very small for testing
            health_history_length=4,
            window_title="NonExistentWindow"
        )
        
        # Create mock game state
        mock_health_state = HealthState(
            p1_health=85.0,
            p2_health=92.0,
            p1_pixels=60,
            p2_pixels=32
        )
        
        mock_game_state = GameState(
            p1_health=85.0,
            p2_health=92.0,
            p1_position=(150, 200),
            p2_position=(350, 200),
            fighter_distance=200.0,
            frame_count=1000,
            timestamp=time.time()
        )
        
        # Test observation collection hook
        print("   Testing _on_observation_collected hook...")
        env._on_observation_collected(mock_game_state)
        
        print(f"   ✅ Observation hook completed without errors")
        print(f"   Manager frame count: {env.hybrid_state_manager.frame_count}")
        
        # Test multiple calls
        for i in range(3):
            # Update the mock game state health values
            mock_game_state.p1_health -= 2.0
            mock_game_state.p2_health -= 1.5
            env._on_observation_collected(mock_game_state)
        
        print(f"   Manager frame count after 4 calls: {env.hybrid_state_manager.frame_count}")
        print(f"   Manager ready: {env.hybrid_state_manager.is_ready()}")
        
        env.close()
        return True
        
    except Exception as e:
        print(f"❌ Observation hook test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_output_format():
    """Test the _observation_to_output method"""
    print("\n🧪 Testing Output Format...")
    
    try:
        from core.dqn_slow_rl_environment import DQNSlowRLEnvironment
        from core.base_slow_rl_environment import SlowState, SlowObservation
        from detection.health_detector import HealthState
        
        env = DQNSlowRLEnvironment(
            frame_stack_size=3,
            img_size=(32, 32),  # Tiny for testing
            health_history_length=3,
            window_title="NonExistentWindow"
        )
        
        # Add some data to hybrid state manager first
        for i in range(5):
            mock_screenshot = np.random.randint(0, 255, (50, 80, 3), dtype=np.uint8)
            mock_health = HealthState(
                p1_health=90.0 - i*2,
                p2_health=95.0 - i*1.5,
                p1_pixels=i*5,
                p2_pixels=i*3
            )
            env.hybrid_state_manager.add_frame(mock_screenshot, mock_health)
        
        # Create mock SlowState
        mock_obs = SlowObservation(
            timestamp=time.time(),
            p1_health=80.0,
            p2_health=85.0,
            p1_position=(100, 200),
            p2_position=(300, 200),
            distance=200.0,
            frame_count=1000
        )
        
        mock_slow_state = SlowState(
            observations=[mock_obs],
            p1_health_start=90.0,
            p1_health_end=80.0,
            p1_health_delta=-10.0,
            p2_health_start=95.0,
            p2_health_end=85.0,
            p2_health_delta=-10.0,
            p1_movement_distance=50.0,
            p2_movement_distance=30.0,
            p1_net_displacement=40.0,
            p2_net_displacement=25.0,
            average_distance=200.0,
            observation_window_duration=4.0
        )
        
        # Test output format
        print("   Testing _observation_to_output...")
        output = env._observation_to_output(mock_slow_state)
        
        # Verify it's a tuple
        assert isinstance(output, tuple), f"Output should be tuple, got {type(output)}"
        assert len(output) == 2, f"Output should be 2-tuple, got {len(output)}"
        
        screenshots, health_history = output
        
        print(f"   ✅ Output format correct:")
        print(f"     Screenshots: {screenshots.shape} dtype={screenshots.dtype}")
        print(f"     Health history: {health_history.shape} dtype={health_history.dtype}")
        
        # Verify shapes and types
        assert screenshots.shape == (3, 32, 32), f"Wrong screenshot shape: {screenshots.shape}"
        assert health_history.shape == (3, 4), f"Wrong health shape: {health_history.shape}"
        assert isinstance(screenshots, np.ndarray), f"Screenshots should be ndarray"
        assert isinstance(health_history, np.ndarray), f"Health should be ndarray"
        
        print("   ✅ Output format validation passed!")
        
        env.close()
        return True
        
    except Exception as e:
        print(f"❌ Output format test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_error_handling():
    """Test error handling scenarios"""
    print("\n🧪 Testing Error Handling...")
    
    try:
        from core.dqn_slow_rl_environment import DQNSlowRLEnvironment
        
        # Test with insufficient data
        env = DQNSlowRLEnvironment(
            frame_stack_size=4,
            img_size=(32, 32),
            health_history_length=4,
            window_title="NonExistentWindow"
        )
        
        # Test output with no data
        print("   Testing output with insufficient data...")
        from core.base_slow_rl_environment import SlowState, SlowObservation
        import time
        
        mock_obs = SlowObservation(
            timestamp=time.time(),
            p1_health=100.0,
            p2_health=100.0,
            p1_position=(100, 200),
            p2_position=(300, 200),
            distance=200.0,
            frame_count=1000
        )
        
        mock_slow_state = SlowState(
            observations=[mock_obs],
            p1_health_start=100.0,
            p1_health_end=100.0,
            p1_health_delta=0.0,
            p2_health_start=100.0,
            p2_health_end=100.0,
            p2_health_delta=0.0,
            p1_movement_distance=0.0,
            p2_movement_distance=0.0,
            p1_net_displacement=0.0,
            p2_net_displacement=0.0,
            average_distance=200.0,
            observation_window_duration=4.0
        )
        
        # Should return zeros when not ready
        output = env._observation_to_output(mock_slow_state)
        screenshots, health_history = output
        
        print(f"   ✅ Handled insufficient data:")
        print(f"     Screenshots: {screenshots.shape} (should be zeros)")
        print(f"     Health: {health_history.shape} (should be zeros)")
        print(f"     All screenshots zero: {np.allclose(screenshots, 0.0)}")
        print(f"     All health zero: {np.allclose(health_history, 0.0)}")
        
        # Test cleanup
        env.close()
        print("   ✅ Environment cleanup successful")
        
        return True
        
    except Exception as e:
        print(f"❌ Error handling test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all DQN environment tests"""
    print("🔬 DQN SLOW ENVIRONMENT ISOLATED TESTS")
    print("="*60)
    print("Testing DQN environment with mock data (no game connection required)")
    print("="*60)
    
    tests = [
        ("Environment Creation", test_dqn_environment_creation),
        ("HybridStateManager Integration", test_hybrid_state_manager_integration),
        ("Observation Hook", test_observation_hook),
        ("Output Format", test_output_format),
        ("Error Handling", test_error_handling)
    ]
    
    all_passed = True
    results = {}
    
    for test_name, test_func in tests:
        success = test_func()
        results[test_name] = success
        if not success:
            all_passed = False
    
    # Summary
    print(f"\n🎉 TEST SUMMARY")
    print("="*60)
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name}: {status}")
    
    if all_passed:
        print(f"\n🎯 All DQN environment tests passed!")
        print("\n📋 What's Working:")
        print("  ✅ DQN environment creates without game connection")
        print("  ✅ HybridStateManager integration works with mock data")
        print("  ✅ Observation collection hook handles mock game states")
        print("  ✅ Output format is correct tuple (screenshots, health_history)")
        print("  ✅ Error handling works for insufficient data")
        
        print("\n🚀 Ready for Next Step:")
        print("  → Test with actual game connection")
        print("  → Test factory pattern integration")
        print("  → Test with DQN agent")
        
    else:
        print(f"\n⚠️ Some tests failed. Fix DQN environment issues before proceeding.")
    
    return all_passed

if __name__ == "__main__":
    main()