#!/usr/bin/env python3
"""
Test Factory Pattern Architecture

Tests that the new abstract environment architecture works correctly
for both PPO and DQN algorithms.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from typing import Tuple

def test_ppo_environments():
    """Test PPO environment creation and interfaces"""
    print("🧪 Testing PPO Environment Factory...")
    
    try:
        # Test direct PPO environment
        from core.ppo_slow_rl_environment import PPOSlowRLEnvironment
        ppo_env = PPOSlowRLEnvironment()
        
        print(f"✅ PPO Environment created")
        print(f"   Observation space: {ppo_env.get_observation_space_size()}")
        print(f"   Action space: {ppo_env.get_action_space_size()}")
        
        ppo_env.close()
        
        # Test PPO through factory pattern
        from core.arcade_rl_environment import ArcadeRLEnvironment
        ppo_arcade = ArcadeRLEnvironment(env_type="ppo", matches_to_win=2)
        
        print(f"✅ PPO Arcade Environment created")
        print(f"   Observation space: {ppo_arcade.get_observation_space_size()}")
        print(f"   Action space: {ppo_arcade.get_action_space_size()}")
        
        ppo_arcade.close()
        
        return True
        
    except Exception as e:
        print(f"❌ PPO environment test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_dqn_environments():
    """Test DQN environment creation and interfaces"""
    print("\n🧪 Testing DQN Environment Factory...")
    
    try:
        # Test direct DQN environment
        from core.dqn_slow_rl_environment import DQNSlowRLEnvironment
        dqn_env = DQNSlowRLEnvironment(
            frame_stack_size=4,
            img_size=(84, 84),  # Smaller for testing
            health_history_length=4
        )
        
        print(f"✅ DQN Environment created")
        print(f"   Observation space: {dqn_env.get_observation_space_size()}")
        print(f"   Action space: {dqn_env.get_action_space_size()}")
        
        dqn_env.close()
        
        # Test DQN through factory pattern
        from core.arcade_rl_environment import ArcadeRLEnvironment
        dqn_arcade = ArcadeRLEnvironment(env_type="dqn", matches_to_win=2)
        
        print(f"✅ DQN Arcade Environment created")
        print(f"   Observation space: {dqn_arcade.get_observation_space_size()}")
        print(f"   Action space: {dqn_arcade.get_action_space_size()}")
        
        dqn_arcade.close()
        
        return True
        
    except Exception as e:
        print(f"❌ DQN environment test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_output_formats():
    """Test that environments return correct output formats"""
    print("\n🧪 Testing Output Formats...")
    
    try:
        # Test hybrid state manager outputs
        from dqn.hybrid_state import HybridStateManager
        from detection.health_detector import HealthState
        
        manager = HybridStateManager(
            frame_stack_size=4,
            img_size=(84, 84),
            health_history_length=4
        )
        
        # Add some dummy data
        for i in range(6):
            screenshot = np.random.randint(0, 255, (100, 150, 3), dtype=np.uint8)
            health = HealthState(
                p1_health=100.0 - i*5,
                p2_health=100.0 - i*3,
                p1_pixels=i*10,
                p2_pixels=i*6
            )
            manager.add_frame(screenshot, health)
        
        screenshots, health_history = manager.get_current_state()
        
        print(f"✅ Hybrid State Manager Output:")
        print(f"   Screenshots: {screenshots.shape} dtype={screenshots.dtype}")
        print(f"   Health history: {health_history.shape} dtype={health_history.dtype}")
        print(f"   Screenshot range: [{screenshots.min():.3f}, {screenshots.max():.3f}]")
        
        # Verify formats
        assert screenshots.shape == (4, 84, 84), f"Wrong screenshot shape: {screenshots.shape}"
        assert health_history.shape == (4, 4), f"Wrong health shape: {health_history.shape}"
        assert screenshots.dtype == np.float32, f"Wrong screenshot dtype: {screenshots.dtype}"
        assert health_history.dtype == np.float32, f"Wrong health dtype: {health_history.dtype}"
        assert 0.0 <= screenshots.min() <= screenshots.max() <= 1.0, "Screenshots not normalized"
        
        print("✅ Output format validation passed!")
        
        return True
        
    except Exception as e:
        print(f"❌ Output format test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_agents():
    """Test that agents work with their respective environments"""
    print("\n🧪 Testing Agent Integration...")
    
    try:
        # Test DQN agent
        from dqn.dqn_agent import DQNAgent
        
        agent = DQNAgent(
            num_actions=10,
            frame_stack=4,
            img_size=(84, 84),
            health_history_length=4,
            epsilon_decay=1000  # Fast for testing
        )
        
        # Test action selection
        screenshots = np.random.rand(4, 84, 84).astype(np.float32)
        health_history = np.random.rand(4, 4).astype(np.float32)
        
        action = agent.select_action(screenshots, health_history)
        print(f"✅ DQN Agent action selection: {action}")
        
        assert 0 <= action < 10, f"Invalid action: {action}"
        
        print("✅ Agent integration test passed!")
        
        return True
        
    except Exception as e:
        print(f"❌ Agent integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all factory pattern tests"""
    print("🏭 FACTORY PATTERN ARCHITECTURE TEST")
    print("="*60)
    print("Testing the new abstract environment architecture...")
    print("="*60)
    
    tests = [
        ("PPO Environments", test_ppo_environments),
        ("DQN Environments", test_dqn_environments), 
        ("Output Formats", test_output_formats),
        ("Agent Integration", test_agents)
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
        print(f"\n🎯 All tests passed! Factory pattern architecture is working correctly.")
        print("\n📋 What's Ready:")
        print("  ✅ Abstract base environment with shared game logic")
        print("  ✅ PPO environment (flat observation vectors)")
        print("  ✅ DQN environment (hybrid screenshots + health)")
        print("  ✅ Arcade/Match environments with factory pattern")
        print("  ✅ DQN agent with experience replay")
        print("  ✅ Hybrid state management with synchronized data")
        
        print("\n🚀 Next Steps:")
        print("  1. Create DQN training script using arcade environment")
        print("  2. Test with actual game connection")
        print("  3. Compare PPO vs DQN performance")
        
    else:
        print(f"\n⚠️ Some tests failed. Fix the issues above before proceeding.")
    
    return all_passed

if __name__ == "__main__":
    main()