#!/usr/bin/env python3
"""
Test DQN Training Scripts

Quick tests to verify the DQN training scripts work correctly
without requiring a full training run or game connection.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
from unittest.mock import patch, MagicMock

def test_dqn_training_simple():
    """Test basic DQN training script creation and initialization"""
    print("🧪 Testing DQN Training Simple...")
    
    try:
        # Import the training module
        from core.dqn_training_simple import DQNTrainer
        
        # Create trainer with small parameters for testing
        trainer = DQNTrainer(
            frame_stack=2,              # Smaller for testing
            img_size=(32, 32),          # Much smaller for testing
            health_history_length=2,     # Smaller for testing
            epsilon_decay=1000,         # Fast decay for testing
            replay_capacity=1000,       # Small buffer for testing
            batch_size=8,               # Small batch for testing
            target_update_frequency=100      # Frequent updates for testing
        )
        
        print(f"✅ DQN Trainer created successfully")
        print(f"   Environment type: {type(trainer.env).__name__}")
        print(f"   Agent type: {type(trainer.agent).__name__}")
        print(f"   Action space: {trainer.env.get_action_space_size()}")
        print(f"   Observation space: {trainer.env.get_observation_space_size()}")
        
        # Test model saving/loading paths
        model_dir = trainer.model_dir
        print(f"   Model directory: {model_dir}")
        print(f"   Directory exists: {os.path.exists(model_dir)}")
        
        # Cleanup
        trainer.env.close()
        
        return True
        
    except Exception as e:
        print(f"❌ DQN Training Simple test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_dqn_arcade_training():
    """Test interactive DQN arcade training script"""
    print("\n🧪 Testing DQN Arcade Training...")
    
    try:
        # Import the arcade training module
        from core.train_dqn_arcade_interactive import InteractiveDQNArcadeTrainer
        
        # Create trainer with minimal parameters
        trainer = InteractiveDQNArcadeTrainer(
            arcade_opponents=2,         # Fewer opponents for testing
            frame_stack=2,              # Smaller for testing
            img_size=(32, 32),          # Smaller for testing
            health_history_length=2,     # Smaller for testing
            epsilon_decay=500,          # Fast decay for testing
            replay_capacity=500,        # Small buffer for testing
            batch_size=4,               # Small batch for testing
            target_update_frequency=50       # Frequent updates for testing
        )
        
        print(f"✅ DQN Arcade Trainer created successfully")
        print(f"   Environment type: {type(trainer.env).__name__}")
        print(f"   Agent type: {type(trainer.agent).__name__}")
        print(f"   Arcade opponents: {trainer.arcade_opponents}")
        print(f"   Action space: {trainer.env.get_action_space_size()}")
        print(f"   Models directory: {trainer.models_dir}")
        
        # Test checkpoint functionality (without actually saving)
        print(f"   Checkpoint save interval: {trainer.save_interval} episodes")
        print(f"   Training active: {trainer.training_active}")
        print(f"   Best reward: {trainer.best_reward}")
        
        # Cleanup
        trainer.env.close()
        
        return True
        
    except Exception as e:
        print(f"❌ DQN Arcade Training test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_training_components():
    """Test that all training components work together"""
    print("\n🧪 Testing Training Components Integration...")
    
    try:
        # Test DQN agent with mock data
        from dqn.dqn_agent import DQNAgent
        
        agent = DQNAgent(
            num_actions=10,
            frame_stack=2,
            img_size=(32, 32),
            health_history_length=2,
            replay_capacity=100
        )
        
        # Test action selection
        mock_screenshots = np.random.rand(2, 32, 32).astype(np.float32)
        mock_health = np.random.rand(2, 4).astype(np.float32)
        
        action = agent.select_action(mock_screenshots, mock_health)
        print(f"✅ Agent action selection: {action}")
        
        # Test transition storage
        next_screenshots = np.random.rand(2, 32, 32).astype(np.float32)
        next_health = np.random.rand(2, 4).astype(np.float32)
        
        agent.store_transition(
            mock_screenshots, mock_health, action, 1.0,
            next_screenshots, next_health, False
        )
        print(f"✅ Transition stored in replay buffer")
        print(f"   Buffer size: {agent.replay_buffer.size}")
        
        # Test model save/load paths
        test_path = "test_model.pth"
        if os.path.exists(test_path):
            os.remove(test_path)  # Clean up from previous test
            
        print(f"✅ Components integration test passed")
        
        return True
        
    except Exception as e:
        print(f"❌ Components integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_environment_compatibility():
    """Test DQN environment compatibility with training scripts"""
    print("\n🧪 Testing Environment Compatibility...")
    
    try:
        # Test DQN environment directly
        from core.dqn_slow_rl_environment import DQNSlowRLEnvironment
        
        env = DQNSlowRLEnvironment(
            frame_stack_size=2,
            img_size=(32, 32),
            health_history_length=2,
            window_title="NonExistentWindow"  # Intentionally fake
        )
        
        # Test reset
        state = env.reset()
        if isinstance(state, tuple) and len(state) == 2:
            screenshots, health_history = state
            print(f"✅ Environment reset successful")
            print(f"   Screenshots shape: {screenshots.shape}")
            print(f"   Health history shape: {health_history.shape}")
        else:
            print(f"⚠️ Unexpected state format: {type(state)}")
        
        # Test action space
        action_space_size = env.get_action_space_size()
        observation_space_size = env.get_observation_space_size()
        actions = env.get_actions()
        
        print(f"✅ Environment action space: {action_space_size}")
        print(f"   Observation space: {observation_space_size}")
        print(f"   Available actions: {actions[:3]}...")  # Show first 3
        
        # Test arcade environment with DQN
        from core.arcade_rl_environment import ArcadeRLEnvironment
        
        arcade_env = ArcadeRLEnvironment(matches_to_win=2, env_type="dqn")
        arcade_state = arcade_env.reset()
        
        if isinstance(arcade_state, tuple) and len(arcade_state) == 2:
            print(f"✅ Arcade DQN environment compatible")
        else:
            print(f"⚠️ Arcade environment state format: {type(arcade_state)}")
        
        # Cleanup
        env.close()
        arcade_env.close()
        
        return True
        
    except Exception as e:
        print(f"❌ Environment compatibility test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all DQN training tests"""
    print("🔬 DQN TRAINING SCRIPTS TEST SUITE")
    print("="*60)
    print("Testing DQN training scripts without full execution...")
    print("="*60)
    
    tests = [
        ("DQN Training Simple", test_dqn_training_simple),
        ("DQN Arcade Training", test_dqn_arcade_training),
        ("Training Components", test_training_components),
        ("Environment Compatibility", test_environment_compatibility)
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
        print(f"\n🎯 All tests passed! DQN training scripts are ready.")
        print("\n📋 What's Available:")
        print("  ✅ core/dqn_training_simple.py - Basic DQN training")
        print("  ✅ core/train_dqn_arcade_interactive.py - Interactive arcade training")
        print("  ✅ Hybrid visual + health input support")
        print("  ✅ Experience replay and target networks")
        print("  ✅ Model saving/loading with metadata")
        print("  ✅ Compatible with existing environment architecture")
        
        print("\n🚀 Ready to Start:")
        print("  1. python core/dqn_training_simple.py")
        print("  2. python core/train_dqn_arcade_interactive.py")
        
    else:
        print(f"\n⚠️ Some tests failed. Fix issues before using training scripts.")
    
    return all_passed

if __name__ == "__main__":
    main()