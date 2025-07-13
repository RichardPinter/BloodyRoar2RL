#!/usr/bin/env python3
"""
Quick Test for DQN Hybrid Architecture

Tests the complete DQN pipeline with arcade environment data shapes
to catch any issues before running full training.
"""

import torch
import numpy as np
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dqn.dqn_agent import DQNAgent
from dqn.hybrid_replay_buffer import HybridReplayBuffer
from dqn.vision_network import DQNVisionNetwork

def test_vision_network():
    """Test DQN Vision Network with hybrid inputs"""
    print("🧠 Testing DQN Vision Network...")
    
    try:
        # Create network with arcade parameters
        network = DQNVisionNetwork(
            frame_stack=8,
            img_size=(84, 84),
            num_actions=10,
            health_history_length=8,
            num_health_features=15  # 4 health + 8 match + 3 arcade
        )
        
        # Create realistic test data
        batch_size = 2
        screenshots = torch.randn(batch_size, 8, 84, 84)  # (batch, frames, h, w)
        health_history = torch.randn(batch_size, 8, 15)   # (batch, timesteps, features)
        
        print(f"  📊 Input shapes:")
        print(f"     Screenshots: {screenshots.shape}")
        print(f"     Health history: {health_history.shape}")
        
        # Test forward pass
        network.eval()
        with torch.no_grad():
            q_values = network(screenshots, health_history)
        
        print(f"  ✅ Forward pass successful!")
        print(f"     Output shape: {q_values.shape}")
        print(f"     Expected: ({batch_size}, 10)")
        
        # Verify output shape
        expected_shape = (batch_size, 10)
        assert q_values.shape == expected_shape, f"Shape mismatch: {q_values.shape} vs {expected_shape}"
        
        # Test single prediction with numpy
        single_screenshots = screenshots[0].numpy()  # (8, 84, 84)
        single_health = health_history[0].numpy()    # (8, 15)
        
        q_vals_single = network.predict_q_values(single_screenshots, single_health)
        print(f"  ✅ Single prediction successful!")
        print(f"     Q-values shape: {q_vals_single.shape}")
        print(f"     Sample Q-values: {q_vals_single[:3]}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Vision Network test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_replay_buffer():
    """Test Hybrid Replay Buffer with 15 features"""
    print("\n🧠 Testing Hybrid Replay Buffer...")
    
    try:
        # Create buffer with arcade parameters
        buffer = HybridReplayBuffer(
            capacity=1000,
            frame_stack=8,
            img_size=(84, 84),
            health_history_length=8,
            num_health_features=15
        )
        
        print(f"  📊 Buffer configuration:")
        print(f"     Capacity: {buffer.capacity}")
        print(f"     Screenshot shape: (8, 84, 84)")
        print(f"     Health shape: (8, 15)")
        
        # Create realistic test data
        screenshots = np.random.rand(8, 84, 84).astype(np.float32)
        health_history = np.random.rand(8, 15).astype(np.float32)
        next_screenshots = np.random.rand(8, 84, 84).astype(np.float32)
        next_health_history = np.random.rand(8, 15).astype(np.float32)
        
        # Test storing transitions
        for i in range(50):  # Store enough for a batch
            buffer.add_transition(
                screenshots=screenshots,
                health_history=health_history,
                action=i % 10,
                reward=float(np.random.randn()),
                next_screenshots=next_screenshots,
                next_health_history=next_health_history,
                done=(i % 20 == 19)
            )
        
        print(f"  ✅ Stored {buffer.size} transitions")
        
        # Test sampling
        batch = buffer.sample_batch(32)
        assert batch is not None, "Failed to sample batch"
        
        screenshots_batch, health_batch, actions, rewards, next_screenshots_batch, next_health_batch, dones = batch
        
        print(f"  ✅ Batch sampling successful!")
        print(f"     Screenshots batch: {screenshots_batch.shape}")
        print(f"     Health batch: {health_batch.shape}")
        print(f"     Actions: {actions.shape}")
        
        # Verify shapes
        assert screenshots_batch.shape == (32, 8, 84, 84), f"Screenshots shape: {screenshots_batch.shape}"
        assert health_batch.shape == (32, 8, 15), f"Health shape: {health_batch.shape}"
        assert actions.shape == (32,), f"Actions shape: {actions.shape}"
        
        return True
        
    except Exception as e:
        print(f"  ❌ Replay Buffer test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_dqn_agent():
    """Test DQN Agent with hybrid inputs"""
    print("\n🤖 Testing DQN Agent...")
    
    try:
        # Create agent with arcade parameters
        agent = DQNAgent(
            num_actions=10,
            frame_stack=8,
            img_size=(84, 84),
            health_history_length=8,
            num_health_features=15,  # 4 health + 8 match + 3 arcade
            replay_capacity=1000
        )
        
        print(f"  ✅ Agent created successfully!")
        
        # Create realistic test data
        screenshots = np.random.rand(8, 84, 84).astype(np.float32)
        health_history = np.random.rand(8, 15).astype(np.float32)
        
        print(f"  📊 Test data shapes:")
        print(f"     Screenshots: {screenshots.shape}")
        print(f"     Health history: {health_history.shape}")
        
        # Test action selection
        action = agent.select_action(screenshots, health_history)
        print(f"  ✅ Action selection successful! Action: {action}")
        assert 0 <= action < 10, f"Invalid action: {action}"
        
        # Test storing transition
        next_screenshots = np.random.rand(8, 84, 84).astype(np.float32)
        next_health_history = np.random.rand(8, 15).astype(np.float32)
        
        agent.store_transition(
            screenshots=screenshots,
            health_history=health_history,
            action=action,
            reward=1.0,
            next_screenshots=next_screenshots,
            next_health_history=next_health_history,
            done=False
        )
        
        print(f"  ✅ Transition storage successful!")
        print(f"     Replay buffer size: {agent.replay_buffer.size}")
        
        # Test multiple transitions for batch training
        for i in range(35):  # Enough for a batch
            action_i = agent.select_action(screenshots, health_history)
            agent.store_transition(
                screenshots=screenshots,
                health_history=health_history,
                action=action_i,
                reward=np.random.randn(),
                next_screenshots=next_screenshots,
                next_health_history=next_health_history,
                done=(i % 10 == 9)
            )
        
        print(f"  ✅ Multiple transitions stored! Buffer size: {agent.replay_buffer.size}")
        
        # Test training update
        if agent.replay_buffer.size >= 32:
            loss, epsilon = agent.update(batch_size=32)
            print(f"  ✅ Training update successful! Loss: {loss:.6f}, Epsilon: {epsilon:.3f}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ DQN Agent test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_integration():
    """Test complete integration similar to training loop"""
    print("\n🔄 Testing Complete Integration...")
    
    try:
        # Simulate arcade environment output
        print("  📊 Simulating arcade environment output...")
        
        # Base health features (4): [p1_health, p2_health, p1_delta, p2_delta]
        base_health = np.array([
            [85.5, 92.3, -2.1, 0.0],
            [83.4, 92.3, -2.1, 0.0],
            [81.0, 89.7, -2.4, -2.6],
            [78.5, 87.1, -2.5, -2.6],
            [76.0, 84.5, -2.5, -2.6],
            [73.5, 82.0, -2.5, -2.5],
            [71.0, 79.5, -2.5, -2.5],
            [68.5, 77.0, -2.5, -2.5],
        ], dtype=np.float32)
        
        # Match features (8): [round, p1_rounds, p2_rounds, match_point_p1, match_point_p2, etc.]
        match_features = np.array([1.0, 0.0, 1.0, 0.0, 1.0, 15.5, 0.0, 0.0], dtype=np.float32)
        
        # Arcade features (3): [opponent, total_wins, is_final]
        arcade_features = np.array([2.0, 1.0, 0.0], dtype=np.float32)
        
        # Combine features for each timestep
        health_history = np.zeros((8, 15), dtype=np.float32)
        for i in range(8):
            health_history[i, :4] = base_health[i]                    # Health features
            health_history[i, 4:12] = match_features                  # Match features  
            health_history[i, 12:15] = arcade_features                # Arcade features
        
        # Screenshots
        screenshots = np.random.rand(8, 84, 84).astype(np.float32)
        
        print(f"     Screenshots shape: {screenshots.shape}")
        print(f"     Health history shape: {health_history.shape}")
        print(f"     Health sample (timestep 0): {health_history[0]}")
        print(f"       - Base health: {health_history[0, :4]}")
        print(f"       - Match context: {health_history[0, 4:12]}")
        print(f"       - Arcade context: {health_history[0, 12:15]}")
        
        # Test with DQN agent
        agent = DQNAgent(
            num_actions=10,
            frame_stack=8,
            img_size=(84, 84), 
            health_history_length=8,
            num_health_features=15
        )
        
        # Test action selection
        action = agent.select_action(screenshots, health_history)
        print(f"  ✅ Integration test successful!")
        print(f"     Selected action: {action}")
        print(f"     Current epsilon: {agent.get_current_epsilon():.3f}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_q_value_learning():
    """Test that Q-values actually change after training (CNN learns)"""
    print("\n🎓 Testing Q-Value Learning...")
    
    try:
        # Create agent
        agent = DQNAgent(
            num_actions=10,
            frame_stack=8,
            img_size=(84, 84),
            health_history_length=8,
            num_health_features=15,
            replay_capacity=1000
        )
        
        # Create test data
        screenshots = np.random.rand(8, 84, 84).astype(np.float32)
        health_history = np.random.rand(8, 15).astype(np.float32)
        
        # Get initial Q-values (before training)
        initial_action = agent.select_action(screenshots, health_history)
        
        # Store some weights to check they change
        initial_conv_weight = agent.q_network.conv1.weight.data.clone()
        initial_fc_weight = agent.q_network.fc1.weight.data.clone()
        
        print(f"  📊 Initial state:")
        print(f"     Initial action: {initial_action}")
        print(f"     Conv1 weight sum: {initial_conv_weight.sum():.6f}")
        print(f"     FC1 weight sum: {initial_fc_weight.sum():.6f}")
        
        # Fill replay buffer with training data
        for i in range(100):  # More data for better training
            next_screenshots = np.random.rand(8, 84, 84).astype(np.float32)
            next_health_history = np.random.rand(8, 15).astype(np.float32)
            
            action = agent.select_action(screenshots, health_history)
            reward = np.random.randn() * 10  # Larger rewards for clearer learning
            
            agent.store_transition(
                screenshots=screenshots,
                health_history=health_history,
                action=action,
                reward=reward,
                next_screenshots=next_screenshots,
                next_health_history=next_health_history,
                done=(i % 20 == 19)
            )
            
            # Update for next iteration
            screenshots = next_screenshots
            health_history = next_health_history
        
        print(f"  ✅ Stored {agent.replay_buffer.size} training transitions")
        
        # Perform multiple training updates
        total_loss = 0
        num_updates = 10
        
        for i in range(num_updates):
            loss, epsilon = agent.update(batch_size=32)
            total_loss += loss
            if i == 0:
                first_loss = loss
            if i == num_updates - 1:
                final_loss = loss
        
        avg_loss = total_loss / num_updates
        print(f"  🧠 Training completed:")
        print(f"     Updates: {num_updates}")
        print(f"     First loss: {first_loss:.6f}")
        print(f"     Final loss: {final_loss:.6f}")
        print(f"     Average loss: {avg_loss:.6f}")
        
        # Check if weights changed
        final_conv_weight = agent.q_network.conv1.weight.data
        final_fc_weight = agent.q_network.fc1.weight.data
        
        conv_weight_change = (final_conv_weight - initial_conv_weight).abs().mean().item()
        fc_weight_change = (final_fc_weight - initial_fc_weight).abs().mean().item()
        
        print(f"  📈 Weight changes:")
        print(f"     Conv1 avg change: {conv_weight_change:.8f}")
        print(f"     FC1 avg change: {fc_weight_change:.8f}")
        
        # Test if Q-values changed for same input
        test_screenshots = np.random.rand(8, 84, 84).astype(np.float32)
        test_health = np.random.rand(8, 15).astype(np.float32)
        
        # Get Q-values before and after training (using same input)
        agent.q_network.eval()
        with torch.no_grad():
            screenshots_tensor = torch.FloatTensor(test_screenshots).unsqueeze(0)
            health_tensor = torch.FloatTensor(test_health).unsqueeze(0)
            
            # Reset network to initial state temporarily
            agent.q_network.load_state_dict(agent.target_network.state_dict())
            initial_q_values = agent.q_network(screenshots_tensor, health_tensor).squeeze().numpy()
            
            # Load trained weights back
            agent.target_network.load_state_dict(agent.q_network.state_dict())
            final_q_values = agent.q_network(screenshots_tensor, health_tensor).squeeze().numpy()
        
        q_value_change = np.abs(final_q_values - initial_q_values).mean()
        
        print(f"  🎯 Q-value analysis:")
        print(f"     Q-value change: {q_value_change:.6f}")
        print(f"     Initial Q-values: {initial_q_values[:3]}")
        print(f"     Final Q-values: {final_q_values[:3]}")
        
        # Verify learning occurred
        learning_occurred = (
            conv_weight_change > 1e-6 and  # Weights changed significantly
            fc_weight_change > 1e-6 and    # FC weights also changed
            q_value_change > 1e-4           # Q-values changed meaningfully
        )
        
        if learning_occurred:
            print(f"  ✅ Learning verified! CNN is updating and learning!")
            return True
        else:
            print(f"  ❌ Learning not detected. Weights or Q-values didn't change enough.")
            print(f"     This might indicate frozen weights or learning issues.")
            return False
        
    except Exception as e:
        print(f"  ❌ Q-value learning test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("🧪 DQN HYBRID ARCHITECTURE QUICK TEST")
    print("=" * 60)
    print("Testing complete DQN pipeline before running full training...")
    print()
    
    tests = [
        ("Vision Network", test_vision_network),
        ("Replay Buffer", test_replay_buffer),
        ("DQN Agent", test_dqn_agent),
        ("Integration", test_integration),
        ("Q-Value Learning", test_q_value_learning)
    ]
    
    results = {}
    all_passed = True
    
    for test_name, test_func in tests:
        success = test_func()
        results[test_name] = success
        if not success:
            all_passed = False
    
    print("\n" + "=" * 60)
    print("🎉 TEST SUMMARY")
    print("=" * 60)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:20} {status}")
    
    print()
    if all_passed:
        print("🎯 ALL TESTS PASSED!")
        print("✅ DQN hybrid architecture is ready for training!")
        print("✅ Arcade environment shapes are compatible!")
        print("✅ You can now run the full training with confidence!")
    else:
        print("⚠️  SOME TESTS FAILED!")
        print("❌ Fix the issues above before running full training.")
        print("💡 This saved you from waiting an hour for a crash!")
    
    print("=" * 60)
    return all_passed

if __name__ == "__main__":
    main()