#!/usr/bin/env python3
"""
DQN Agent

Deep Q-Network agent that uses hybrid state input (screenshots + health history).
Mirrors the PPO agent structure but implements Q-learning with experience replay.

Single Responsibility:
- Combine CNN vision network with health features
- Implement epsilon-greedy action selection
- Handle experience replay training
- Provide same interface as PPO agent (select_action, update, save/load)
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from typing import Tuple, List, Optional
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dqn.vision_network import DQNVisionNetwork
from dqn.hybrid_replay_buffer import HybridReplayBuffer


class DQNAgent:
    """
    DQN Agent for hybrid screenshot + health input.
    
    Mirrors PPO agent interface but uses Q-learning with experience replay.
    Combines CNN for screenshots with health history features.
    """
    
    def __init__(self, 
                 num_actions: int = 10,
                 frame_stack: int = 4,
                 img_size: Tuple[int, int] = (168, 168),
                 health_history_length: int = 4,
                 lr: float = 1e-4,
                 epsilon_start: float = 1.0,
                 epsilon_end: float = 0.1,
                 epsilon_decay: int = 50000,
                 replay_capacity: int = 50000,
                 target_update_frequency: int = 1000):
        """
        Initialize DQN agent.
        
        Args:
            num_actions: Number of possible actions
            frame_stack: Number of screenshot frames to stack
            img_size: Size of preprocessed screenshots (height, width)
            health_history_length: Number of health frames to track
            lr: Learning rate
            epsilon_start: Initial exploration rate
            epsilon_end: Final exploration rate
            epsilon_decay: Steps to decay epsilon over
            replay_capacity: Experience replay buffer size
            target_update_frequency: How often to update target network
        """
        self.num_actions = num_actions
        self.frame_stack = frame_stack
        self.img_size = img_size
        self.health_history_length = health_history_length
        
        # Exploration parameters
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.target_update_frequency = target_update_frequency
        
        # Training state
        self.steps_done = 0
        self.training_step = 0
        
        # Networks
        self.q_network = DQNVisionNetwork(
            frame_stack=frame_stack,
            img_size=img_size,
            num_actions=num_actions
        )
        
        # Target network (for stable training)
        self.target_network = DQNVisionNetwork(
            frame_stack=frame_stack,
            img_size=img_size,
            num_actions=num_actions
        )
        
        # Initialize target network with same weights
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()  # Target network is always in eval mode
        
        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        
        # Experience replay
        self.replay_buffer = HybridReplayBuffer(
            capacity=replay_capacity,
            frame_stack=frame_stack,
            img_size=img_size,
            health_history_length=health_history_length
        )
        
        print(f"🤖 DQN Agent initialized:")
        print(f"   Actions: {num_actions}")
        print(f"   Screenshot input: {frame_stack} × {img_size}")
        print(f"   Health history: {health_history_length} frames")
        print(f"   Exploration: ε {epsilon_start:.1f} → {epsilon_end:.1f} over {epsilon_decay} steps")
        print(f"   Replay buffer: {replay_capacity} transitions")
        print(f"   Target update: every {target_update_frequency} steps")
    
    def select_action(self, screenshots: np.ndarray, health_history: np.ndarray) -> int:
        """
        Select action using epsilon-greedy policy.
        
        Args:
            screenshots: Stacked screenshots (frame_stack, height, width)
            health_history: Health data (health_history_length, 4)
            
        Returns:
            Selected action index
        """
        # Calculate current epsilon
        epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
                 np.exp(-1.0 * self.steps_done / self.epsilon_decay)
        
        self.steps_done += 1
        
        if random.random() < epsilon:
            # Exploration: random action
            return random.randrange(self.num_actions)
        else:
            # Exploitation: best action according to Q-network
            with torch.no_grad():
                # Convert inputs to tensors
                screenshots_tensor = torch.FloatTensor(screenshots).unsqueeze(0)  # Add batch dimension
                health_tensor = torch.FloatTensor(health_history).unsqueeze(0)    # Add batch dimension
                
                # Get Q-values
                q_values = self.q_network(screenshots_tensor, health_tensor)
                
                # Select action with highest Q-value
                return q_values.argmax().item()
    
    def store_transition(self, 
                        screenshots: np.ndarray,
                        health_history: np.ndarray,
                        action: int,
                        reward: float,
                        next_screenshots: np.ndarray,
                        next_health_history: np.ndarray,
                        done: bool):
        """
        Store transition in experience replay buffer.
        
        Args:
            screenshots: Current stacked screenshots
            health_history: Current health history
            action: Action taken
            reward: Reward received
            next_screenshots: Next stacked screenshots
            next_health_history: Next health history
            done: Whether episode ended
        """
        self.replay_buffer.add_transition(
            screenshots=screenshots,
            health_history=health_history,
            action=action,
            reward=reward,
            next_screenshots=next_screenshots,
            next_health_history=next_health_history,
            done=done
        )
    
    def update(self, batch_size: int = 32, gamma: float = 0.99) -> Tuple[float, float]:
        """
        Update Q-network using experience replay.
        
        Args:
            batch_size: Number of transitions to sample
            gamma: Discount factor
            
        Returns:
            (loss_value, current_epsilon)
        """
        # Check if we have enough experience
        if len(self.replay_buffer) < batch_size:
            return 0.0, self._get_current_epsilon()
        
        # Sample batch from replay buffer
        batch = self.replay_buffer.sample_batch(batch_size)
        
        if batch is None:
            return 0.0, self._get_current_epsilon()
        
        screenshots, health_history, actions, rewards, next_screenshots, next_health_history, dones = batch
        
        # Convert to tensors
        screenshots = torch.FloatTensor(screenshots)
        health_history = torch.FloatTensor(health_history)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_screenshots = torch.FloatTensor(next_screenshots)
        next_health_history = torch.FloatTensor(next_health_history)
        dones = torch.BoolTensor(dones)
        
        # Current Q-values
        current_q_values = self.q_network(screenshots, health_history)
        current_q_values = current_q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Next Q-values from target network
        with torch.no_grad():
            next_q_values = self.target_network(next_screenshots, next_health_history)
            max_next_q_values = next_q_values.max(1)[0]
            
            # Target Q-values using Bellman equation
            target_q_values = rewards + (gamma * max_next_q_values * ~dones)
        
        # Compute loss
        loss = F.mse_loss(current_q_values, target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=10.0)
        
        self.optimizer.step()
        self.training_step += 1
        
        # Update target network periodically
        if self.training_step % self.target_update_frequency == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
            print(f"   🎯 Target network updated (step {self.training_step})")
        
        return loss.item(), self._get_current_epsilon()
    
    def _get_current_epsilon(self) -> float:
        """Get current epsilon value."""
        return self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
               np.exp(-1.0 * self.steps_done / self.epsilon_decay)
    
    def get_current_epsilon(self) -> float:
        """Get current epsilon value (public interface)."""
        return self._get_current_epsilon()
    
    def save(self, path: str, metadata: dict = None):
        """
        Save model weights and training state.
        
        Args:
            path: File path to save to
            metadata: Optional additional data to save
        """
        checkpoint = {
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_step': self.training_step,
            'steps_done': self.steps_done,
            'epsilon_start': self.epsilon_start,
            'epsilon_end': self.epsilon_end,
            'epsilon_decay': self.epsilon_decay,
            'target_update_frequency': self.target_update_frequency
        }
        
        if metadata:
            checkpoint['metadata'] = metadata
        
        torch.save(checkpoint, path)
        print(f"💾 DQN model saved to {path}")
        print(f"   Training step: {self.training_step}")
        print(f"   Exploration steps: {self.steps_done}")
        print(f"   Current ε: {self._get_current_epsilon():.3f}")
    
    def load(self, path: str):
        """
        Load model weights and training state.
        
        Args:
            path: File path to load from
        """
        checkpoint = torch.load(path)
        
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.training_step = checkpoint['training_step']
        self.steps_done = checkpoint['steps_done']
        
        # Load exploration parameters if available
        if 'epsilon_start' in checkpoint:
            self.epsilon_start = checkpoint['epsilon_start']
            self.epsilon_end = checkpoint['epsilon_end']
            self.epsilon_decay = checkpoint['epsilon_decay']
            self.target_update_frequency = checkpoint['target_update_frequency']
        
        print(f"📂 DQN model loaded from {path}")
        print(f"   Training step: {self.training_step}")
        print(f"   Exploration steps: {self.steps_done}")
        print(f"   Current ε: {self._get_current_epsilon():.3f}")
    
    def get_stats(self) -> dict:
        """Get current agent statistics."""
        return {
            'training_step': self.training_step,
            'steps_done': self.steps_done,
            'current_epsilon': self._get_current_epsilon(),
            'replay_buffer_size': len(self.replay_buffer),
            'replay_buffer_capacity': self.replay_buffer.capacity
        }
    
    def print_stats(self):
        """Print current agent statistics."""
        stats = self.get_stats()
        
        print(f"\n🤖 DQN Agent Stats:")
        print(f"   Training steps: {stats['training_step']}")
        print(f"   Exploration steps: {stats['steps_done']}")
        print(f"   Current ε: {stats['current_epsilon']:.3f}")
        print(f"   Replay buffer: {stats['replay_buffer_size']}/{stats['replay_buffer_capacity']}")


class DQNAgentTester:
    """Test suite for DQN Agent."""
    
    def __init__(self):
        self.agent = None
    
    def test_basic_functionality(self):
        """Test basic agent operations."""
        print("\n🧪 Testing Basic Functionality...")
        
        try:
            # Create agent
            self.agent = DQNAgent(
                num_actions=5,
                frame_stack=4,
                img_size=(84, 84),
                health_history_length=4,
                epsilon_decay=1000,  # Small for testing
                replay_capacity=100   # Small for testing
            )
            
            # Test action selection with dummy data
            screenshots = np.random.randint(0, 255, (4, 84, 84), dtype=np.uint8).astype(np.float32) / 255.0
            health_history = np.random.rand(4, 4).astype(np.float32)
            
            # Test multiple action selections
            actions = []
            for i in range(10):
                action = self.agent.select_action(screenshots, health_history)
                actions.append(action)
                print(f"   Step {i+1}: Action={action}, ε={self.agent._get_current_epsilon():.3f}")
            
            # Verify actions are in valid range
            assert all(0 <= a < 5 for a in actions), f"Invalid actions: {actions}"
            
            # Test storing transitions
            for i in range(50):  # Store enough for a batch
                next_screenshots = np.random.randint(0, 255, (4, 84, 84), dtype=np.uint8).astype(np.float32) / 255.0
                next_health_history = np.random.rand(4, 4).astype(np.float32)
                
                self.agent.store_transition(
                    screenshots=screenshots,
                    health_history=health_history,
                    action=random.randint(0, 4),
                    reward=random.uniform(-1, 1),
                    next_screenshots=next_screenshots,
                    next_health_history=next_health_history,
                    done=random.random() < 0.1
                )
                
                screenshots = next_screenshots
                health_history = next_health_history
            
            print(f"   Stored {len(self.agent.replay_buffer)} transitions")
            
            # Test update
            loss, epsilon = self.agent.update(batch_size=32)
            print(f"   Update: Loss={loss:.4f}, ε={epsilon:.3f}")
            
            print("✅ Basic functionality test passed!")
            return True
            
        except Exception as e:
            print(f"❌ Basic functionality test failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def test_save_load(self):
        """Test save/load functionality."""
        print("\n🧪 Testing Save/Load...")
        
        try:
            if self.agent is None:
                self.agent = DQNAgent(num_actions=5, epsilon_decay=1000)
            
            # Save original stats
            original_stats = self.agent.get_stats()
            print(f"   Original stats: {original_stats}")
            
            # Save model
            test_path = "/tmp/test_dqn_agent.pth"
            self.agent.save(test_path, metadata={'test': True})
            
            # Create new agent and load
            new_agent = DQNAgent(num_actions=5, epsilon_decay=1000)
            new_agent.load(test_path)
            
            # Compare stats
            loaded_stats = new_agent.get_stats()
            print(f"   Loaded stats: {loaded_stats}")
            
            # Verify key stats match
            assert original_stats['training_step'] == loaded_stats['training_step'], "Training step mismatch"
            assert original_stats['steps_done'] == loaded_stats['steps_done'], "Steps done mismatch"
            
            # Clean up
            os.remove(test_path)
            
            print("✅ Save/load test passed!")
            return True
            
        except Exception as e:
            print(f"❌ Save/load test failed: {e}")
            return False
    
    def test_exploration_decay(self):
        """Test epsilon exploration decay."""
        print("\n🧪 Testing Exploration Decay...")
        
        try:
            agent = DQNAgent(
                num_actions=3,
                epsilon_start=1.0,
                epsilon_end=0.1,
                epsilon_decay=100  # Fast decay for testing
            )
            
            # Test epsilon values at different steps
            epsilons = []
            for step in range(0, 300, 50):
                agent.steps_done = step
                epsilon = agent._get_current_epsilon()
                epsilons.append(epsilon)
                print(f"   Step {step}: ε={epsilon:.3f}")
            
            # Verify epsilon decreases
            assert epsilons[0] > epsilons[-1], "Epsilon should decrease over time"
            assert epsilons[-1] >= 0.1, "Epsilon should not go below minimum"
            
            print("✅ Exploration decay test passed!")
            return True
            
        except Exception as e:
            print(f"❌ Exploration decay test failed: {e}")
            return False
    
    def run_all_tests(self):
        """Run complete test suite."""
        print("🧪 DQN AGENT TEST SUITE")
        print("="*50)
        
        tests = [
            ("Basic Functionality", self.test_basic_functionality),
            ("Save/Load", self.test_save_load),
            ("Exploration Decay", self.test_exploration_decay)
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
        print("="*50)
        for test_name, result in results.items():
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"{test_name}: {status}")
        
        if all_passed:
            print(f"\n🎯 All tests passed! DQN Agent ready for integration.")
        else:
            print(f"\n⚠️ Some tests failed. Check the output above for details.")
        
        return all_passed


def main():
    """Main testing function."""
    print("🤖 DQN AGENT")
    print("="*40)
    print("Testing DQN agent with hybrid screenshot + health input.")
    print("="*40)
    
    # Run tests
    tester = DQNAgentTester()
    success = tester.run_all_tests()
    
    if success:
        print("\n✅ DQN Agent is ready!")
        print("Next step: Create DQN environment that uses this agent.")
    else:
        print("\n❌ Fix the issues above before proceeding.")


if __name__ == "__main__":
    main()