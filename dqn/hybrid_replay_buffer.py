#!/usr/bin/env python3
"""
Hybrid Replay Buffer for DQN

Stores transitions with both screenshot frames and health history for 
hybrid visual + health reinforcement learning.
"""

import numpy as np
from typing import Tuple, Optional
from collections import deque


class HybridReplayBuffer:
    """
    Experience replay buffer for hybrid states (screenshots + health history).
    
    Stores transitions with:
    - Stacked screenshot frames (frame_stack, height, width)
    - Health history (health_history_length, num_health_features)
    - Actions, rewards, dones
    
    Key Features:
    - Memory efficient storage for screenshots (uint8)
    - Separate storage for health data (float32)
    - Random sampling to break correlation
    - Circular buffer for fixed memory usage
    """
    
    def __init__(self, 
                 capacity: int = 50000,
                 frame_stack: int = 8,
                 img_size: Tuple[int, int] = (168, 168),
                 health_history_length: int = 8,
                 num_health_features: int = 4):
        """
        Initialize hybrid replay buffer.
        
        Args:
            capacity: Maximum number of transitions to store
            frame_stack: Number of consecutive frames to stack
            img_size: Size of preprocessed frames (height, width)
            health_history_length: Number of health frames to track
            num_health_features: Number of features per health timestep
        """
        self.capacity = capacity
        self.frame_stack = frame_stack
        self.img_height, self.img_width = img_size
        self.health_history_length = health_history_length
        self.num_health_features = num_health_features
        
        # Screenshot storage (memory efficient uint8)
        self.screenshots = np.zeros(
            (capacity, frame_stack, self.img_height, self.img_width), 
            dtype=np.uint8
        )
        self.next_screenshots = np.zeros(
            (capacity, frame_stack, self.img_height, self.img_width), 
            dtype=np.uint8
        )
        
        # Health history storage
        self.health_history = np.zeros(
            (capacity, health_history_length, num_health_features), 
            dtype=np.float32
        )
        self.next_health_history = np.zeros(
            (capacity, health_history_length, num_health_features), 
            dtype=np.float32
        )
        
        # Action, reward, done storage
        self.actions = np.zeros(capacity, dtype=np.int32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=bool)
        
        # Buffer management
        self.position = 0
        self.size = 0
        self.is_full = False
        
        # Calculate memory usage
        screenshots_mb = (2 * capacity * frame_stack * self.img_height * self.img_width) / (1024**2)
        health_mb = (2 * capacity * health_history_length * num_health_features * 4) / (1024**2)  # 4 bytes per float32
        other_mb = (capacity * (4 + 4 + 1)) / (1024**2)  # action + reward + done
        total_mb = screenshots_mb + health_mb + other_mb
        
        print(f"🧠 Hybrid ReplayBuffer initialized:")
        print(f"   Capacity: {capacity:,} transitions")
        print(f"   Screenshots: {frame_stack} × {img_size}")
        print(f"   Health history: {health_history_length} frames")
        print(f"   Memory usage: ~{total_mb:.1f} MB")
    
    def add_transition(self, 
                      screenshots: np.ndarray,
                      health_history: np.ndarray,
                      action: int,
                      reward: float,
                      next_screenshots: np.ndarray,
                      next_health_history: np.ndarray,
                      done: bool):
        """
        Add a hybrid transition to the buffer.
        
        Args:
            screenshots: Current stacked screenshots (frame_stack, height, width) as float32 [0,1]
            health_history: Current health history (health_history_length, num_health_features) as float32
            action: Action taken (0 to num_actions-1)
            reward: Reward received
            next_screenshots: Next stacked screenshots (frame_stack, height, width) as float32 [0,1]
            next_health_history: Next health history (health_history_length, num_health_features) as float32
            done: Whether episode ended
        """
        # Validate input shapes
        expected_screenshot_shape = (self.frame_stack, self.img_height, self.img_width)
        expected_health_shape = (self.health_history_length, self.num_health_features)
        
        assert screenshots.shape == expected_screenshot_shape, \
            f"Screenshots shape {screenshots.shape} doesn't match expected {expected_screenshot_shape}"
        assert next_screenshots.shape == expected_screenshot_shape, \
            f"Next screenshots shape {next_screenshots.shape} doesn't match expected {expected_screenshot_shape}"
        assert health_history.shape == expected_health_shape, \
            f"Health history shape {health_history.shape} doesn't match expected {expected_health_shape}"
        assert next_health_history.shape == expected_health_shape, \
            f"Next health history shape {next_health_history.shape} doesn't match expected {expected_health_shape}"
        
        # Convert screenshots from float32 [0,1] to uint8 [0,255] for storage efficiency
        screenshots_uint8 = (screenshots * 255).astype(np.uint8)
        next_screenshots_uint8 = (next_screenshots * 255).astype(np.uint8)
        
        # Store transition
        self.screenshots[self.position] = screenshots_uint8
        self.health_history[self.position] = health_history
        self.actions[self.position] = action
        self.rewards[self.position] = reward
        self.next_screenshots[self.position] = next_screenshots_uint8
        self.next_health_history[self.position] = next_health_history
        self.dones[self.position] = done
        
        # Update buffer management
        self.position = (self.position + 1) % self.capacity
        if self.size < self.capacity:
            self.size += 1
        else:
            self.is_full = True
    
    def sample_batch(self, batch_size: int = 32) -> Optional[Tuple[np.ndarray, ...]]:
        """
        Sample a random batch of transitions.
        
        Args:
            batch_size: Number of transitions to sample
            
        Returns:
            Tuple of (screenshots, health_history, actions, rewards, 
                     next_screenshots, next_health_history, dones) or None if not enough data
        """
        if self.size < batch_size:
            return None
        
        # Sample random indices
        indices = np.random.choice(self.size, batch_size, replace=False)
        
        # Extract batch data and convert screenshots back to float32 [0,1]
        batch_screenshots = self.screenshots[indices].astype(np.float32) / 255.0
        batch_health_history = self.health_history[indices]
        batch_actions = self.actions[indices]
        batch_rewards = self.rewards[indices]
        batch_next_screenshots = self.next_screenshots[indices].astype(np.float32) / 255.0
        batch_next_health_history = self.next_health_history[indices]
        batch_dones = self.dones[indices]
        
        return (batch_screenshots, batch_health_history, batch_actions, batch_rewards,
                batch_next_screenshots, batch_next_health_history, batch_dones)
    
    def size(self) -> int:
        """Get current number of stored transitions"""
        return self.size
    
    def __len__(self) -> int:
        """Get current number of stored transitions"""
        return self.size
    
    def is_ready(self, min_size: int = 1000) -> bool:
        """Check if buffer has enough transitions for training"""
        return self.size >= min_size
    
    def get_memory_usage(self) -> dict:
        """Get detailed memory usage information"""
        screenshots_mb = (2 * self.capacity * self.frame_stack * self.img_height * self.img_width) / (1024**2)
        health_mb = (2 * self.capacity * self.health_history_length * self.num_health_features * 4) / (1024**2)
        other_mb = (self.capacity * (4 + 4 + 1)) / (1024**2)
        total_mb = screenshots_mb + health_mb + other_mb
        
        return {
            'screenshots_mb': screenshots_mb,
            'health_mb': health_mb,
            'other_mb': other_mb,
            'total_mb': total_mb,
            'capacity': self.capacity,
            'current_size': self.size,
            'fill_percentage': (self.size / self.capacity) * 100
        }


def test_hybrid_replay_buffer():
    """Test the hybrid replay buffer functionality"""
    print("🧪 Testing Hybrid Replay Buffer...")
    
    # Create small buffer for testing
    buffer = HybridReplayBuffer(
        capacity=100,
        frame_stack=4,
        img_size=(32, 32),
        health_history_length=4
    )
    
    print(f"✅ Buffer created: {len(buffer)} transitions")
    
    # Add some test transitions
    for i in range(10):
        # Create mock data
        screenshots = np.random.rand(4, 32, 32).astype(np.float32)
        health_history = np.random.rand(4, 4).astype(np.float32)
        action = i % 10
        reward = float(i)
        next_screenshots = np.random.rand(4, 32, 32).astype(np.float32)
        next_health_history = np.random.rand(4, 4).astype(np.float32)
        done = (i == 9)
        
        buffer.add_transition(
            screenshots, health_history, action, reward,
            next_screenshots, next_health_history, done
        )
    
    print(f"✅ Added 10 transitions: {len(buffer)} total")
    
    # Test sampling
    batch = buffer.sample_batch(5)
    if batch is not None:
        screenshots, health, actions, rewards, next_screenshots, next_health, dones = batch
        print(f"✅ Sampled batch successfully:")
        print(f"   Screenshots: {screenshots.shape} dtype={screenshots.dtype}")
        print(f"   Health: {health.shape} dtype={health.dtype}")
        print(f"   Actions: {actions.shape}")
        print(f"   Rewards: {rewards.shape}")
        print(f"   Dones: {dones.shape}")
        print(f"   Value range: screenshots [{screenshots.min():.3f}, {screenshots.max():.3f}]")
    else:
        print("❌ Failed to sample batch")
    
    # Test memory usage
    memory_info = buffer.get_memory_usage()
    print(f"✅ Memory usage: {memory_info['total_mb']:.1f} MB")
    
    print("🎯 Hybrid Replay Buffer test complete!")


if __name__ == "__main__":
    test_hybrid_replay_buffer()