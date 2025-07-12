#!/usr/bin/env python3
"""
Experience Replay Buffer for DQN

Efficiently stores and samples visual experiences for Deep Q-Network training.
Optimized for screenshot-based learning with frame stacking.
"""

import numpy as np
import random
from collections import deque
from typing import Tuple, Optional, List
import cv2


class ReplayBuffer:
    """
    Experience replay buffer for DQN with vision-based states.
    
    Key Features:
    - Memory efficient storage (uint8 frames)
    - Frame stacking for temporal information
    - Circular buffer for fixed memory usage
    - Random sampling to break correlation
    """
    
    def __init__(self, 
                 capacity: int = 50000,
                 frame_stack: int = 4,
                 img_size: Tuple[int, int] = (168, 168)):
        """
        Initialize replay buffer.
        
        Args:
            capacity: Maximum number of transitions to store
            frame_stack: Number of consecutive frames to stack
            img_size: Size of preprocessed frames (height, width)
        """
        self.capacity = capacity
        self.frame_stack = frame_stack
        self.img_height, self.img_width = img_size
        
        # Circular buffer for transitions
        self.states = np.zeros((capacity, self.img_height, self.img_width), dtype=np.uint8)
        self.actions = np.zeros(capacity, dtype=np.int32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros((capacity, self.img_height, self.img_width), dtype=np.uint8)
        self.dones = np.zeros(capacity, dtype=np.bool_)
        
        # Buffer management
        self.position = 0
        self.size = 0
        self.is_full = False
        
        print(f"🧠 ReplayBuffer initialized:")
        print(f"   Capacity: {capacity:,} transitions")
        print(f"   Frame stack: {frame_stack}")
        print(f"   Image size: {img_size}")
        print(f"   Memory usage: ~{self._estimate_memory_mb():.1f} MB")
    
    def add(self, 
            state: np.ndarray, 
            action: int, 
            reward: float, 
            next_state: np.ndarray, 
            done: bool):
        """
        Add a transition to the buffer.
        
        Args:
            state: Current frame (height, width) as uint8
            action: Action taken (0 to num_actions-1)
            reward: Reward received
            next_state: Next frame (height, width) as uint8
            done: Whether episode ended
        """
        # Validate input shapes
        assert state.shape == (self.img_height, self.img_width), \
            f"State shape {state.shape} doesn't match expected {(self.img_height, self.img_width)}"
        assert next_state.shape == (self.img_height, self.img_width), \
            f"Next state shape {next_state.shape} doesn't match expected {(self.img_height, self.img_width)}"
        assert state.dtype == np.uint8, f"State should be uint8, got {state.dtype}"
        assert next_state.dtype == np.uint8, f"Next state should be uint8, got {next_state.dtype}"
        
        # Store transition
        self.states[self.position] = state
        self.actions[self.position] = action
        self.rewards[self.position] = reward
        self.next_states[self.position] = next_state
        self.dones[self.position] = done
        
        # Update buffer management
        self.position = (self.position + 1) % self.capacity
        if self.size < self.capacity:
            self.size += 1
        else:
            self.is_full = True
    
    def sample(self, batch_size: int = 32) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Sample a batch of transitions with frame stacking.
        
        Args:
            batch_size: Number of transitions to sample
            
        Returns:
            Tuple of (stacked_states, actions, rewards, stacked_next_states, dones)
            - stacked_states: (batch_size, frame_stack, height, width)
            - actions: (batch_size,)
            - rewards: (batch_size,)
            - stacked_next_states: (batch_size, frame_stack, height, width)
            - dones: (batch_size,)
        """
        if self.size < batch_size:
            raise ValueError(f"Not enough samples in buffer: {self.size} < {batch_size}")
        
        # Sample random indices (ensuring we can stack frames)
        indices = self._sample_valid_indices(batch_size)
        
        # Get stacked frames for states and next states
        stacked_states = self._get_stacked_frames(indices, use_next_state=False)
        stacked_next_states = self._get_stacked_frames(indices, use_next_state=True)
        
        # Get other components
        actions = self.actions[indices]
        rewards = self.rewards[indices]
        dones = self.dones[indices]
        
        return stacked_states, actions, rewards, stacked_next_states, dones
    
    def _sample_valid_indices(self, batch_size: int) -> np.ndarray:
        """
        Sample indices that allow for proper frame stacking.
        
        Ensures sampled transitions have enough history for frame stacking
        and don't cross episode boundaries.
        """
        valid_indices = []
        max_attempts = batch_size * 10  # Prevent infinite loops
        attempts = 0
        
        while len(valid_indices) < batch_size and attempts < max_attempts:
            # Sample random index
            if self.is_full:
                # Can sample from anywhere in circular buffer
                idx = random.randint(self.frame_stack - 1, self.capacity - 1)
            else:
                # Can only sample from filled portion
                idx = random.randint(self.frame_stack - 1, self.size - 1)
            
            # Check if we can stack frames without crossing episode boundaries
            if self._is_valid_for_stacking(idx):
                valid_indices.append(idx)
            
            attempts += 1
        
        if len(valid_indices) < batch_size:
            print(f"⚠️ Warning: Could only sample {len(valid_indices)}/{batch_size} valid indices")
        
        return np.array(valid_indices[:batch_size])
    
    def _is_valid_for_stacking(self, idx: int) -> bool:
        """
        Check if an index is valid for frame stacking.
        
        Args:
            idx: Index to check
            
        Returns:
            True if we can stack frames without crossing episode boundaries
        """
        # Check if we have enough history
        if idx < self.frame_stack - 1:
            return False
        
        # Check for episode boundaries in the frame stack history
        for i in range(1, self.frame_stack):
            prev_idx = idx - i
            if self.is_full:
                prev_idx = prev_idx % self.capacity
            
            # If previous frame ended episode, can't stack
            if self.dones[prev_idx]:
                return False
        
        return True
    
    def _get_stacked_frames(self, indices: np.ndarray, use_next_state: bool = False) -> np.ndarray:
        """
        Get frame stacks for given indices.
        
        Args:
            indices: Array of indices to get stacks for
            use_next_state: If True, stack next_states instead of states
            
        Returns:
            Stacked frames with shape (batch_size, frame_stack, height, width)
        """
        batch_size = len(indices)
        stacked = np.zeros((batch_size, self.frame_stack, self.img_height, self.img_width), dtype=np.float32)
        
        source_array = self.next_states if use_next_state else self.states
        
        for i, idx in enumerate(indices):
            for j in range(self.frame_stack):
                frame_idx = idx - (self.frame_stack - 1 - j)
                
                if self.is_full:
                    frame_idx = frame_idx % self.capacity
                
                # Convert uint8 to float32 and normalize to [0, 1]
                stacked[i, j] = source_array[frame_idx].astype(np.float32) / 255.0
        
        return stacked
    
    def _estimate_memory_mb(self) -> float:
        """Estimate memory usage in megabytes."""
        state_memory = self.capacity * self.img_height * self.img_width * 2  # states + next_states
        other_memory = self.capacity * (4 + 4 + 1)  # actions + rewards + dones
        total_bytes = state_memory + other_memory
        return total_bytes / (1024 * 1024)
    
    def can_sample(self, batch_size: int = 32) -> bool:
        """Check if buffer has enough samples for training."""
        return self.size >= max(batch_size, self.frame_stack * 2)
    
    def __len__(self) -> int:
        """Return current buffer size."""
        return self.size
    
    def get_stats(self) -> dict:
        """Get buffer statistics."""
        return {
            'size': self.size,
            'capacity': self.capacity,
            'position': self.position,
            'is_full': self.is_full,
            'memory_mb': self._estimate_memory_mb(),
            'fill_percentage': (self.size / self.capacity) * 100
        }


def preprocess_frame(frame: np.ndarray, 
                    target_size: Tuple[int, int] = (168, 168)) -> np.ndarray:
    """
    Preprocess a screenshot for DQN training.
    
    Args:
        frame: Raw screenshot as BGR numpy array
        target_size: Target size (height, width)
        
    Returns:
        Preprocessed frame as uint8 grayscale
    """
    # Convert BGR to RGB if needed (OpenCV uses BGR)
    if len(frame.shape) == 3 and frame.shape[2] == 3:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Convert to grayscale
    if len(frame.shape) == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    else:
        gray = frame
    
    # Resize to target size
    resized = cv2.resize(gray, target_size, interpolation=cv2.INTER_AREA)
    
    # Ensure uint8 format
    if resized.dtype != np.uint8:
        resized = np.clip(resized, 0, 255).astype(np.uint8)
    
    return resized


def test_replay_buffer():
    """Test the replay buffer functionality."""
    print("🧪 Testing ReplayBuffer...")
    
    # Create buffer
    buffer = ReplayBuffer(capacity=1000, frame_stack=4, img_size=(84, 84))
    
    # Add some dummy transitions
    for i in range(100):
        state = np.random.randint(0, 255, (84, 84), dtype=np.uint8)
        action = random.randint(0, 9)
        reward = random.uniform(-1, 1)
        next_state = np.random.randint(0, 255, (84, 84), dtype=np.uint8)
        done = random.random() < 0.1  # 10% chance of episode end
        
        buffer.add(state, action, reward, next_state, done)
    
    print(f"✅ Added 100 transitions")
    print(f"Buffer stats: {buffer.get_stats()}")
    
    # Test sampling
    if buffer.can_sample(32):
        states, actions, rewards, next_states, dones = buffer.sample(32)
        print(f"✅ Sampled batch:")
        print(f"   States shape: {states.shape}")
        print(f"   Actions shape: {actions.shape}")
        print(f"   Rewards shape: {rewards.shape}")
        print(f"   Next states shape: {next_states.shape}")
        print(f"   Dones shape: {dones.shape}")
    else:
        print("❌ Cannot sample yet")
    
    print("🎉 ReplayBuffer test complete!")


if __name__ == "__main__":
    test_replay_buffer()