#!/usr/bin/env python3
"""
CNN Vision Network for DQN

Processes stacked game screenshots to output Q-values for each action.
Optimized for CPU training with fighting game specifics.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple


class DQNVisionNetwork(nn.Module):
    """
    Convolutional Neural Network for processing game screenshots in DQN.
    
    Architecture optimized for:
    - CPU training (smaller filter counts)
    - Fighting game patterns (character positions, health bars)
    - 1 FPS gameplay (temporal patterns across 4 frames)
    """
    
    def __init__(self, 
                 frame_stack: int = 4,
                 img_size: Tuple[int, int] = (168, 168),
                 num_actions: int = 10,
                 hidden_size: int = 256):
        """
        Initialize CNN vision network.
        
        Args:
            frame_stack: Number of stacked frames (temporal history)
            img_size: Input image dimensions (height, width)
            num_actions: Number of possible actions
            hidden_size: Size of fully connected layer
        """
        super(DQNVisionNetwork, self).__init__()
        
        self.frame_stack = frame_stack
        self.img_height, self.img_width = img_size
        self.num_actions = num_actions
        self.hidden_size = hidden_size
        
        # Convolutional layers for feature extraction
        self.conv1 = nn.Conv2d(
            in_channels=frame_stack,  # 4 stacked frames
            out_channels=16,          # Small for CPU efficiency
            kernel_size=8,            # Large kernels for downsampling
            stride=4,                 # Aggressive stride to reduce spatial dims
            padding=2
        )
        
        self.conv2 = nn.Conv2d(
            in_channels=16,
            out_channels=32,          # Moderate increase in features
            kernel_size=4,
            stride=2,
            padding=1
        )
        
        # Calculate size after convolutions for fully connected layer
        self.conv_output_size = self._calculate_conv_output_size()
        
        # Fully connected layers for decision making
        self.fc1 = nn.Linear(self.conv_output_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_actions)
        
        # Initialize weights
        self._initialize_weights()
        
        print(f"🧠 DQN Vision Network initialized:")
        print(f"   Input: ({frame_stack}, {img_size[0]}, {img_size[1]})")
        print(f"   Conv output size: {self.conv_output_size}")
        print(f"   Hidden size: {hidden_size}")
        print(f"   Output: {num_actions} Q-values")
        print(f"   Total parameters: {self._count_parameters():,}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor (batch_size, frame_stack, height, width)
            
        Returns:
            Q-values for each action (batch_size, num_actions)
        """
        # Validate input shape
        expected_shape = (x.shape[0], self.frame_stack, self.img_height, self.img_width)
        if x.shape != expected_shape:
            raise ValueError(f"Input shape {x.shape} doesn't match expected {expected_shape}")
        
        # Convolutional feature extraction
        x = F.relu(self.conv1(x))  # (batch, 16, 42, 42) for 168x168 input
        x = F.relu(self.conv2(x))  # (batch, 32, 21, 21)
        
        # Flatten spatial dimensions
        x = x.view(x.size(0), -1)  # (batch, 32*21*21)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))    # (batch, hidden_size)
        q_values = self.fc2(x)     # (batch, num_actions)
        
        return q_values
    
    def _calculate_conv_output_size(self) -> int:
        """Calculate the output size after convolutional layers."""
        # Create dummy input to trace through conv layers
        dummy_input = torch.zeros(1, self.frame_stack, self.img_height, self.img_width)
        
        with torch.no_grad():
            x = F.relu(self.conv1(dummy_input))
            x = F.relu(self.conv2(x))
            conv_output_size = x.numel()  # Total number of elements
        
        return conv_output_size
    
    def _initialize_weights(self):
        """Initialize network weights using Xavier initialization."""
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
    
    def _count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def predict_q_values(self, stacked_frames: np.ndarray) -> np.ndarray:
        """
        Predict Q-values for a single state (convenience method).
        
        Args:
            stacked_frames: Numpy array (frame_stack, height, width) normalized to [0,1]
            
        Returns:
            Q-values array (num_actions,)
        """
        # Convert to tensor and add batch dimension
        if isinstance(stacked_frames, np.ndarray):
            tensor_input = torch.FloatTensor(stacked_frames).unsqueeze(0)  # Add batch dim
        else:
            tensor_input = stacked_frames.unsqueeze(0)
        
        # Forward pass
        self.eval()
        with torch.no_grad():
            q_values = self.forward(tensor_input)
        
        return q_values.squeeze(0).numpy()  # Remove batch dim and convert to numpy
    
    def select_action(self, stacked_frames: np.ndarray, epsilon: float = 0.0) -> int:
        """
        Select action using epsilon-greedy policy.
        
        Args:
            stacked_frames: Current state as stacked frames
            epsilon: Exploration probability (0 = greedy, 1 = random)
            
        Returns:
            Selected action index
        """
        if np.random.random() < epsilon:
            # Random exploration
            return np.random.randint(0, self.num_actions)
        else:
            # Greedy action selection
            q_values = self.predict_q_values(stacked_frames)
            return int(np.argmax(q_values))
    
    def get_action_values_with_names(self, stacked_frames: np.ndarray, action_names: list = None) -> dict:
        """
        Get Q-values with human-readable action names.
        
        Args:
            stacked_frames: Current state
            action_names: List of action names (e.g., ['left', 'right', ...])
            
        Returns:
            Dictionary mapping action names to Q-values
        """
        if action_names is None:
            action_names = ['left', 'right', 'jump', 'squat', 'transform', 
                          'kick', 'punch', 'special', 'block', 'throw']
        
        q_values = self.predict_q_values(stacked_frames)
        
        return {name: float(q_val) for name, q_val in zip(action_names, q_values)}


class VisionNetworkTester:
    """Test suite for DQN Vision Network"""
    
    def __init__(self):
        self.network = None
        self.test_results = {}
    
    def test_network_creation(self):
        """Test network creation with different parameters."""
        print("\n🏗️ Testing Network Creation...")
        
        try:
            # Test default parameters
            self.network = DQNVisionNetwork()
            print("✅ Default network created successfully")
            
            # Test custom parameters
            custom_network = DQNVisionNetwork(
                frame_stack=2,
                img_size=(84, 84),
                num_actions=6,
                hidden_size=128
            )
            print("✅ Custom network created successfully")
            
            self.test_results['creation'] = True
            return True
            
        except Exception as e:
            print(f"❌ Network creation failed: {e}")
            self.test_results['creation'] = False
            return False
    
    def test_forward_pass(self):
        """Test forward pass with dummy data."""
        print("\n🔄 Testing Forward Pass...")
        
        if self.network is None:
            print("❌ No network available for testing")
            return False
        
        try:
            # Create dummy input batch
            batch_size = 4
            dummy_input = torch.randn(batch_size, 4, 168, 168)
            
            # Forward pass
            self.network.eval()
            with torch.no_grad():
                output = self.network(dummy_input)
            
            print(f"✅ Forward pass successful")
            print(f"   Input shape: {dummy_input.shape}")
            print(f"   Output shape: {output.shape}")
            print(f"   Expected output shape: ({batch_size}, {self.network.num_actions})")
            
            # Validate output shape
            expected_shape = (batch_size, self.network.num_actions)
            if output.shape == expected_shape:
                print("✅ Output shape correct")
                self.test_results['forward_pass'] = True
                return True
            else:
                print(f"❌ Output shape mismatch: {output.shape} vs {expected_shape}")
                self.test_results['forward_pass'] = False
                return False
                
        except Exception as e:
            print(f"❌ Forward pass failed: {e}")
            self.test_results['forward_pass'] = False
            return False
    
    def test_action_selection(self):
        """Test action selection methods."""
        print("\n🎯 Testing Action Selection...")
        
        if self.network is None:
            print("❌ No network available for testing")
            return False
        
        try:
            # Create dummy state
            dummy_state = np.random.rand(4, 168, 168).astype(np.float32)
            
            # Test Q-value prediction
            q_values = self.network.predict_q_values(dummy_state)
            print(f"✅ Q-value prediction: shape {q_values.shape}")
            print(f"   Q-values: {q_values}")
            
            # Test greedy action selection
            action_greedy = self.network.select_action(dummy_state, epsilon=0.0)
            print(f"✅ Greedy action: {action_greedy}")
            
            # Test random action selection
            action_random = self.network.select_action(dummy_state, epsilon=1.0)
            print(f"✅ Random action: {action_random}")
            
            # Test action values with names
            action_dict = self.network.get_action_values_with_names(dummy_state)
            print(f"✅ Action values with names:")
            for action, value in sorted(action_dict.items(), key=lambda x: x[1], reverse=True):
                print(f"      {action}: {value:.3f}")
            
            self.test_results['action_selection'] = True
            return True
            
        except Exception as e:
            print(f"❌ Action selection failed: {e}")
            self.test_results['action_selection'] = False
            return False
    
    def test_performance(self):
        """Test network performance and timing."""
        print("\n⚡ Testing Performance...")
        
        if self.network is None:
            print("❌ No network available for testing")
            return False
        
        try:
            import time
            
            # Test single inference time
            dummy_state = np.random.rand(4, 168, 168).astype(np.float32)
            
            # Warm up
            for _ in range(5):
                self.network.predict_q_values(dummy_state)
            
            # Time multiple inferences
            num_tests = 50
            start_time = time.time()
            
            for _ in range(num_tests):
                q_values = self.network.predict_q_values(dummy_state)
            
            end_time = time.time()
            avg_time = (end_time - start_time) / num_tests
            
            print(f"✅ Performance test complete")
            print(f"   Average inference time: {avg_time*1000:.2f}ms")
            print(f"   Throughput: {1/avg_time:.1f} inferences/second")
            
            # Check if fast enough for 1 FPS
            if avg_time < 0.5:  # Should be much less than 1 second
                print(f"✅ Fast enough for 1 FPS gameplay!")
                self.test_results['performance'] = True
                return True
            else:
                print(f"⚠️ Might be too slow for real-time gameplay")
                self.test_results['performance'] = False
                return False
                
        except Exception as e:
            print(f"❌ Performance test failed: {e}")
            self.test_results['performance'] = False
            return False
    
    def run_all_tests(self):
        """Run complete test suite."""
        print("🧪 CNN VISION NETWORK TEST SUITE")
        print("="*50)
        
        tests = [
            ('Network Creation', self.test_network_creation),
            ('Forward Pass', self.test_forward_pass),
            ('Action Selection', self.test_action_selection),
            ('Performance', self.test_performance)
        ]
        
        all_passed = True
        for test_name, test_func in tests:
            success = test_func()
            if not success:
                all_passed = False
        
        # Summary
        print("\n🎉 TEST SUMMARY")
        print("="*50)
        for test_name, result in self.test_results.items():
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"{test_name}: {status}")
        
        if all_passed:
            print("\n🎯 All tests passed! CNN Vision Network ready for DQN training.")
        else:
            print("\n⚠️ Some tests failed. Check the output above for details.")
        
        return all_passed


def main():
    """Main testing function."""
    print("🧠 CNN VISION NETWORK FOR DQN")
    print("="*40)
    print("Testing the neural network that processes game screenshots.")
    print("="*40)
    
    # Run tests
    tester = VisionNetworkTester()
    success = tester.run_all_tests()
    
    if success:
        print("\n✅ CNN Vision Network is ready!")
        print("Next step: Implement the DQN Agent that uses this network.")
    else:
        print("\n❌ Fix the issues above before proceeding.")


if __name__ == "__main__":
    main()