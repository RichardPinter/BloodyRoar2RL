#!/usr/bin/env python3
"""
DQN Slow RL Environment

DQN-specific implementation of the slow RL environment.
Returns (screenshots, health_history) tuples for visual RL agents.

Integrates with HybridStateManager to provide synchronized screenshot + health data.
"""

import numpy as np
from typing import Tuple
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.base_slow_rl_environment import BaseSlowRLEnvironment, SlowState, GameState
from dqn.hybrid_state import HybridStateManager
from detection.window_capture import WindowCapture


class DQNSlowRLEnvironment(BaseSlowRLEnvironment):
    """
    DQN-specific slow RL environment.
    
    Returns (screenshots, health_history) tuples for visual RL agents.
    Uses HybridStateManager to synchronize screenshot and health data.
    """
    
    def __init__(self, 
                 frame_stack_size: int = 8,
                 img_size: Tuple[int, int] = (168, 168),
                 health_history_length: int = 8,
                 observation_window_seconds: int = 8,
                 window_title: str = "Bloody Roar II (USA) [PlayStation] - BizHawk"):
        """
        Initialize DQN environment.
        
        Args:
            frame_stack_size: Number of screenshot frames to stack
            img_size: Target size for screenshots (height, width)  
            health_history_length: Number of health frames to track
            observation_window_seconds: How many seconds to observe (1 screenshot per second)
            window_title: Game window title for screenshot capture
        """
        # Initialize base environment with configurable observation window
        super().__init__(observation_window=observation_window_seconds)
        
        self.frame_stack_size = frame_stack_size
        self.img_size = img_size
        self.health_history_length = health_history_length
        
        # Initialize screenshot capture
        self.window_capture = WindowCapture(window_title)
        if not self.window_capture.is_valid:
            print(f"⚠️ Warning: Game window '{window_title}' not found!")
            print("   Screenshot capture will be disabled.")
            self.window_capture = None
        
        # Initialize hybrid state manager
        self.hybrid_state_manager = HybridStateManager(
            frame_stack_size=frame_stack_size,
            img_size=img_size,
            health_history_length=health_history_length
        )
        
        print(f"🎯 DQN Slow RL Environment initialized (hybrid visual + health input)")
        print(f"   Screenshot frames: {frame_stack_size} × {img_size}")
        print(f"   Health history: {health_history_length} frames")
    
    def _observation_to_output(self, state: SlowState) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert SlowState to (screenshots, health_history) tuple for DQN agent.
        
        Returns:
            Tuple of (stacked_screenshots, health_history):
            - stacked_screenshots: (frame_stack_size, height, width) as float32 [0,1]
            - health_history: (health_history_length, 4) as float32
        """
        if not self.hybrid_state_manager.is_ready():
            print("⚠️ Warning: HybridStateManager not ready, returning zeros")
            # Return zero arrays with correct shapes
            zero_screenshots = np.zeros((self.frame_stack_size, *self.img_size), dtype=np.float32)
            zero_health = np.zeros((self.health_history_length, 4), dtype=np.float32)
            return zero_screenshots, zero_health
        
        # Get current hybrid state
        screenshots, health_history = self.hybrid_state_manager.get_current_state()
        return screenshots, health_history
    
    def get_observation_space_size(self) -> int:
        """
        Get size of observation space for DQN agent.
        
        Note: DQN uses tuple output, so this is mainly for compatibility.
        Returns total number of elements in both arrays combined.
        """
        screenshot_elements = self.frame_stack_size * self.img_size[0] * self.img_size[1]
        health_elements = self.health_history_length * 4
        return screenshot_elements + health_elements
    
    def _on_observation_collected(self, game_state: GameState):
        """
        Hook called when observation is collected.
        Captures screenshot and adds frame to HybridStateManager.
        
        Args:
            game_state: Current game state with health information
        """
        if self.window_capture is None:
            print("⚠️ No screenshot capture available, using dummy data")
            # Create dummy screenshot for testing
            screenshot = np.zeros((100, 150, 3), dtype=np.uint8)
        else:
            # Capture current screenshot
            screenshot = self.window_capture.capture()
            if screenshot is None:
                print("⚠️ Screenshot capture failed, using dummy data")
                screenshot = np.zeros((100, 150, 3), dtype=np.uint8)
        
        # Add frame to hybrid state manager
        # Create HealthState from GameState data
        from detection.health_detector import HealthState
        health_state = HealthState(
            p1_health=game_state.p1_health,
            p2_health=game_state.p2_health,
            p1_pixels=0,  # Not available from GameState, but not needed for DQN
            p2_pixels=0   # Not available from GameState, but not needed for DQN
        )
        
        self.hybrid_state_manager.add_frame(screenshot, health_state)
    
    def reset(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Reset environment and return initial hybrid state.
        
        Returns:
            Tuple of (initial_screenshots, initial_health_history)
        """
        # Reset hybrid state manager
        self.hybrid_state_manager.reset()
        
        # Call parent reset (which will trigger _on_observation_collected)
        return super().reset()
    
    def close(self):
        """Clean up environment"""
        super().close()
        if self.window_capture:
            # WindowCapture doesn't have a close method, just clear the reference
            self.window_capture = None
        print("DQN Slow RL Environment closed")


def test_dqn_environment():
    """Test the DQN-specific environment"""
    print("🧪 Testing DQN Slow RL Environment")
    print("=" * 60)
    print("Testing hybrid (screenshots + health) output for DQN agents...")
    
    env = DQNSlowRLEnvironment(
        frame_stack_size=4,
        img_size=(84, 84),  # Smaller for testing
        health_history_length=4
    )
    
    try:
        print(f"✅ DQN Environment created successfully")
        print(f"   Observation space size: {env.get_observation_space_size()}")
        print(f"   Action space size: {env.get_action_space_size()}")
        print(f"   Actions: {env.get_actions()}")
        
        print("\n📊 Testing without game connection (dry run)...")
        
        # Test observation output format by manually triggering hybrid state manager
        from detection.health_detector import HealthState
        import time
        
        # Add some dummy data to hybrid state manager
        for i in range(6):  # Add more than frame stack size
            dummy_screenshot = np.random.randint(0, 255, (100, 150, 3), dtype=np.uint8)
            dummy_health = HealthState(
                p1_health=100.0 - i*5,
                p2_health=100.0 - i*3,
                p1_pixels=i*10,
                p2_pixels=i*6
            )
            
            env.hybrid_state_manager.add_frame(dummy_screenshot, dummy_health)
            print(f"   Added dummy frame {i+1}: P1={dummy_health.p1_health:.1f}% P2={dummy_health.p2_health:.1f}%")
        
        # Test output format
        screenshots, health_history = env.hybrid_state_manager.get_current_state()
        
        print(f"\n📸 Screenshot output:")
        print(f"   Type: {type(screenshots)}")
        print(f"   Shape: {screenshots.shape}")
        print(f"   Dtype: {screenshots.dtype}")
        print(f"   Value range: [{screenshots.min():.3f}, {screenshots.max():.3f}]")
        
        print(f"\n🩺 Health history output:")
        print(f"   Type: {type(health_history)}")
        print(f"   Shape: {health_history.shape}")
        print(f"   Dtype: {health_history.dtype}")
        print(f"   Latest health: P1={health_history[-1, 0]:.1f}% P2={health_history[-1, 1]:.1f}%")
        print(f"   Latest deltas: P1={health_history[-1, 2]:+.1f}% P2={health_history[-1, 3]:+.1f}%")
        
        # Verify output formats
        assert isinstance(screenshots, np.ndarray), "Screenshots should be numpy array"
        assert isinstance(health_history, np.ndarray), "Health history should be numpy array"
        assert screenshots.shape == (4, 84, 84), f"Screenshot shape should be (4, 84, 84), got {screenshots.shape}"
        assert health_history.shape == (4, 4), f"Health shape should be (4, 4), got {health_history.shape}"
        assert screenshots.dtype == np.float32, f"Screenshot dtype should be float32, got {screenshots.dtype}"
        assert health_history.dtype == np.float32, f"Health dtype should be float32, got {health_history.dtype}"
        assert 0.0 <= screenshots.min() <= screenshots.max() <= 1.0, "Screenshots should be normalized [0,1]"
        
        print("\n✅ DQN Environment test passed!")
        print("   - Correct hybrid tuple output")
        print("   - Screenshots properly normalized")
        print("   - Health history with deltas")
        print("   - Compatible with DQN agents")
        
        return True
        
    except Exception as e:
        print(f"❌ DQN Environment test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        env.close()


if __name__ == "__main__":
    test_dqn_environment()