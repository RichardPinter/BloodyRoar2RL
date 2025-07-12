#!/usr/bin/env python3
"""
Hybrid State Manager

Synchronizes screenshots and health history into unified state representation.
Bridges visual data (screenshots) with precise numerical data (health tracking).

Single Responsibility:
- Maintain synchronized screenshot + health frame stacks
- Provide clean state interface for DQN agent
- Handle frame timing and alignment
"""

import numpy as np
from typing import Tuple, Optional
from collections import deque
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from detection.health_detector import HealthState
from dqn.health_history import HealthHistoryTracker
from dqn.replay_buffer import preprocess_frame


class HybridStateManager:
    """
    Manages synchronized screenshot and health history for DQN input.
    
    Combines visual patterns (screenshots) with precise numerical data (health tracking)
    into a unified state representation that DQN can learn from.
    """
    
    def __init__(self, 
                 frame_stack_size: int = 8,
                 img_size: Tuple[int, int] = (168, 168),
                 health_history_length: int = 8):
        """
        Initialize hybrid state manager.
        
        Args:
            frame_stack_size: Number of screenshot frames to stack
            img_size: Target size for preprocessed screenshots (height, width)
            health_history_length: Number of health frames to track
        """
        self.frame_stack_size = frame_stack_size
        self.img_height, self.img_width = img_size
        self.health_history_length = health_history_length
        
        # Screenshot frame stacking (circular buffer)
        self.screenshot_frames = deque(maxlen=frame_stack_size)
        
        # Health history tracking
        self.health_tracker = HealthHistoryTracker(history_length=health_history_length)
        
        # Frame synchronization
        self.frame_count = 0
        
        print(f"🔗 HybridStateManager initialized:")
        print(f"   Screenshot frames: {frame_stack_size} × {img_size}")
        print(f"   Health history: {health_history_length} frames")
        print(f"   Output: Visual({frame_stack_size}, {img_size[0]}, {img_size[1]}) + Health({health_history_length}, 4)")
    
    def add_frame(self, 
                  screenshot: np.ndarray, 
                  health_state: HealthState):
        """
        Add a new synchronized frame (screenshot + health data).
        
        Args:
            screenshot: Raw screenshot as BGR numpy array
            health_state: Health detection results
        """
        # Preprocess screenshot for neural network
        processed_screenshot = preprocess_frame(screenshot, target_size=(self.img_height, self.img_width))
        
        # Add to screenshot stack
        self.screenshot_frames.append(processed_screenshot)
        
        # Add to health history
        self.health_tracker.add_health_state(health_state)
        
        # Update frame count
        self.frame_count += 1
        
        # Debug logging for all frames during collection
        print(f"   Frame {self.frame_count}: Screenshot {processed_screenshot.shape}, "
              f"Health P1={health_state.p1_health:.1f}% P2={health_state.p2_health:.1f}%")
    
    def get_current_state(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get current hybrid state (screenshots + health history).
        
        Returns:
            Tuple of (stacked_screenshots, health_history):
            - stacked_screenshots: (frame_stack_size, height, width) as float32 [0,1]
            - health_history: (health_history_length, 4) as float32
        """
        # Get stacked screenshots
        stacked_screenshots = self._get_stacked_screenshots()
        
        # Get health history
        health_history = self.health_tracker.get_health_history()
        
        return stacked_screenshots, health_history
    
    def _get_stacked_screenshots(self) -> np.ndarray:
        """
        Get stacked screenshots as numpy array.
        
        Returns:
            Array of shape (frame_stack_size, height, width) normalized to [0,1]
        """
        if len(self.screenshot_frames) == 0:
            # No frames yet, return zeros
            return np.zeros((self.frame_stack_size, self.img_height, self.img_width), dtype=np.float32)
        
        # Convert deque to list for easier handling
        frames_list = list(self.screenshot_frames)
        
        # Pad if we don't have enough frames yet
        if len(frames_list) < self.frame_stack_size:
            padding_needed = self.frame_stack_size - len(frames_list)
            # Pad with first frame repeated (better than zeros)
            first_frame = frames_list[0] if len(frames_list) > 0 else np.zeros((self.img_height, self.img_width), dtype=np.uint8)
            padding = [first_frame] * padding_needed
            frames_list = padding + frames_list
        
        # Stack frames and normalize to [0, 1]
        stacked = np.stack(frames_list, axis=0).astype(np.float32) / 255.0
        
        return stacked
    
    def is_ready(self) -> bool:
        """
        Check if manager has enough data for meaningful state.
        
        Returns:
            True if we have at least 2 frames (for health deltas)
        """
        return (len(self.screenshot_frames) >= 2 and 
                self.health_tracker.is_ready())
    
    def is_full(self) -> bool:
        """
        Check if manager has full frame stacks.
        
        Returns:
            True if both screenshot and health stacks are full
        """
        return (len(self.screenshot_frames) >= self.frame_stack_size and
                self.health_tracker.is_full())
    
    def get_latest_health(self) -> Optional[Tuple[float, float]]:
        """
        Get most recent health values.
        
        Returns:
            (p1_health, p2_health) or None if no data
        """
        return self.health_tracker.get_latest_health()
    
    def get_latest_deltas(self) -> Optional[Tuple[float, float]]:
        """
        Get most recent health deltas.
        
        Returns:
            (p1_delta, p2_delta) or None if insufficient data
        """
        return self.health_tracker.get_latest_deltas()
    
    def reset(self):
        """Reset the state manager (clear all history)."""
        self.screenshot_frames.clear()
        self.health_tracker.reset()
        self.frame_count = 0
        print(f"🔗 HybridStateManager reset")
    
    def get_state_summary(self) -> dict:
        """Get current state statistics and info."""
        screenshots, health_history = self.get_current_state()
        
        return {
            'frame_count': self.frame_count,
            'screenshot_frames': len(self.screenshot_frames),
            'is_ready': self.is_ready(),
            'is_full': self.is_full(),
            'screenshot_shape': screenshots.shape,
            'health_shape': health_history.shape,
            'latest_health': self.get_latest_health(),
            'latest_deltas': self.get_latest_deltas(),
            'health_stats': self.health_tracker.get_stats()
        }
    
    def print_current_state(self):
        """Print detailed current state information."""
        screenshots, health_history = self.get_current_state()
        
        print(f"\n🔗 Hybrid State Summary:")
        print(f"   Frame count: {self.frame_count}")
        print(f"   Screenshot stack: {screenshots.shape} (min: {screenshots.min():.3f}, max: {screenshots.max():.3f})")
        print(f"   Health history: {health_history.shape}")
        print(f"   Ready: {self.is_ready()}, Full: {self.is_full()}")
        
        # Show latest health info
        latest_health = self.get_latest_health()
        latest_deltas = self.get_latest_deltas()
        
        if latest_health:
            p1_health, p2_health = latest_health
            print(f"   Latest health: P1={p1_health:.1f}% P2={p2_health:.1f}%")
        
        if latest_deltas:
            p1_delta, p2_delta = latest_deltas
            print(f"   Latest deltas: P1={p1_delta:+.1f}% P2={p2_delta:+.1f}%")
        
        # Show health history details
        print(f"\n   Health History Details:")
        self.health_tracker.print_current_state()


class HybridStateTester:
    """Test suite for HybridStateManager."""
    
    def __init__(self):
        self.manager = None
    
    def test_basic_functionality(self):
        """Test basic add frame and get state operations."""
        print("\n🧪 Testing Basic Functionality...")
        
        try:
            # Create manager
            self.manager = HybridStateManager(
                frame_stack_size=4,
                img_size=(84, 84),  # Smaller for testing
                health_history_length=4
            )
            
            # Create dummy frames
            for i in range(6):  # Add more than stack size
                # Create dummy screenshot (random noise)
                screenshot = np.random.randint(0, 255, (100, 150, 3), dtype=np.uint8)
                
                # Create dummy health state
                health_state = HealthState(
                    p1_health=100.0 - i*5,
                    p2_health=100.0 - i*3,
                    p1_pixels=i*10,
                    p2_pixels=i*6
                )
                
                print(f"Adding frame {i+1}...")
                self.manager.add_frame(screenshot, health_state)
                
                # Check readiness
                print(f"   Ready: {self.manager.is_ready()}, Full: {self.manager.is_full()}")
            
            # Get current state
            screenshots, health_history = self.manager.get_current_state()
            
            print(f"\n✅ Final state shapes:")
            print(f"   Screenshots: {screenshots.shape}")
            print(f"   Health history: {health_history.shape}")
            
            # Verify shapes
            expected_screenshot_shape = (4, 84, 84)
            expected_health_shape = (4, 4)
            
            assert screenshots.shape == expected_screenshot_shape, f"Screenshot shape mismatch: {screenshots.shape}"
            assert health_history.shape == expected_health_shape, f"Health shape mismatch: {health_history.shape}"
            
            # Verify value ranges
            assert 0.0 <= screenshots.min() <= screenshots.max() <= 1.0, f"Screenshot values out of range [0,1]"
            
            print("✅ Basic functionality test passed!")
            return True
            
        except Exception as e:
            print(f"❌ Basic functionality test failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def test_insufficient_data(self):
        """Test behavior with insufficient data."""
        print("\n🧪 Testing Insufficient Data Handling...")
        
        try:
            # Create fresh manager
            manager = HybridStateManager(frame_stack_size=4, img_size=(84, 84), health_history_length=4)
            
            # Test with no data
            screenshots, health_history = manager.get_current_state()
            print(f"✅ No data - Screenshots: {screenshots.shape}, Health: {health_history.shape}")
            
            # Test with one frame
            screenshot = np.random.randint(0, 255, (100, 150, 3), dtype=np.uint8)
            health_state = HealthState(p1_health=50.0, p2_health=75.0, p1_pixels=0, p2_pixels=0)
            manager.add_frame(screenshot, health_state)
            
            screenshots, health_history = manager.get_current_state()
            print(f"✅ One frame - Screenshots: {screenshots.shape}, Health: {health_history.shape}")
            print(f"   Ready: {manager.is_ready()}")
            
            return True
            
        except Exception as e:
            print(f"❌ Insufficient data test failed: {e}")
            return False
    
    def test_state_synchronization(self):
        """Test that screenshot and health data stay synchronized."""
        print("\n🧪 Testing State Synchronization...")
        
        try:
            manager = HybridStateManager(frame_stack_size=3, img_size=(64, 64), health_history_length=3)
            
            # Add frames with known patterns
            for i in range(5):
                # Create screenshot with identifiable pattern (all pixels = i*50)
                screenshot = np.full((80, 80, 3), i*50, dtype=np.uint8)
                
                # Create health state with known values
                health_state = HealthState(
                    p1_health=90.0 - i*10,  # 90, 80, 70, 60, 50
                    p2_health=95.0 - i*5,   # 95, 90, 85, 80, 75
                    p1_pixels=i*20,
                    p2_pixels=i*10
                )
                
                manager.add_frame(screenshot, health_state)
            
            # Check final synchronized state
            screenshots, health_history = manager.get_current_state()
            
            # Verify screenshot synchronization (should have last 3 frames)
            # Last 3 screenshots should have patterns 2, 3, 4 (i*50 = 100, 150, 200)
            expected_patterns = [100, 150, 200]  # Normalized to [0,1]: [100/255, 150/255, 200/255]
            
            for frame_idx, expected_value in enumerate(expected_patterns):
                normalized_expected = expected_value / 255.0
                frame_mean = screenshots[frame_idx].mean()
                
                print(f"   Frame {frame_idx}: expected {normalized_expected:.3f}, got {frame_mean:.3f}")
                assert abs(frame_mean - normalized_expected) < 0.01, f"Frame {frame_idx} synchronization error"
            
            # Verify health synchronization (should have last 3 health readings)
            # Last 3 health states: (70,85), (60,80), (50,75)
            expected_health = [(70.0, 85.0), (60.0, 80.0), (50.0, 75.0)]
            
            for frame_idx, (exp_p1, exp_p2) in enumerate(expected_health):
                actual_p1, actual_p2 = health_history[frame_idx, 0], health_history[frame_idx, 1]
                print(f"   Health {frame_idx}: expected P1={exp_p1} P2={exp_p2}, got P1={actual_p1} P2={actual_p2}")
                
                assert abs(actual_p1 - exp_p1) < 0.1, f"P1 health sync error frame {frame_idx}"
                assert abs(actual_p2 - exp_p2) < 0.1, f"P2 health sync error frame {frame_idx}"
            
            print("✅ State synchronization test passed!")
            return True
            
        except Exception as e:
            print(f"❌ State synchronization test failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def run_all_tests(self):
        """Run complete test suite."""
        print("🧪 HYBRID STATE MANAGER TEST SUITE")
        print("="*50)
        
        tests = [
            ("Basic Functionality", self.test_basic_functionality),
            ("Insufficient Data", self.test_insufficient_data),
            ("State Synchronization", self.test_state_synchronization)
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
            print(f"\n🎯 All tests passed! HybridStateManager ready for DQN integration.")
        else:
            print(f"\n⚠️ Some tests failed. Check the output above for details.")
        
        return all_passed


def main():
    """Main testing function."""
    print("🔗 HYBRID STATE MANAGER")
    print("="*40)
    print("Testing synchronized screenshot + health state management.")
    print("="*40)
    
    # Run tests
    tester = HybridStateTester()
    success = tester.run_all_tests()
    
    if success:
        print("\n✅ HybridStateManager is ready!")
        print("Next step: Create DQN Agent that uses hybrid states.")
    else:
        print("\n❌ Fix the issues above before proceeding.")


if __name__ == "__main__":
    main()