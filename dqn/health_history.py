#!/usr/bin/env python3
"""
Health History Tracker

Tracks player health percentages and deltas over multiple frames.
Designed to work with existing HealthDetector for DQN input.

Single Responsibility:
- Store health history over configurable N frames
- Calculate frame-to-frame health deltas  
- Provide clean numpy arrays for neural network input
"""

import numpy as np
from collections import deque
from typing import Optional, List, Tuple
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from detection.health_detector import HealthState


class HealthHistoryTracker:
    """
    Tracks health percentages and deltas over multiple frames.
    
    Provides temporal health information for DQN training:
    - Current and historical health percentages
    - Frame-to-frame health changes (deltas)
    - Configurable history length
    """
    
    def __init__(self, history_length: int = 4):
        """
        Initialize health history tracker.
        
        Args:
            history_length: Number of frames to track (default 4)
        """
        self.history_length = history_length
        
        # Store health data as [p1_health, p2_health] for each frame
        self.health_history = deque(maxlen=history_length)
        
        # Track frame count for debugging
        self.frame_count = 0
        
        print(f"🩺 HealthHistoryTracker initialized:")
        print(f"   History length: {history_length} frames")
        print(f"   Output shape: ({history_length}, 4) [p1_health, p2_health, p1_delta, p2_delta]")
    
    def add_health_state(self, health_state: HealthState):
        """
        Add a new health reading to the history.
        
        Args:
            health_state: HealthState from HealthDetector
        """
        if health_state is None:
            print("⚠️ Warning: Received None health_state, skipping")
            return
        
        # Extract health percentages
        current_health = [health_state.p1_health, health_state.p2_health]
        
        # Add to history
        self.health_history.append(current_health)
        self.frame_count += 1
        
        # Debug logging for first few frames
        if self.frame_count <= 3:
            print(f"   Frame {self.frame_count}: P1={health_state.p1_health:.1f}% P2={health_state.p2_health:.1f}%")
    
    def get_health_history(self) -> np.ndarray:
        """
        Get health history with deltas as numpy array.
        
        Returns:
            Array of shape (history_length, 4) where each row is:
            [p1_health, p2_health, p1_delta, p2_delta]
            
            If insufficient frames, pads with zeros.
        """
        # Convert to numpy array
        if len(self.health_history) == 0:
            # No data yet, return zeros
            return np.zeros((self.history_length, 4), dtype=np.float32)
        
        # Get all health data as array
        health_data = np.array(list(self.health_history), dtype=np.float32)
        
        # Pad if we don't have enough frames yet
        if len(health_data) < self.history_length:
            padding_needed = self.history_length - len(health_data)
            # Pad with first frame repeated (better than zeros)
            first_frame = health_data[0:1] if len(health_data) > 0 else np.zeros((1, 2))
            padding = np.repeat(first_frame, padding_needed, axis=0)
            health_data = np.vstack([padding, health_data])
        
        # Calculate deltas with 0% filtering
        deltas = np.zeros((self.history_length, 2), dtype=np.float32)
        
        for i in range(1, self.history_length):
            # P1 delta calculation
            current_p1 = health_data[i, 0]
            if current_p1 == 0.0:
                # Don't trust 0% readings
                deltas[i, 0] = 0.0
            else:
                # Find last valid (non-zero) P1 health reading
                last_valid_p1 = self._find_last_valid_health(health_data[:i+1, 0])
                deltas[i, 0] = current_p1 - last_valid_p1 if last_valid_p1 is not None else 0.0
            
            # P2 delta calculation
            current_p2 = health_data[i, 1]
            if current_p2 == 0.0:
                # Don't trust 0% readings
                deltas[i, 1] = 0.0
            else:
                # Find last valid (non-zero) P2 health reading
                last_valid_p2 = self._find_last_valid_health(health_data[:i+1, 1])
                deltas[i, 1] = current_p2 - last_valid_p2 if last_valid_p2 is not None else 0.0
        
        # For first frame, delta is 0 (no previous frame)
        deltas[0] = [0.0, 0.0]
        
        # Combine health percentages and deltas
        # Shape: (history_length, 4) = [p1_health, p2_health, p1_delta, p2_delta]
        result = np.hstack([health_data, deltas])
        
        return result
    
    def _find_last_valid_health(self, health_sequence: np.ndarray) -> Optional[float]:
        """
        Find the last valid (non-zero) health reading in a sequence.
        
        Args:
            health_sequence: Array of health values to search backwards
            
        Returns:
            Last valid health value, or None if no valid readings found
        """
        # Search backwards from most recent
        for i in range(len(health_sequence) - 2, -1, -1):  # Skip current frame (last element)
            if health_sequence[i] > 0.0:
                return health_sequence[i]
        
        # No valid health found, return None
        return None
    
    def get_latest_health(self) -> Optional[Tuple[float, float]]:
        """
        Get the most recent health values.
        
        Returns:
            (p1_health, p2_health) or None if no data
        """
        if len(self.health_history) == 0:
            return None
        
        latest = self.health_history[-1]
        return (latest[0], latest[1])
    
    def get_latest_deltas(self) -> Optional[Tuple[float, float]]:
        """
        Get the most recent health deltas.
        
        Returns:
            (p1_delta, p2_delta) or None if insufficient data
        """
        if len(self.health_history) < 2:
            return None
        
        current = self.health_history[-1]
        previous = self.health_history[-2]
        
        p1_delta = current[0] - previous[0]
        p2_delta = current[1] - previous[1]
        
        return (p1_delta, p2_delta)
    
    def is_ready(self) -> bool:
        """
        Check if tracker has enough data for meaningful deltas.
        
        Returns:
            True if we have at least 2 frames (for delta calculation)
        """
        return len(self.health_history) >= 2
    
    def is_full(self) -> bool:
        """
        Check if tracker has full history.
        
        Returns:
            True if we have history_length frames
        """
        return len(self.health_history) >= self.history_length
    
    def reset(self):
        """Reset the tracker (clear all history)."""
        self.health_history.clear()
        self.frame_count = 0
        print(f"🩺 HealthHistoryTracker reset")
    
    def get_stats(self) -> dict:
        """Get tracker statistics."""
        if len(self.health_history) == 0:
            return {
                'frame_count': self.frame_count,
                'history_size': 0,
                'is_ready': False,
                'is_full': False
            }
        
        latest_health = self.get_latest_health()
        latest_deltas = self.get_latest_deltas()
        
        return {
            'frame_count': self.frame_count,
            'history_size': len(self.health_history),
            'is_ready': self.is_ready(),
            'is_full': self.is_full(),
            'latest_health': latest_health,
            'latest_deltas': latest_deltas,
            'history_length': self.history_length
        }
    
    def print_current_state(self):
        """Print current health history in readable format."""
        history_array = self.get_health_history()
        
        print(f"\n🩺 Health History (last {self.history_length} frames):")
        print("   Frame | P1 Health | P2 Health | P1 Delta | P2 Delta")
        print("   ------|-----------|-----------|----------|----------")
        
        for i, frame_data in enumerate(history_array):
            p1_health, p2_health, p1_delta, p2_delta = frame_data
            frame_num = max(0, self.frame_count - self.history_length + i + 1)
            
            print(f"   {frame_num:5d} | {p1_health:8.1f}% | {p2_health:8.1f}% | "
                  f"{p1_delta:+7.1f}% | {p2_delta:+7.1f}%")


class HealthHistoryTester:
    """Test suite for HealthHistoryTracker."""
    
    def __init__(self):
        self.tracker = None
    
    def test_basic_functionality(self):
        """Test basic add and get operations."""
        print("\n🧪 Testing Basic Functionality...")
        
        try:
            # Create tracker
            self.tracker = HealthHistoryTracker(history_length=4)
            
            # Create dummy health states
            health_states = [
                HealthState(p1_health=100.0, p2_health=100.0, p1_pixels=0, p2_pixels=0),
                HealthState(p1_health=95.0, p2_health=98.0, p1_pixels=20, p2_pixels=8),
                HealthState(p1_health=87.0, p2_health=95.0, p1_pixels=52, p2_pixels=20),
                HealthState(p1_health=82.0, p2_health=90.0, p1_pixels=72, p2_pixels=40),
            ]
            
            # Add health states
            for i, health_state in enumerate(health_states):
                print(f"Adding frame {i+1}...")
                self.tracker.add_health_state(health_state)
                
                # Check readiness
                print(f"   Ready: {self.tracker.is_ready()}, Full: {self.tracker.is_full()}")
                
                if i >= 1:  # Can get deltas after 2 frames
                    latest_deltas = self.tracker.get_latest_deltas()
                    print(f"   Latest deltas: {latest_deltas}")
            
            # Get full history
            history = self.tracker.get_health_history()
            print(f"\n✅ Full history shape: {history.shape}")
            print(f"Expected shape: ({self.tracker.history_length}, 4)")
            
            # Print readable state
            self.tracker.print_current_state()
            
            return True
            
        except Exception as e:
            print(f"❌ Basic functionality test failed: {e}")
            return False
    
    def test_insufficient_data(self):
        """Test behavior with insufficient data."""
        print("\n🧪 Testing Insufficient Data Handling...")
        
        try:
            # Create fresh tracker
            tracker = HealthHistoryTracker(history_length=4)
            
            # Test with no data
            history = tracker.get_health_history()
            print(f"✅ No data - history shape: {history.shape}")
            
            # Test with one frame
            health_state = HealthState(p1_health=50.0, p2_health=75.0, p1_pixels=0, p2_pixels=0)
            tracker.add_health_state(health_state)
            
            history = tracker.get_health_history()
            print(f"✅ One frame - history shape: {history.shape}")
            print(f"   Ready: {tracker.is_ready()}")
            
            return True
            
        except Exception as e:
            print(f"❌ Insufficient data test failed: {e}")
            return False
    
    def test_different_history_lengths(self):
        """Test with different history lengths."""
        print("\n🧪 Testing Different History Lengths...")
        
        try:
            lengths_to_test = [2, 6, 8]
            
            for length in lengths_to_test:
                print(f"\nTesting history length: {length}")
                tracker = HealthHistoryTracker(history_length=length)
                
                # Add some data
                for i in range(length + 2):  # Add more than history length
                    health_state = HealthState(
                        p1_health=100.0 - i*5,
                        p2_health=100.0 - i*3,
                        p1_pixels=i*10,
                        p2_pixels=i*6
                    )
                    tracker.add_health_state(health_state)
                
                history = tracker.get_health_history()
                print(f"   ✅ Length {length}: {history.shape}")
                assert history.shape == (length, 4), f"Wrong shape: {history.shape}"
            
            return True
            
        except Exception as e:
            print(f"❌ Different lengths test failed: {e}")
            return False
    
    def test_zero_filtering(self):
        """Test filtering of 0% health readings."""
        print("\n🧪 Testing 0% Health Filtering...")
        
        try:
            tracker = HealthHistoryTracker(history_length=4)
            
            # Test sequence with 0% readings: [100%, 95%, 0%, 90%]
            health_states = [
                HealthState(p1_health=100.0, p2_health=100.0, p1_pixels=0, p2_pixels=0),
                HealthState(p1_health=95.0, p2_health=98.0, p1_pixels=20, p2_pixels=8),
                HealthState(p1_health=0.0, p2_health=0.0, p1_pixels=400, p2_pixels=400),  # False reading
                HealthState(p1_health=90.0, p2_health=95.0, p1_pixels=40, p2_pixels=20),
            ]
            
            # Add health states
            for health_state in health_states:
                tracker.add_health_state(health_state)
            
            # Get history with deltas
            history = tracker.get_health_history()
            
            print("Health sequence with 0% filtering:")
            print("Frame | P1 Health | P2 Health | P1 Delta | P2 Delta | Expected Delta")
            print("------|-----------|-----------|----------|----------|---------------")
            
            expected_p1_deltas = [0.0, -5.0, 0.0, -5.0]  # 0% reading should give 0 delta, 90% should be 90-95=-5
            expected_p2_deltas = [0.0, -2.0, 0.0, -3.0]  # 0% reading should give 0 delta, 95% should be 95-98=-3
            
            for i, frame_data in enumerate(history):
                p1_health, p2_health, p1_delta, p2_delta = frame_data
                exp_p1, exp_p2 = expected_p1_deltas[i], expected_p2_deltas[i]
                
                print(f"{i+1:5d} | {p1_health:8.1f}% | {p2_health:8.1f}% | "
                      f"{p1_delta:+7.1f}% | {p2_delta:+7.1f}% | P1:{exp_p1:+.1f} P2:{exp_p2:+.1f}")
                
                # Verify deltas are as expected
                assert abs(p1_delta - exp_p1) < 0.1, f"P1 delta mismatch: {p1_delta} vs {exp_p1}"
                assert abs(p2_delta - exp_p2) < 0.1, f"P2 delta mismatch: {p2_delta} vs {exp_p2}"
            
            print("✅ 0% filtering working correctly!")
            
            # Test multiple zeros in a row
            print("\nTesting multiple 0% readings in a row...")
            tracker.reset()
            
            multiple_zero_states = [
                HealthState(p1_health=80.0, p2_health=90.0, p1_pixels=80, p2_pixels=40),
                HealthState(p1_health=0.0, p2_health=0.0, p1_pixels=400, p2_pixels=400),  # False reading 1
                HealthState(p1_health=0.0, p2_health=0.0, p1_pixels=400, p2_pixels=400),  # False reading 2  
                HealthState(p1_health=75.0, p2_health=85.0, p1_pixels=100, p2_pixels=60),
            ]
            
            for health_state in multiple_zero_states:
                tracker.add_health_state(health_state)
            
            history = tracker.get_health_history()
            
            # Last frame should have delta of 75-80=-5 and 85-90=-5 (ignoring the 0% readings)
            last_frame = history[-1]
            p1_delta, p2_delta = last_frame[2], last_frame[3]
            
            print(f"Multiple zeros test - Final deltas: P1={p1_delta:+.1f}%, P2={p2_delta:+.1f}%")
            print("Expected: P1=-5.0%, P2=-5.0% (ignoring 0% readings)")
            
            assert abs(p1_delta - (-5.0)) < 0.1, f"Multiple zeros P1 delta wrong: {p1_delta}"
            assert abs(p2_delta - (-5.0)) < 0.1, f"Multiple zeros P2 delta wrong: {p2_delta}"
            
            print("✅ Multiple 0% filtering working correctly!")
            
            return True
            
        except Exception as e:
            print(f"❌ Zero filtering test failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def run_all_tests(self):
        """Run complete test suite."""
        print("🧪 HEALTH HISTORY TRACKER TEST SUITE")
        print("="*50)
        
        tests = [
            ("Basic Functionality", self.test_basic_functionality),
            ("Insufficient Data", self.test_insufficient_data),
            ("Different Lengths", self.test_different_history_lengths),
            ("Zero Filtering", self.test_zero_filtering)
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
            print(f"\n🎯 All tests passed! HealthHistoryTracker ready for DQN integration.")
        else:
            print(f"\n⚠️ Some tests failed. Check the output above for details.")
        
        return all_passed


def main():
    """Main testing function."""
    print("🩺 HEALTH HISTORY TRACKER")
    print("="*40)
    print("Testing health tracking component for DQN hybrid input.")
    print("="*40)
    
    # Run tests
    tester = HealthHistoryTester()
    success = tester.run_all_tests()
    
    if success:
        print("\n✅ HealthHistoryTracker is ready!")
        print("Next step: Create DQN environment that uses this tracker.")
    else:
        print("\n❌ Fix the issues above before proceeding.")


if __name__ == "__main__":
    main()