#!/usr/bin/env python3
"""
Screenshot Testing Script for DQN

Tests the screenshot capture pipeline:
1. Raw screenshot capture from BizHawk
2. Preprocessing for DQN training
3. Frame stacking simulation
4. Visual verification of the pipeline
"""

import sys
import os
import time
import cv2
import numpy as np
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from detection.window_capture import WindowCapture
from dqn.replay_buffer import preprocess_frame, ReplayBuffer


class ScreenshotTester:
    """Test screenshot capture and preprocessing for DQN"""
    
    def __init__(self, window_title: str = "Bloody Roar II (USA) [PlayStation] - BizHawk"):
        self.window_title = window_title
        self.capture = None
        self.output_dir = "screenshot_tests"
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        print("🖼️  Screenshot Tester Initialized")
        print(f"   Target window: {window_title}")
        print(f"   Output dir: {self.output_dir}")
    
    def test_basic_capture(self):
        """Test basic screenshot capture"""
        print("\n📸 Testing Basic Screenshot Capture...")
        
        try:
            self.capture = WindowCapture(self.window_title)
            frame = self.capture.capture()
            
            if frame is not None:
                print(f"✅ Screenshot captured successfully!")
                print(f"   Shape: {frame.shape}")
                print(f"   Data type: {frame.dtype}")
                print(f"   Size: {frame.shape[0]}×{frame.shape[1]} pixels")
                print(f"   Memory: ~{frame.nbytes / (1024*1024):.1f} MB")
                
                # Save raw screenshot
                timestamp = datetime.now().strftime("%H%M%S")
                filename = os.path.join(self.output_dir, f"raw_screenshot_{timestamp}.png")
                cv2.imwrite(filename, frame)
                print(f"   Saved: {filename}")
                
                return frame
            else:
                print(f"❌ Screenshot capture failed!")
                print(f"   Check if BizHawk window '{self.window_title}' is open")
                return None
                
        except Exception as e:
            print(f"❌ Error during capture: {e}")
            return None
    
    def test_preprocessing(self, frame: np.ndarray):
        """Test frame preprocessing for DQN"""
        print("\n🔧 Testing Frame Preprocessing...")
        
        if frame is None:
            print("❌ No frame to preprocess")
            return None
        
        try:
            # Test different sizes
            sizes_to_test = [(84, 84), (168, 168)]
            processed_frames = {}
            
            for size in sizes_to_test:
                processed = preprocess_frame(frame, target_size=size)
                processed_frames[size] = processed
                
                print(f"✅ Processed to {size}:")
                print(f"   Shape: {processed.shape}")
                print(f"   Data type: {processed.dtype}")
                print(f"   Value range: {processed.min()}-{processed.max()}")
                print(f"   Memory: ~{processed.nbytes / 1024:.1f} KB")
                
                # Save processed frame
                timestamp = datetime.now().strftime("%H%M%S")
                filename = os.path.join(self.output_dir, f"processed_{size[0]}x{size[1]}_{timestamp}.png")
                cv2.imwrite(filename, processed)
                print(f"   Saved: {filename}")
            
            return processed_frames
            
        except Exception as e:
            print(f"❌ Error during preprocessing: {e}")
            return None
    
    def test_frame_stacking(self, num_frames: int = 5):
        """Test frame stacking by capturing multiple consecutive frames"""
        print(f"\n📚 Testing Frame Stacking ({num_frames} frames)...")
        
        if self.capture is None:
            print("❌ No capture object available")
            return
        
        try:
            frames = []
            frame_times = []
            
            print(f"Capturing {num_frames} frames (1 per second)...")
            
            for i in range(num_frames):
                print(f"  📸 Capturing frame {i+1}/{num_frames}...")
                
                start_time = time.time()
                frame = self.capture.capture()
                
                if frame is not None:
                    # Preprocess frame
                    processed = preprocess_frame(frame, target_size=(168, 168))
                    frames.append(processed)
                    frame_times.append(time.time() - start_time)
                    
                    print(f"     ✅ Frame {i+1} captured in {frame_times[-1]:.3f}s")
                else:
                    print(f"     ❌ Frame {i+1} capture failed")
                
                # Wait 1 second between frames (simulate 1 FPS)
                if i < num_frames - 1:
                    time.sleep(1.0)
            
            if len(frames) >= 4:
                print(f"\n✅ Frame stacking test:")
                print(f"   Captured {len(frames)} frames")
                print(f"   Average capture time: {np.mean(frame_times):.3f}s")
                print(f"   Frame shape: {frames[0].shape}")
                
                # Simulate frame stacking (like in replay buffer)
                stacked = np.stack(frames[-4:], axis=0)  # Last 4 frames
                print(f"   Stacked shape: {stacked.shape} (frame_stack, height, width)")
                
                # Save comparison image
                self._save_frame_comparison(frames)
                
                return frames
            else:
                print(f"❌ Not enough frames captured for stacking test")
                return None
                
        except Exception as e:
            print(f"❌ Error during frame stacking test: {e}")
            return None
    
    def test_replay_buffer_integration(self, frames: list):
        """Test integration with replay buffer"""
        print(f"\n🧠 Testing ReplayBuffer Integration...")
        
        if not frames or len(frames) < 4:
            print("❌ Not enough frames for replay buffer test")
            return
        
        try:
            # Create small replay buffer
            buffer = ReplayBuffer(capacity=100, frame_stack=4, img_size=(168, 168))
            
            # Add dummy transitions using real frames
            for i in range(len(frames) - 1):
                state = frames[i]
                action = np.random.randint(0, 10)  # Random action
                reward = np.random.uniform(-1, 1)  # Random reward
                next_state = frames[i + 1]
                done = False  # Don't end episodes for this test
                
                buffer.add(state, action, reward, next_state, done)
            
            print(f"✅ Added {len(frames)-1} transitions to buffer")
            print(f"   Buffer stats: {buffer.get_stats()}")
            
            # Test sampling
            if buffer.can_sample(2):
                states, actions, rewards, next_states, dones = buffer.sample(2)
                print(f"✅ Sampled batch from buffer:")
                print(f"   States shape: {states.shape}")
                print(f"   Value range: {states.min():.3f}-{states.max():.3f}")
                print(f"   Actions: {actions}")
                print(f"   Rewards: {rewards}")
                
                return True
            else:
                print("❌ Cannot sample from buffer yet")
                return False
                
        except Exception as e:
            print(f"❌ Error during replay buffer test: {e}")
            return False
    
    def _save_frame_comparison(self, frames: list):
        """Save a comparison image showing multiple frames"""
        if len(frames) < 2:
            return
        
        # Take first 4 frames for comparison
        compare_frames = frames[:min(4, len(frames))]
        
        # Create comparison image
        h, w = compare_frames[0].shape
        comparison = np.zeros((h, w * len(compare_frames)), dtype=np.uint8)
        
        for i, frame in enumerate(compare_frames):
            comparison[:, i*w:(i+1)*w] = frame
        
        # Save comparison
        timestamp = datetime.now().strftime("%H%M%S")
        filename = os.path.join(self.output_dir, f"frame_sequence_{timestamp}.png")
        cv2.imwrite(filename, comparison)
        print(f"   Frame sequence saved: {filename}")
    
    def run_full_test(self):
        """Run complete screenshot testing pipeline"""
        print("🚀 Running Full Screenshot Test Pipeline")
        print("="*60)
        
        # Test 1: Basic capture
        frame = self.test_basic_capture()
        if frame is None:
            print("\n❌ Basic capture failed - stopping tests")
            return False
        
        # Test 2: Preprocessing
        processed_frames = self.test_preprocessing(frame)
        if processed_frames is None:
            print("\n❌ Preprocessing failed - stopping tests")
            return False
        
        # Test 3: Frame stacking
        frames = self.test_frame_stacking(num_frames=5)
        if frames is None:
            print("\n❌ Frame stacking failed - continuing anyway")
            frames = [processed_frames[(168, 168)]]  # Use single processed frame
        
        # Test 4: Replay buffer integration
        buffer_success = self.test_replay_buffer_integration(frames)
        
        # Summary
        print("\n🎉 Screenshot Test Summary:")
        print("="*60)
        print(f"✅ Basic capture: {'PASS' if frame is not None else 'FAIL'}")
        print(f"✅ Preprocessing: {'PASS' if processed_frames is not None else 'FAIL'}")
        print(f"✅ Frame stacking: {'PASS' if frames is not None else 'FAIL'}")
        print(f"✅ Buffer integration: {'PASS' if buffer_success else 'FAIL'}")
        print(f"📁 Output saved to: {self.output_dir}/")
        
        if all([frame is not None, processed_frames is not None, buffer_success]):
            print("\n🎯 All tests passed! Ready for DQN training with screenshots.")
            return True
        else:
            print("\n⚠️  Some tests failed. Check the output above for details.")
            return False


def main():
    """Main testing function"""
    print("🖼️  SCREENSHOT TESTING FOR DQN")
    print("="*50)
    print("This script tests the screenshot capture pipeline for DQN training.")
    print("Make sure BizHawk is running with Bloody Roar 2 loaded!")
    print("="*50)
    
    # Check if user wants to continue
    try:
        input("Press Enter to start testing (or Ctrl+C to cancel)...")
    except KeyboardInterrupt:
        print("\n❌ Test cancelled by user")
        return
    
    # Run tests
    tester = ScreenshotTester()
    success = tester.run_full_test()
    
    if success:
        print("\n✅ Screenshot testing complete! You're ready for vision-based DQN.")
    else:
        print("\n❌ Some issues found. Fix them before proceeding with DQN training.")


if __name__ == "__main__":
    main()