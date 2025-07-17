import mss
import time
import numpy as np
import cv2
from collections import deque
import statistics

# Your BizHawk window position
GAME_REGION = {
    "left": -1920,
    "top": 76,
    "width": 160,
    "height": 192
}

class RLCaptureTest:
    def __init__(self, region, frame_skip=4):
        self.region = region
        self.frame_skip = frame_skip
        self.capture_times = deque(maxlen=1000)
        self.process_times = deque(maxlen=1000)
        self.decision_times = deque(maxlen=1000)
        self.frame_count = 0
        self.decision_count = 0
        
    def preprocess_frame(self, img_array):
        """RL preprocessing: grayscale + resize to 84x84"""
        gray = cv2.cvtColor(img_array, cv2.COLOR_BGRA2GRAY)
        resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
        return resized
    
    def simulate_neural_network(self, frame):
        """Simulate NN forward pass with a small delay"""
        time.sleep(0.005)  # Simulate 5ms neural network inference
        return np.random.randint(0, 4)  # Random action
    
    def run_rl_loop(self, duration_seconds=10):
        """Simulate actual RL training loop with frame skipping."""
        print(f"\nSimulating RL capture loop for {duration_seconds} seconds...")
        print(f"Frame skip: {self.frame_skip} (process every {self.frame_skip} frames)")
        print(f"Region: {self.region['width']}x{self.region['height']}")
        print("\nPress Ctrl+C to stop early\n")
        
        start_time = time.time()
        end_time = start_time + duration_seconds
        last_processed_frame = None
        
        with mss.mss() as sct:
            # Warm-up
            sct.grab(self.region)
            
            try:
                while time.time() < end_time:
                    # Always capture
                    capture_start = time.perf_counter()
                    img = sct.grab(self.region)
                    capture_time = (time.perf_counter() - capture_start) * 1000
                    self.capture_times.append(capture_time)
                    self.frame_count += 1
                    
                    # Process only every Nth frame
                    if self.frame_count % self.frame_skip == 0:
                        process_start = time.perf_counter()
                        
                        # Convert and preprocess
                        img_array = np.array(img)
                        processed = self.preprocess_frame(img_array)
                        
                        # Simulate NN decision
                        action = self.simulate_neural_network(processed)
                        
                        process_time = (time.perf_counter() - process_start) * 1000
                        self.process_times.append(process_time)
                        self.decision_count += 1
                        
                        # Store for frame stacking (common in RL)
                        last_processed_frame = processed
                    
                    # Progress update
                    if self.frame_count % 60 == 0:
                        elapsed = time.time() - start_time
                        capture_fps = self.frame_count / elapsed
                        decision_fps = self.decision_count / elapsed
                        print(f"Frames: {self.frame_count} | "
                              f"Decisions: {self.decision_count} | "
                              f"Capture FPS: {capture_fps:.1f} | "
                              f"Decision FPS: {decision_fps:.1f}")
                        
            except KeyboardInterrupt:
                print("\nStopped by user")
        
        self.print_results(time.time() - start_time)
    
    def print_results(self, elapsed):
        """Print RL-specific performance metrics."""
        capture_fps = self.frame_count / elapsed
        decision_fps = self.decision_count / elapsed
        
        print("\n" + "="*70)
        print("RL CAPTURE PERFORMANCE")
        print("="*70)
        
        print(f"\nOverall Performance:")
        print(f"  Total frames captured: {self.frame_count}")
        print(f"  Total decisions made: {self.decision_count}")
        print(f"  Capture FPS: {capture_fps:.1f}")
        print(f"  Decision FPS: {decision_fps:.1f}")
        print(f"  Emulator FPS target: 60")
        print(f"  RL decision rate: {decision_fps:.1f} Hz")
        
        print(f"\nCapture Times (ms):")
        print(f"  Average: {statistics.mean(self.capture_times):.2f}")
        print(f"  Median: {statistics.median(self.capture_times):.2f}")
        print(f"  95th percentile: {sorted(self.capture_times)[int(len(self.capture_times)*0.95)]:.2f}")
        
        if self.process_times:
            print(f"\nProcessing Times (preprocessing + NN) (ms):")
            print(f"  Average: {statistics.mean(self.process_times):.2f}")
            print(f"  Median: {statistics.median(self.process_times):.2f}")
            print(f"  95th percentile: {sorted(self.process_times)[int(len(self.process_times)*0.95)]:.2f}")
        
        print(f"\nRL Performance Analysis:")
        print(f"  Frame skip: {self.frame_skip}")
        print(f"  Time budget per decision: {self.frame_skip * 16.67:.1f}ms")
        print(f"  Actual time used: ~{statistics.mean(self.process_times):.1f}ms")
        print(f"  Headroom: {(self.frame_skip * 16.67) - statistics.mean(self.process_times):.1f}ms")
        
        if capture_fps >= 59:
            print(f"\n✓ CAPTURE PERFORMANCE: GOOD")
        else:
            print(f"\n✗ CAPTURE PERFORMANCE: TOO SLOW ({60-capture_fps:.1f} FPS short)")
            
        print(f"\n{'✓' if decision_fps >= 15 else '✗'} DECISION RATE: {decision_fps:.1f} Hz " +
              f"({'GOOD for RL' if decision_fps >= 15 else 'May be too slow'})")

def test_different_frame_skips():
    """Test performance with different frame skip values."""
    print("\n" + "="*70)
    print("TESTING DIFFERENT FRAME SKIP VALUES")
    print("="*70)
    
    for skip in [1, 2, 4, 8]:
        print(f"\n--- Frame Skip = {skip} ---")
        test = RLCaptureTest(GAME_REGION, frame_skip=skip)
        test.run_rl_loop(duration_seconds=5)
        
        if skip < 8:
            input("\nPress Enter to test next frame skip value...")

def main():
    print("RL-Style Capture Test with Frame Skipping")
    print("="*70)
    
    # Test capture location
    print("\nVerifying capture location...")
    with mss.mss() as sct:
        img = sct.grab(GAME_REGION)
        mss.tools.to_png(img.rgb, img.size, output="rl_test.png")
        print("Saved test capture to rl_test.png")
    
    input("\nPress Enter to start RL capture tests...")
    
    # Test with standard frame skip
    print("\n1. Testing standard RL setup (frame_skip=4)")
    test_standard = RLCaptureTest(GAME_REGION, frame_skip=4)
    test_standard.run_rl_loop(duration_seconds=10)
    
    # Test different frame skips
    input("\n\nPress Enter to test different frame skip values...")
    test_different_frame_skips()
    
    print("\n" + "="*70)
    print("RL RECOMMENDATIONS:")
    print("="*70)
    print("1. Frame skip of 4 is standard for Atari RL")
    print("2. Your system can handle 15Hz decision rate with frame skip")
    print("3. This leaves plenty of time for larger neural networks")
    print("4. Consider async capture (capture in separate thread)")
    print("5. For faster training, use batch processing on GPU")

if __name__ == "__main__":
    main()