import mss
import time
import numpy as np
import cv2
from collections import deque
import statistics
import sys

# Your BizHawk window position (update these!)
# These should match where you positioned your small BizHawk window
GAME_REGION = {
    "left": -1920,  # Adjust based on actual position
    "top": 76,      # Adjust based on actual position  
    "width": 160,   # Atari native width
    "height": 192   # Atari native height
}

class ContinuousCaptureTest:
    def __init__(self, region):
        self.region = region
        self.frame_times = deque(maxlen=1000)  # Keep last 1000 frame times
        self.numpy_times = deque(maxlen=1000)
        self.preprocess_times = deque(maxlen=1000)
        self.total_times = deque(maxlen=1000)
        self.frame_count = 0
        self.start_time = None
        
    def preprocess_frame(self, img_array):
        """Simulate RL preprocessing: grayscale + resize to 84x84"""
        start = time.perf_counter()
        # Convert BGRA to grayscale
        gray = cv2.cvtColor(img_array, cv2.COLOR_BGRA2GRAY)
        # Resize to 84x84 (standard for Atari RL)
        resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
        preprocess_time = (time.perf_counter() - start) * 1000
        return resized, preprocess_time
    
    def capture_continuous(self, duration_seconds=10, with_preprocessing=False):
        """Capture frames continuously for specified duration."""
        print(f"\nCapturing for {duration_seconds} seconds...")
        print(f"Region: {self.region}")
        print(f"Preprocessing: {'Yes (grayscale + 84x84 resize)' if with_preprocessing else 'No'}")
        print("\nPress Ctrl+C to stop early\n")
        
        self.start_time = time.time()
        end_time = self.start_time + duration_seconds
        
        with mss.mss() as sct:
            # Warm-up
            sct.grab(self.region)
            
            try:
                while time.time() < end_time:
                    loop_start = time.perf_counter()
                    
                    # Capture
                    capture_start = time.perf_counter()
                    img = sct.grab(self.region)
                    capture_time = (time.perf_counter() - capture_start) * 1000
                    
                    # Convert to numpy
                    numpy_start = time.perf_counter()
                    img_array = np.array(img)
                    numpy_time = (time.perf_counter() - numpy_start) * 1000
                    
                    # Optional preprocessing
                    if with_preprocessing:
                        processed, preprocess_time = self.preprocess_frame(img_array)
                        self.preprocess_times.append(preprocess_time)
                    else:
                        preprocess_time = 0
                    
                    # Total time
                    total_time = (time.perf_counter() - loop_start) * 1000
                    
                    # Store timings
                    self.frame_times.append(capture_time)
                    self.numpy_times.append(numpy_time)
                    self.total_times.append(total_time)
                    self.frame_count += 1
                    
                    # Progress update every second
                    if self.frame_count % 60 == 0:
                        elapsed = time.time() - self.start_time
                        actual_fps = self.frame_count / elapsed
                        print(f"Frames: {self.frame_count} | "
                              f"Actual FPS: {actual_fps:.1f} | "
                              f"Capture: {statistics.mean(list(self.frame_times)[-60:]):.1f}ms")
                        
            except KeyboardInterrupt:
                print("\nStopped by user")
        
        self.print_results(with_preprocessing)
    
    def print_results(self, with_preprocessing):
        """Print detailed statistics."""
        elapsed = time.time() - self.start_time
        actual_fps = self.frame_count / elapsed
        
        print("\n" + "="*70)
        print("CONTINUOUS CAPTURE RESULTS")
        print("="*70)
        
        print(f"\nOverall Performance:")
        print(f"  Total frames: {self.frame_count}")
        print(f"  Total time: {elapsed:.1f} seconds")
        print(f"  Actual FPS: {actual_fps:.1f}")
        print(f"  Target FPS: 60")
        print(f"  Performance: {'✓ GOOD' if actual_fps >= 60 else '✗ TOO SLOW'}")
        
        print(f"\nCapture Statistics (ms):")
        print(f"  Average: {statistics.mean(self.frame_times):.2f}")
        print(f"  Min: {min(self.frame_times):.2f}")
        print(f"  Max: {max(self.frame_times):.2f}")
        print(f"  StdDev: {statistics.stdev(self.frame_times):.2f}")
        
        print(f"\nNumPy Conversion (ms):")
        print(f"  Average: {statistics.mean(self.numpy_times):.2f}")
        print(f"  Min: {min(self.numpy_times):.2f}")
        print(f"  Max: {max(self.numpy_times):.2f}")
        
        if with_preprocessing and self.preprocess_times:
            print(f"\nPreprocessing (grayscale + resize to 84x84) (ms):")
            print(f"  Average: {statistics.mean(self.preprocess_times):.2f}")
            print(f"  Min: {min(self.preprocess_times):.2f}")
            print(f"  Max: {max(self.preprocess_times):.2f}")
        
        print(f"\nTotal Pipeline (ms):")
        print(f"  Average: {statistics.mean(self.total_times):.2f}")
        print(f"  Min: {min(self.total_times):.2f}")
        print(f"  Max: {max(self.total_times):.2f}")
        print(f"  Budget per frame (60 FPS): 16.67ms")
        print(f"  Remaining for RL model: {16.67 - statistics.mean(self.total_times):.2f}ms")
        
        # Frame drop analysis
        slow_frames = sum(1 for t in self.total_times if t > 16.67)
        print(f"\nFrame Analysis:")
        print(f"  Frames over 16.67ms: {slow_frames} ({100*slow_frames/len(self.total_times):.1f}%)")
        
        # Distribution
        print(f"\nTiming Distribution:")
        percentiles = [50, 90, 95, 99]
        for p in percentiles:
            value = sorted(self.total_times)[int(len(self.total_times) * p / 100)]
            print(f"  {p}th percentile: {value:.2f}ms")

def main():
    print("Continuous Frame Capture Benchmark")
    print("="*70)
    
    # Test capture location first
    print("\nTesting capture location...")
    with mss.mss() as sct:
        img = sct.grab(GAME_REGION)
        mss.tools.to_png(img.rgb, img.size, output="continuous_test.png")
        print("Saved test capture to continuous_test.png")
        print("CHECK THIS IMAGE! If it's not capturing your game, adjust GAME_REGION")
    
    input("\nPress Enter to start continuous capture test...")
    
    # Test 1: Raw capture speed
    test1 = ContinuousCaptureTest(GAME_REGION)
    test1.capture_continuous(duration_seconds=5, with_preprocessing=False)
    
    # Test 2: With preprocessing
    print("\n" + "="*70)
    input("Press Enter to test with preprocessing...")
    test2 = ContinuousCaptureTest(GAME_REGION)
    test2.capture_continuous(duration_seconds=5, with_preprocessing=True)
    
    # Summary
    print("\n" + "="*70)
    print("RECOMMENDATIONS FOR RL:")
    print("="*70)
    print("1. If FPS < 60: Make BizHawk window smaller")
    print("2. If FPS >> 60: You have headroom for larger windows")
    print("3. Preprocessing adds ~1-2ms (acceptable)")
    print("4. Budget ~5-10ms for your neural network")
    print("5. Consider frame skipping (process every 4th frame)")

if __name__ == "__main__":
    main()