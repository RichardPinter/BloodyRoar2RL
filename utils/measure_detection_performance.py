#!/usr/bin/env python3
"""
Performance Measurement for Character Detection Pipeline

Measures the time taken by each component in the character detection pipeline:
- Screen capture from emulator
- YOLO inference 
- Post-processing
- Total end-to-end time

Helps identify bottlenecks and calculate actual FPS.
"""

import time
import statistics
from typing import List, Dict
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from detection.window_capture import WindowCapture
from detection.fighter_detector import FighterDetector

class DetectionPerformanceProfiler:
    """Profile performance of character detection pipeline"""
    
    def __init__(self, window_title: str = "Bloody Roar II (USA) [PlayStation] - BizHawk"):
        self.window_title = window_title
        self.capture = WindowCapture(window_title)
        self.detector = FighterDetector()
        
        # Storage for timing data
        self.capture_times: List[float] = []
        self.yolo_times: List[float] = []
        self.postprocess_times: List[float] = []
        self.total_times: List[float] = []
        
    def measure_screen_capture(self) -> tuple:
        """Measure screen capture performance"""
        start_time = time.perf_counter()
        frame = self.capture.capture()
        end_time = time.perf_counter()
        
        capture_duration = (end_time - start_time) * 1000  # Convert to ms
        return frame, capture_duration
    
    def measure_yolo_inference(self, frame) -> tuple:
        """Measure YOLO inference performance with sub-timing"""
        
        # Time the full YOLO detection
        start_time = time.perf_counter()
        
        # The actual YOLO model call (this is the expensive part)
        yolo_start = time.perf_counter()
        results = self.detector.model(frame, 
                                    classes=[0], 
                                    conf=self.detector.confidence_threshold,
                                    verbose=False)
        yolo_end = time.perf_counter()
        yolo_inference_time = (yolo_end - yolo_start) * 1000
        
        # Post-processing (convert YOLO results to Fighter objects)
        postprocess_start = time.perf_counter()
        
        fighters = []
        for result in results:
            if result.boxes is not None:
                for box in result.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    center_x = int((x1 + x2) / 2)
                    center_y = int((y1 + y2) / 2)
                    
                    if center_y < self.detector.min_y_position:
                        continue
                    
                    from detection.fighter_detector import Fighter
                    fighter = Fighter(
                        center=(center_x, center_y),
                        bbox=(int(x1), int(y1), int(x2), int(y2)),
                        confidence=box.conf[0].item()
                    )
                    fighters.append(fighter)
        
        # Assign to players (simplified version of FighterDetector logic)
        fighters.sort(key=lambda f: f.center[0])
        detection_result = None
        if len(fighters) >= 2:
            detection_result = (fighters[0].center, fighters[1].center)
        elif len(fighters) == 1:
            detection_result = (fighters[0].center, None)
        
        postprocess_end = time.perf_counter()
        postprocess_time = (postprocess_end - postprocess_start) * 1000
        
        end_time = time.perf_counter()
        total_detection_time = (end_time - start_time) * 1000
        
        return detection_result, yolo_inference_time, postprocess_time, total_detection_time
    
    def run_performance_test(self, num_frames: int = 100):
        """Run performance test for specified number of frames"""
        
        if not self.capture.is_valid:
            print(f"❌ Window '{self.window_title}' not found!")
            return
        
        print(f"🔬 Performance Analysis - Testing {num_frames} frames")
        print("=" * 60)
        print("Running tests...")
        
        successful_frames = 0
        
        for i in range(num_frames):
            try:
                # Measure total pipeline time
                pipeline_start = time.perf_counter()
                
                # 1. Screen capture
                frame, capture_time = self.measure_screen_capture()
                if frame is None:
                    continue
                
                # 2. YOLO detection
                detection, yolo_time, postprocess_time, _ = self.measure_yolo_inference(frame)
                
                pipeline_end = time.perf_counter()
                total_time = (pipeline_end - pipeline_start) * 1000
                
                # Store results
                self.capture_times.append(capture_time)
                self.yolo_times.append(yolo_time)
                self.postprocess_times.append(postprocess_time)
                self.total_times.append(total_time)
                
                successful_frames += 1
                
                # Progress indicator
                if (i + 1) % 10 == 0:
                    print(f"  Completed {i + 1}/{num_frames} frames...")
                    
            except Exception as e:
                print(f"Error on frame {i}: {e}")
                continue
        
        print(f"✅ Completed {successful_frames}/{num_frames} frames")
        return successful_frames
    
    def print_results(self):
        """Print detailed performance analysis"""
        if not self.total_times:
            print("❌ No timing data available")
            return
        
        def stats(times):
            return {
                'avg': statistics.mean(times),
                'min': min(times),
                'max': max(times),
                'median': statistics.median(times)
            }
        
        capture_stats = stats(self.capture_times)
        yolo_stats = stats(self.yolo_times) 
        postprocess_stats = stats(self.postprocess_times)
        total_stats = stats(self.total_times)
        
        print("\n" + "=" * 60)
        print("📊 PERFORMANCE ANALYSIS RESULTS")
        print("=" * 60)
        
        print(f"Screen Capture:   {capture_stats['avg']:.1f}ms avg ({capture_stats['min']:.1f}-{capture_stats['max']:.1f}ms range)")
        print(f"YOLO Inference:   {yolo_stats['avg']:.1f}ms avg ({yolo_stats['min']:.1f}-{yolo_stats['max']:.1f}ms range) {'⚠️  BOTTLENECK' if yolo_stats['avg'] > 15 else '✅'}")
        print(f"Post-processing:  {postprocess_stats['avg']:.1f}ms avg ({postprocess_stats['min']:.1f}-{postprocess_stats['max']:.1f}ms range)")
        print(f"Total Pipeline:   {total_stats['avg']:.1f}ms avg ({total_stats['min']:.1f}-{total_stats['max']:.1f}ms range)")
        
        print("\n" + "-" * 40)
        print("FPS ANALYSIS")
        print("-" * 40)
        
        target_60fps = 16.7  # ms per frame for 60fps
        target_30fps = 33.3  # ms per frame for 30fps
        actual_fps = 1000 / total_stats['avg']
        
        print(f"Target 60fps:     {target_60fps:.1f}ms per frame")
        print(f"Target 30fps:     {target_30fps:.1f}ms per frame") 
        print(f"Actual:           {total_stats['avg']:.1f}ms per frame ({actual_fps:.1f} fps)")
        
        if total_stats['avg'] <= target_60fps:
            print("✅ Can achieve 60fps")
        elif total_stats['avg'] <= target_30fps:
            print("⚠️  Can achieve 30fps but not 60fps")
        else:
            print("❌ Cannot achieve 30fps consistently")
        
        print("\n" + "-" * 40)
        print("BOTTLENECK ANALYSIS")
        print("-" * 40)
        
        yolo_percent = (yolo_stats['avg'] / total_stats['avg']) * 100
        capture_percent = (capture_stats['avg'] / total_stats['avg']) * 100
        postprocess_percent = (postprocess_stats['avg'] / total_stats['avg']) * 100
        
        print(f"YOLO Inference:   {yolo_percent:.1f}% of total time")
        print(f"Screen Capture:   {capture_percent:.1f}% of total time")
        print(f"Post-processing:  {postprocess_percent:.1f}% of total time")
        
        if yolo_percent > 70:
            print("\n💡 OPTIMIZATION SUGGESTIONS:")
            print("   - Use a smaller/faster YOLO model (yolov8n -> yolov8s)")
            print("   - Reduce input image resolution")
            print("   - Skip frames (detect every 2nd or 3rd frame)")
            print("   - Use GPU acceleration if available")

def main():
    """Main function"""
    profiler = DetectionPerformanceProfiler()
    
    try:
        # Run performance test
        num_frames = 50  # Start with 50 frames for quick results
        successful_frames = profiler.run_performance_test(num_frames)
        
        if successful_frames > 0:
            profiler.print_results()
        else:
            print("❌ No successful frames captured")
            
    except KeyboardInterrupt:
        print("\n⏹️  Test interrupted by user")
        if profiler.total_times:
            profiler.print_results()
    except Exception as e:
        print(f"❌ Error during performance test: {e}")

if __name__ == "__main__":
    main()