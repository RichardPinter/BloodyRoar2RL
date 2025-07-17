import time
import statistics
import mss
import dxcam
import numpy as np

def test_mss_capture(region, iterations=100):
    """Test MSS capture speed for comparison."""
    timings = []
    
    with mss.mss() as sct:
        # Convert region format for MSS
        mss_region = {
            "left": region[0],
            "top": region[1], 
            "width": region[2] - region[0],
            "height": region[3] - region[1]
        }
        
        # Warm-up
        sct.grab(mss_region)
        
        for _ in range(iterations):
            start = time.perf_counter()
            img = sct.grab(mss_region)
            end = time.perf_counter()
            timings.append((end - start) * 1000)
    
    return timings

def test_dxcam_single_capture(region, iterations=100):
    """Test DXcam single capture speed."""
    timings = []
    
    # Create DXcam instance
    camera = dxcam.create()
    
    if camera is None:
        print("Failed to create DXcam camera!")
        return []
    
    # Warm-up
    camera.grab(region=region)
    
    for _ in range(iterations):
        start = time.perf_counter()
        frame = camera.grab(region=region)
        end = time.perf_counter()
        timings.append((end - start) * 1000)
    
    camera.release()
    return timings

def test_dxcam_streaming(region, duration=5):
    """Test DXcam streaming mode for continuous capture."""
    camera = dxcam.create()
    
    if camera is None:
        print("Failed to create DXcam camera!")
        return [], 0
    
    print(f"Testing DXcam streaming mode for {duration} seconds...")
    
    # Start streaming
    camera.start(target_fps=120, region=region)
    
    frame_times = []
    frame_count = 0
    start_time = time.time()
    
    try:
        while time.time() - start_time < duration:
            frame_start = time.perf_counter()
            frame = camera.get_latest_frame()
            
            if frame is not None:
                frame_time = (time.perf_counter() - frame_start) * 1000
                frame_times.append(frame_time)
                frame_count += 1
                
                # Progress update
                if frame_count % 60 == 0:
                    elapsed = time.time() - start_time
                    fps = frame_count / elapsed
                    print(f"  Frames: {frame_count}, FPS: {fps:.1f}")
    
    except KeyboardInterrupt:
        print("Stopped by user")
    
    camera.stop()
    camera.release()
    
    elapsed = time.time() - start_time
    actual_fps = frame_count / elapsed
    
    return frame_times, actual_fps

def print_comparison(mss_times, dxcam_times, test_name):
    """Print side-by-side comparison of MSS vs DXcam."""
    print(f"\n{test_name}")
    print("=" * 60)
    
    if not mss_times or not dxcam_times:
        print("ERROR: Missing timing data!")
        return
    
    mss_avg = statistics.mean(mss_times)
    dxcam_avg = statistics.mean(dxcam_times)
    speedup = mss_avg / dxcam_avg
    
    print(f"{'Metric':<20} {'MSS':<15} {'DXcam':<15} {'Speedup':<10}")
    print("-" * 60)
    print(f"{'Average (ms)':<20} {mss_avg:<15.3f} {dxcam_avg:<15.3f} {speedup:<10.2f}x")
    print(f"{'Min (ms)':<20} {min(mss_times):<15.3f} {min(dxcam_times):<15.3f}")
    print(f"{'Max (ms)':<20} {max(mss_times):<15.3f} {max(dxcam_times):<15.3f}")
    print(f"{'Median (ms)':<20} {statistics.median(mss_times):<15.3f} {statistics.median(dxcam_times):<15.3f}")
    print(f"{'StdDev (ms)':<20} {statistics.stdev(mss_times):<15.3f} {statistics.stdev(dxcam_times):<15.3f}")
    print(f"{'Potential FPS':<20} {1000/mss_avg:<15.1f} {1000/dxcam_avg:<15.1f}")
    
    if speedup > 1.1:
        print(f"\n✓ DXcam is {speedup:.1f}x FASTER than MSS!")
    elif speedup < 0.9:
        print(f"\n✗ DXcam is {1/speedup:.1f}x SLOWER than MSS!")
    else:
        print(f"\n≈ DXcam and MSS have similar performance")

def test_api_availability():
    """Test if DXcam can actually access Desktop Duplication API."""
    print("Testing DXcam API availability...")
    
    try:
        camera = dxcam.create()
        if camera is None:
            print("✗ Failed to create DXcam camera - Desktop Duplication API not available")
            return False
        
        # Try to capture
        frame = camera.grab()
        if frame is None:
            print("✗ DXcam camera created but failed to capture")
            camera.release()
            return False
        
        print(f"✓ DXcam working - captured {frame.shape} frame")
        camera.release()
        return True
        
    except Exception as e:
        print(f"✗ DXcam error: {e}")
        return False

def main():
    print("DXcam vs MSS Performance Comparison")
    print("=" * 70)
    
    # First, test if DXcam actually works
    if not test_api_availability():
        print("\nDXcam not available - falling back to MSS analysis only")
        return
    
    # Test regions (left, top, right, bottom)
    test_cases = [
        ("Small (160x192)", (-1920, 76, -1760, 268)),  # Your leftmost monitor
        ("Medium (320x240)", (-1920, 76, -1600, 316)),
        ("Large (640x480)", (-1920, 76, -1280, 556)),
        ("Primary monitor small", (100, 100, 260, 292)),  # Primary monitor
    ]
    
    for test_name, region in test_cases:
        print(f"\n" + "=" * 70)
        print(f"Testing: {test_name}")
        print(f"Region: {region}")
        
        # Test MSS
        print("Testing MSS...")
        mss_times = test_mss_capture(region, iterations=50)
        
        # Test DXcam single capture
        print("Testing DXcam single capture...")
        dxcam_times = test_dxcam_single_capture(region, iterations=50)
        
        # Compare results
        if mss_times and dxcam_times:
            print_comparison(mss_times, dxcam_times, f"{test_name} - Single Capture")
        
        # Test streaming mode for small region only
        if "Small" in test_name:
            print(f"\nTesting DXcam streaming mode...")
            stream_times, stream_fps = test_dxcam_streaming(region, duration=3)
            
            if stream_times:
                print(f"\nDXcam Streaming Results:")
                print(f"  Frames captured: {len(stream_times)}")
                print(f"  Average FPS: {stream_fps:.1f}")
                print(f"  Average frame time: {statistics.mean(stream_times):.3f}ms")
                print(f"  Min frame time: {min(stream_times):.3f}ms")
                print(f"  Max frame time: {max(stream_times):.3f}ms")
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("If DXcam is significantly faster:")
    print("  - The Desktop Duplication API is working")
    print("  - Your 16ms MSS limitation is bypassed")
    print("  - Perfect for RL training!")
    print("\nIf DXcam is similar/slower:")
    print("  - May be hitting same Windows limitations")
    print("  - Try disabling Hardware Accelerated GPU Scheduling")
    print("  - Consider other Windows optimizations")

if __name__ == "__main__":
    main()