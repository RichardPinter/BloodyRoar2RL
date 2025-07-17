import time
import statistics
import mss
import dxcam
import numpy as np

def explore_dxcam_capabilities():
    """Explore DXcam's monitor selection and coordinate system."""
    print("Exploring DXcam capabilities...")
    print("=" * 50)
    
    # Try to get available devices/monitors
    try:
        # Check if DXcam has device listing
        devices = dxcam.device_info()
        print(f"DXcam devices: {devices}")
    except:
        print("DXcam device_info() not available")
    
    # Try different camera creation methods
    print("\nTesting camera creation methods:")
    
    # Default camera
    camera1 = dxcam.create()
    if camera1:
        print(f"  Default camera: {camera1.width}x{camera1.height}")
        # Test full screen capture
        frame = camera1.grab()
        if frame is not None:
            print(f"    Full capture: {frame.shape}")
        camera1.release()
    
    # Try specific device indices
    for device_idx in range(3):  # Try first 3 devices
        try:
            camera = dxcam.create(device_idx=device_idx)
            if camera:
                print(f"  Device {device_idx}: {camera.width}x{camera.height}")
                camera.release()
        except Exception as e:
            print(f"  Device {device_idx}: Failed - {e}")
    
    # Try output indices (monitors)
    for output_idx in range(3):  # Try first 3 outputs
        try:
            camera = dxcam.create(output_idx=output_idx)
            if camera:
                print(f"  Output {output_idx}: {camera.width}x{camera.height}")
                camera.release()
        except Exception as e:
            print(f"  Output {output_idx}: Failed - {e}")

def test_mss_vs_dxcam_same_monitor():
    """Test MSS vs DXcam on the same monitor with proper coordinates."""
    print("\nTesting MSS vs DXcam on primary monitor...")
    print("=" * 50)
    
    # Test regions on primary monitor (0,0 to 1920,1080)
    regions = [
        ("Small (160x192)", (100, 100, 260, 292)),
        ("Medium (320x240)", (100, 100, 420, 340)),
        ("Large (640x480)", (100, 100, 740, 580)),
        ("Extra Large (800x600)", (100, 100, 900, 700))
    ]
    
    for test_name, region in regions:
        print(f"\n{test_name} - Region: {region}")
        
        # Test MSS (convert to MSS format)
        mss_region = {
            "left": region[0],
            "top": region[1],
            "width": region[2] - region[0],
            "height": region[3] - region[1]
        }
        
        print("  Testing MSS...")
        mss_times = []
        with mss.mss() as sct:
            # Use primary monitor offset if needed
            monitor = sct.monitors[1]  # Primary monitor
            adjusted_region = {
                "left": monitor['left'] + mss_region['left'],
                "top": monitor['top'] + mss_region['top'],
                "width": mss_region['width'],
                "height": mss_region['height']
            }
            
            for _ in range(30):
                start = time.perf_counter()
                img = sct.grab(adjusted_region)
                end = time.perf_counter()
                mss_times.append((end - start) * 1000)
        
        # Test DXcam
        print("  Testing DXcam...")
        dxcam_times = []
        camera = dxcam.create()
        
        if camera:
            for _ in range(30):
                start = time.perf_counter()
                frame = camera.grab(region=region)
                end = time.perf_counter()
                dxcam_times.append((end - start) * 1000)
            
            camera.release()
        
        # Compare results
        if mss_times and dxcam_times:
            mss_avg = statistics.mean(mss_times)
            dxcam_avg = statistics.mean(dxcam_times)
            speedup = mss_avg / dxcam_avg
            
            print(f"    MSS avg: {mss_avg:.3f}ms")
            print(f"    DXcam avg: {dxcam_avg:.3f}ms")
            print(f"    Speedup: {speedup:.2f}x")
            
            if speedup > 1.2:
                print(f"    ✓ DXcam is {speedup:.1f}x faster!")
            elif speedup < 0.8:
                print(f"    ✗ DXcam is {1/speedup:.1f}x slower!")
            else:
                print(f"    ≈ Similar performance")

def test_dxcam_streaming_performance():
    """Test DXcam's streaming mode for maximum performance."""
    print("\nTesting DXcam streaming mode performance...")
    print("=" * 50)
    
    camera = dxcam.create()
    if not camera:
        print("Failed to create DXcam camera")
        return
    
    # Test different target FPS values
    target_fps_values = [60, 120, 240]
    region = (100, 100, 260, 292)  # Small region
    
    for target_fps in target_fps_values:
        print(f"\nTesting target_fps={target_fps}")
        
        try:
            camera.start(target_fps=target_fps, region=region)
            
            frame_times = []
            frame_count = 0
            start_time = time.time()
            test_duration = 3  # seconds
            
            while time.time() - start_time < test_duration:
                loop_start = time.perf_counter()
                frame = camera.get_latest_frame()
                
                if frame is not None:
                    frame_time = (time.perf_counter() - loop_start) * 1000
                    frame_times.append(frame_time)
                    frame_count += 1
            
            camera.stop()
            
            elapsed = time.time() - start_time
            actual_fps = frame_count / elapsed
            
            if frame_times:
                avg_time = statistics.mean(frame_times)
                print(f"    Actual FPS: {actual_fps:.1f}")
                print(f"    Average frame time: {avg_time:.3f}ms")
                print(f"    Min frame time: {min(frame_times):.3f}ms")
                print(f"    Max frame time: {max(frame_times):.3f}ms")
                
                # Check if we beat the 16ms barrier
                if avg_time < 10:
                    print(f"    ✓ Significantly faster than MSS!")
                elif avg_time < 16:
                    print(f"    ✓ Faster than MSS")
                else:
                    print(f"    ✗ Still hitting timing limits")
        
        except Exception as e:
            print(f"    Error: {e}")
    
    camera.release()

def test_leftmost_monitor_workaround():
    """Try to capture from leftmost monitor area using different approaches."""
    print("\nTesting leftmost monitor capture workarounds...")
    print("=" * 50)
    
    # If DXcam can't select monitors, maybe we can capture a larger area
    # and crop to the leftmost region
    
    camera = dxcam.create()
    if not camera:
        print("Failed to create DXcam camera")
        return
    
    print(f"DXcam camera resolution: {camera.width}x{camera.height}")
    
    # Try to capture full screen and see what we get
    print("\nTesting full screen capture...")
    start = time.perf_counter()
    full_frame = camera.grab()
    end = time.perf_counter()
    
    if full_frame is not None:
        print(f"  Full screen shape: {full_frame.shape}")
        print(f"  Capture time: {(end-start)*1000:.3f}ms")
        
        # If this is multi-monitor, maybe we can find the leftmost area
        # This would be a workaround if DXcam captures all monitors
        if full_frame.shape[1] > 1920:  # Width > single monitor
            print(f"  Possible multi-monitor capture detected!")
            print(f"  You might be able to crop the leftmost region")
    
    camera.release()

def main():
    print("DXcam Fixed Performance Test")
    print("=" * 70)
    
    # Explore what DXcam can do
    explore_dxcam_capabilities()
    
    # Test on primary monitor with proper coordinates
    test_mss_vs_dxcam_same_monitor()
    
    # Test streaming performance
    test_dxcam_streaming_performance()
    
    # Try leftmost monitor workaround
    test_leftmost_monitor_workaround()
    
    print("\n" + "=" * 70)
    print("CONCLUSIONS:")
    print("=" * 70)
    print("1. If DXcam shows <10ms captures: It's genuinely faster!")
    print("2. If DXcam shows ~16ms: It's hitting same limits as MSS")
    print("3. If DXcam can't access leftmost monitor: Limitation for your setup")
    print("4. Monitor selection capabilities determine usability for your RL setup")

if __name__ == "__main__":
    main()