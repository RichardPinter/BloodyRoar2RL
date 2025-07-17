import time
import dxcam
import numpy as np
import cv2

def simple_capture_test():
    """Simple single capture test - no loops, no complexity."""
    print("Simple DXcam Capture Test")
    print("=" * 40)
    
    # Create camera
    print("Creating camera...")
    camera = dxcam.create()
    
    if camera is None:
        print("Failed to create camera!")
        return
    
    print(f"Camera created: {camera.width}x{camera.height}")
    
    # Single capture test
    print("\nTesting single capture...")
    
    try:
        start = time.perf_counter()
        frame = camera.grab()
        capture_time = (time.perf_counter() - start) * 1000
        
        if frame is not None:
            print(f"✓ Capture successful: {frame.shape}")
            print(f"  Capture time: {capture_time:.3f}ms")
            
            # Test preprocessing
            print("\nTesting preprocessing...")
            preprocess_start = time.perf_counter()
            
            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Resize to 84x84 (standard RL size)
            resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
            
            preprocess_time = (time.perf_counter() - preprocess_start) * 1000
            
            print(f"✓ Preprocessing successful: {resized.shape}")
            print(f"  Preprocessing time: {preprocess_time:.3f}ms")
            print(f"  Total time: {capture_time + preprocess_time:.3f}ms")
            print(f"  Potential FPS: {1000/(capture_time + preprocess_time):.1f}")
            
            # Save both images for inspection
            cv2.imwrite("dxcam_original.png", frame)
            cv2.imwrite("dxcam_processed.png", resized)
            print(f"\n✓ Saved images: dxcam_original.png, dxcam_processed.png")
            
        else:
            print("✗ Capture failed - no frame returned")
            
    except Exception as e:
        print(f"✗ Capture error: {e}")
    
    finally:
        # Clean up
        print("\nCleaning up...")
        try:
            camera.release()
            print("✓ Camera released")
        except Exception as e:
            print(f"✗ Cleanup error: {e}")

def test_small_region():
    """Test capturing a small region."""
    print("\n" + "=" * 40)
    print("Testing Small Region Capture")
    print("=" * 40)
    
    camera = dxcam.create()
    
    if camera is None:
        print("Failed to create camera!")
        return
    
    # Small region (160x192 like Atari)
    region = (100, 100, 260, 292)
    print(f"Testing region: {region}")
    
    try:
        start = time.perf_counter()
        frame = camera.grab(region=region)
        capture_time = (time.perf_counter() - start) * 1000
        
        if frame is not None:
            print(f"✓ Small region capture: {frame.shape}")
            print(f"  Capture time: {capture_time:.3f}ms")
            
            # Preprocessing
            preprocess_start = time.perf_counter()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
            preprocess_time = (time.perf_counter() - preprocess_start) * 1000
            
            print(f"  Preprocessing time: {preprocess_time:.3f}ms")
            print(f"  Total time: {capture_time + preprocess_time:.3f}ms")
            print(f"  Potential FPS: {1000/(capture_time + preprocess_time):.1f}")
            
            cv2.imwrite("dxcam_small_region.png", frame)
            cv2.imwrite("dxcam_small_processed.png", resized)
            print(f"✓ Saved: dxcam_small_region.png, dxcam_small_processed.png")
            
        else:
            print("✗ Small region capture failed")
            
    except Exception as e:
        print(f"✗ Small region error: {e}")
    
    finally:
        try:
            camera.release()
        except:
            pass

def compare_with_mss():
    """Quick comparison with MSS for the same region."""
    print("\n" + "=" * 40)
    print("Comparing DXcam vs MSS")
    print("=" * 40)
    
    import mss
    
    region_dxcam = (100, 100, 260, 292)  # DXcam format
    region_mss = {"left": 100, "top": 100, "width": 160, "height": 192}  # MSS format
    
    # Test MSS
    print("Testing MSS...")
    try:
        with mss.mss() as sct:
            start = time.perf_counter()
            img = sct.grab(region_mss)
            mss_time = (time.perf_counter() - start) * 1000
            print(f"  MSS capture time: {mss_time:.3f}ms")
    except Exception as e:
        print(f"  MSS error: {e}")
        mss_time = None
    
    # Test DXcam
    print("Testing DXcam...")
    dxcam_time = None
    camera = dxcam.create()
    
    if camera:
        try:
            start = time.perf_counter()
            frame = camera.grab(region=region_dxcam)
            dxcam_time = (time.perf_counter() - start) * 1000
            
            if frame is not None:
                print(f"  DXcam capture time: {dxcam_time:.3f}ms")
            else:
                print("  DXcam capture failed")
                dxcam_time = None
                
        except Exception as e:
            print(f"  DXcam error: {e}")
        
        try:
            camera.release()
        except:
            pass
    
    # Compare
    if mss_time and dxcam_time:
        speedup = mss_time / dxcam_time
        print(f"\nComparison:")
        print(f"  MSS: {mss_time:.3f}ms")
        print(f"  DXcam: {dxcam_time:.3f}ms")
        print(f"  Speedup: {speedup:.2f}x")
        
        if speedup > 2:
            print(f"  ✓ DXcam is significantly faster!")
        elif speedup > 1.2:
            print(f"  ✓ DXcam is faster")
        else:
            print(f"  ≈ Similar performance")

if __name__ == "__main__":
    # Step 1: Basic capture test
    simple_capture_test()
    
    # Step 2: Small region test
    test_small_region()
    
    # Step 3: Quick comparison
    compare_with_mss()
    
    print("\n" + "=" * 40)
    print("SUMMARY")
    print("=" * 40)
    print("Check the saved images to verify capture is working")
    print("If DXcam times are <5ms, you have a fast solution!")
    print("If both methods are slow, it's a system-wide issue")