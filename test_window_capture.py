# test_window_capture.py
import cv2
import time
from window_capture import WindowCapture

def test_window_capture():
    """Test window capture with visualization"""
    WINDOW_TITLE = "Bloody Roar II (USA) [PlayStation] - BizHawk"
    
    capture = WindowCapture(WINDOW_TITLE)
    
    if not capture.is_valid:
        print(f"Error: Could not find window '{WINDOW_TITLE}'")
        print("Make sure BizHawk is running with the game loaded")
        return
    
    print(f"Window found!")
    print(f"Position: ({capture.window_info.x}, {capture.window_info.y})")
    print(f"Size: {capture.window_info.width}x{capture.window_info.height}")
    
    # Create preview window
    cv2.namedWindow('Window Capture Test', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Window Capture Test', 800, 600)
    
    print("\nCapturing... Press 'q' to quit")
    print("Press 'r' to test region capture (health bar area)")
    print("Press 'f' to test full window capture")
    
    mode = 'full'
    
    while True:
        start_time = time.perf_counter()
        
        if mode == 'full':
            # Capture full window
            image = capture.capture()
        else:
            # Capture just health bar region (as example)
            image = capture.capture_region(400, 100, 800, 100)
        
        if image is None:
            print("Failed to capture - is window still open?")
            time.sleep(1)
            continue
        
        # Calculate and display FPS
        fps = 1.0 / (time.perf_counter() - start_time)
        
        # Add info overlay
        display = image.copy()
        cv2.putText(display, f"Mode: {mode} | FPS: {fps:.1f}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                   1, (0, 255, 0), 2)
        
        # Show captured image
        cv2.imshow('Window Capture Test', display)
        
        # Handle keys
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            mode = 'region'
            print("Switched to region capture mode")
        elif key == ord('f'):
            mode = 'full'
            print("Switched to full capture mode")
    
    cv2.destroyAllWindows()
    print("\nTest completed!")

if __name__ == "__main__":
    test_window_capture()