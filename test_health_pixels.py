import time
import numpy as np
from window_capture import WindowCapture
from health_detector import HealthDetector, HealthBarConfig

def test_health_bar_pixels():
    """
    Test function to show percentage of yellow pixels in health bar regions
    Prints results to terminal frame by frame
    """
    
    WINDOW_TITLE = "Bloody Roar II (USA) [PlayStation] - BizHawk"
    
    # Initialize components
    capture = WindowCapture(WINDOW_TITLE)
    health_detector = HealthDetector()
    config = health_detector.config
    
    if not capture.is_valid:
        print(f"ERROR: Window '{WINDOW_TITLE}' not found!")
        print("Make sure BizHawk is running with Bloody Roar 2")
        return
    
    print("Health Bar Pixel Test")
    print("=" * 60)
    print(f"P1 Health Bar: x={config.p1_x}, y={config.bar_y}, width={config.bar_length}")
    print(f"P2 Health Bar: x={config.p2_x - config.bar_length}, y={config.bar_y}, width={config.bar_length}")
    print(f"Yellow color range: BGR({config.lower_bgr}) to BGR({config.upper_bgr})")
    print("=" * 60)
    print("Press Ctrl+C to stop")
    print()
    
    frame_count = 0
    start_time = time.time()
    
    try:
        while True:
            frame_count += 1
            
            # Capture P1 health bar region
            p1_strip = capture.capture_region(
                x=config.p1_x,
                y=config.bar_y,
                width=config.bar_length,
                height=config.bar_height
            )
            
            # Capture P2 health bar region
            p2_strip = capture.capture_region(
                x=config.p2_x - config.bar_length,
                y=config.bar_y,
                width=config.bar_length,
                height=config.bar_height
            )
            
            # Check if we got valid captures
            if p1_strip is None or p2_strip is None:
                print(f"\rFrame {frame_count:4d} | ERROR: Failed to capture health bar regions", end='', flush=True)
                time.sleep(0.1)
                continue
            
            # Calculate yellow pixel percentages
            p1_yellow_pct, p1_yellow_count = calculate_yellow_percentage(p1_strip, config)
            p2_yellow_pct, p2_yellow_count = calculate_yellow_percentage(p2_strip, config)
            
            # Calculate total pixels in each region
            total_pixels = config.bar_length * config.bar_height
            
            # Print results
            elapsed = time.time() - start_time
            print(f"\rFrame {frame_count:4d} ({elapsed:6.1f}s) | P1: {p1_yellow_pct:5.1f}% ({p1_yellow_count:3d}/{total_pixels}) | P2: {p2_yellow_pct:5.1f}% ({p2_yellow_count:3d}/{total_pixels})", 
                  end='', flush=True)
            
            # Small delay to avoid overwhelming the terminal
            time.sleep(0.05)  # ~20fps
            
    except KeyboardInterrupt:
        print(f"\n\nTest stopped after {frame_count} frames ({time.time() - start_time:.1f}s)")
        print("Health bar pixel test completed")

def calculate_yellow_percentage(pixel_strip, config):
    """
    Calculate percentage of yellow pixels in a health bar strip
    
    Args:
        pixel_strip: BGR pixel array
        config: HealthBarConfig with color ranges
        
    Returns:
        (percentage, yellow_pixel_count)
    """
    if pixel_strip is None:
        return 0.0, 0
    
    # Handle different array shapes
    if len(pixel_strip.shape) == 3:
        if pixel_strip.shape[0] == 1:
            # Shape is (1, width, 3) - remove height dimension
            pixel_strip = pixel_strip[0]
        elif pixel_strip.shape[2] == 3:
            # Shape is (height, width, 3) - flatten height
            pixel_strip = pixel_strip.reshape(-1, 3)
    
    # Extract BGR channels
    b = pixel_strip[:, 0]
    g = pixel_strip[:, 1]
    r = pixel_strip[:, 2]
    
    # Create mask for yellow pixels
    yellow_mask = (
        (r >= config.lower_bgr[2]) & (r <= config.upper_bgr[2]) &
        (g >= config.lower_bgr[1]) & (g <= config.upper_bgr[1]) &
        (b >= config.lower_bgr[0]) & (b <= config.upper_bgr[0])
    )
    
    # Count yellow pixels
    yellow_count = np.sum(yellow_mask)
    total_pixels = len(pixel_strip)
    
    # Calculate percentage
    if total_pixels > 0:
        percentage = (yellow_count / total_pixels) * 100.0
    else:
        percentage = 0.0
    
    return percentage, yellow_count

def test_health_detection_comparison():
    """
    Test that compares our pixel counting with the existing health detector
    """
    
    WINDOW_TITLE = "Bloody Roar II (USA) [PlayStation] - BizHawk"
    
    # Initialize components
    capture = WindowCapture(WINDOW_TITLE)
    health_detector = HealthDetector()
    
    if not capture.is_valid:
        print(f"ERROR: Window '{WINDOW_TITLE}' not found!")
        return
    
    print("Health Detection Comparison Test")
    print("=" * 80)
    print("Comparing pixel percentage vs health detector results")
    print("Press Ctrl+C to stop")
    print()
    
    frame_count = 0
    
    try:
        while True:
            frame_count += 1
            
            # Get health detection results
            health_state = health_detector.detect(capture)
            
            if health_state:
                # Also get raw pixel percentages
                config = health_detector.config
                
                p1_strip = capture.capture_region(
                    x=config.p1_x, y=config.bar_y,
                    width=config.bar_length, height=config.bar_height
                )
                p2_strip = capture.capture_region(
                    x=config.p2_x - config.bar_length, y=config.bar_y,
                    width=config.bar_length, height=config.bar_height
                )
                
                if p1_strip is not None and p2_strip is not None:
                    p1_yellow_pct, p1_count = calculate_yellow_percentage(p1_strip, config)
                    p2_yellow_pct, p2_count = calculate_yellow_percentage(p2_strip, config)
                    
                    print(f"\rFrame {frame_count:4d} | Health: P1={health_state.p1_health:5.1f}% P2={health_state.p2_health:5.1f}% | Yellow: P1={p1_yellow_pct:5.1f}% P2={p2_yellow_pct:5.1f}%", 
                          end='', flush=True)
                else:
                    print(f"\rFrame {frame_count:4d} | Health: P1={health_state.p1_health:5.1f}% P2={health_state.p2_health:5.1f}% | Yellow: ERROR", 
                          end='', flush=True)
            else:
                print(f"\rFrame {frame_count:4d} | Health: FAILED | Yellow: N/A", end='', flush=True)
            
            time.sleep(0.1)
            
    except KeyboardInterrupt:
        print(f"\n\nComparison test stopped after {frame_count} frames")

if __name__ == "__main__":
    print("Health Bar Pixel Tests")
    print("=" * 40)
    print("1. Pixel percentage test")
    print("2. Health detection comparison")
    print("3. Exit")
    
    choice = input("\nEnter choice (1-3): ").strip()
    
    if choice == '1':
        test_health_bar_pixels()
    elif choice == '2':
        test_health_detection_comparison()
    elif choice == '3':
        print("Exiting...")
    else:
        print("Invalid choice, running pixel percentage test...")
        test_health_bar_pixels()