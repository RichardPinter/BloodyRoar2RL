import mss
import time

def show_monitor_info():
    """Display information about all available monitors."""
    with mss.mss() as sct:
        print("Monitor Information:")
        print("=" * 50)
        
        # monitors[0] is all monitors combined
        for i, monitor in enumerate(sct.monitors):
            print(f"\nMonitor {i}:")
            print(f"  Left: {monitor['left']}")
            print(f"  Top: {monitor['top']}")
            print(f"  Width: {monitor['width']}")
            print(f"  Height: {monitor['height']}")
            
            if i == 0:
                print("  (This is the combined virtual screen)")
            elif i == 1:
                print("  (Primary monitor)")
            else:
                print(f"  (Secondary monitor {i-1})")

def capture_from_specific_monitor(monitor_index=1, iterations=10):
    """Capture from a specific monitor and measure timing."""
    timings = []
    
    with mss.mss() as sct:
        if monitor_index >= len(sct.monitors):
            print(f"Monitor {monitor_index} not found!")
            return
        
        monitor = sct.monitors[monitor_index]
        print(f"\nCapturing from Monitor {monitor_index}:")
        print(f"  Position: ({monitor['left']}, {monitor['top']})")
        print(f"  Size: {monitor['width']}x{monitor['height']}")
        
        # Warm-up
        sct.grab(monitor)
        
        for i in range(iterations):
            start = time.perf_counter()
            img = sct.grab(monitor)
            end = time.perf_counter()
            timings.append((end - start) * 1000)
            
            if i == 0:
                # Save first capture to see what we got
                mss.tools.to_png(img.rgb, img.size, output=f"monitor_{monitor_index}_capture.png")
                print(f"  Saved first capture to monitor_{monitor_index}_capture.png")
        
        avg_time = sum(timings) / len(timings)
        print(f"  Average capture time: {avg_time:.2f} ms")

def capture_game_region(left, top, width, height):
    """Capture a specific region (where your game window is)."""
    region = {"left": left, "top": top, "width": width, "height": height}
    
    with mss.mss() as sct:
        print(f"\nCapturing region:")
        print(f"  Position: ({left}, {top})")
        print(f"  Size: {width}x{height}")
        
        # Capture and save
        img = sct.grab(region)
        mss.tools.to_png(img.rgb, img.size, output="game_region.png")
        print("  Saved to game_region.png")
        
        # Quick timing test
        timings = []
        for _ in range(10):
            start = time.perf_counter()
            img = sct.grab(region)
            end = time.perf_counter()
            timings.append((end - start) * 1000)
        
        avg_time = sum(timings) / len(timings)
        print(f"  Average capture time: {avg_time:.2f} ms")

if __name__ == "__main__":
    print("MSS Multi-Monitor Setup")
    print("=" * 70)
    
    # Show all monitors
    show_monitor_info()
    
    print("\n" + "=" * 70)
    print("Since your game is on the leftmost screen, it's likely:")
    print("- Monitor 1 if that's your primary")
    print("- Monitor 2 or 3 if it's a secondary")
    print("- The monitor with the most negative 'left' value")
    
    # Test captures from different monitors
    with mss.mss() as sct:
        num_monitors = len(sct.monitors) - 1  # Exclude the combined virtual screen
        
        if num_monitors > 1:
            print(f"\nYou have {num_monitors} monitors. Testing each...")
            for i in range(1, len(sct.monitors)):
                capture_from_specific_monitor(i, iterations=5)
    
    print("\n" + "=" * 70)
    print("NEXT STEPS:")
    print("1. Check the saved PNG files to identify which monitor has your game")
    print("2. Note the monitor's left/top coordinates")
    print("3. Add your game window's offset within that monitor")
    print("4. Use those coordinates for fast region capture")
    
    print("\nExample: If your game is at (100, 100) on the leftmost monitor")
    print("and that monitor starts at (-1920, 0), then your game region is:")
    print("left = -1920 + 100 = -1820")
    print("top = 0 + 100 = 100")