import mss
import time
import statistics

# Your leftmost monitor coordinates
LEFTMOST_MONITOR = {
    "left": -1920,
    "top": 76,
    "width": 1920,
    "height": 1080
}

def find_game_window_coordinates():
    """
    Helper to find where your BizHawk window is on the leftmost monitor.
    You'll need to adjust these based on where you position your window.
    """
    # Common BizHawk window sizes
    print("Common emulator window sizes:")
    print("- NES: 256x240 (plus window borders)")
    print("- SNES: 256x224")
    print("- Genesis: 320x224")
    print("- Atari 2600: 160x192")
    print("\nYou'll need to add your window's position relative to the monitor")
    
    # Example positions (you'll need to adjust these)
    examples = [
        # If BizHawk is at top-left of the leftmost monitor
        {"name": "Top-left", "left": -1920 + 10, "top": 76 + 30},
        # If centered on the leftmost monitor
        {"name": "Centered", "left": -1920 + 640, "top": 76 + 300},
        # Custom position
        {"name": "Custom", "left": -1920 + 100, "top": 76 + 100}
    ]
    
    return examples

def benchmark_game_capture(left, top, width, height, iterations=100):
    """Benchmark capturing a specific game window region."""
    region = {"left": left, "top": top, "width": width, "height": height}
    timings = []
    
    with mss.mss() as sct:
        print(f"\nBenchmarking capture region:")
        print(f"  Position: ({left}, {top})")
        print(f"  Size: {width}x{height}")
        print(f"  Monitor: Leftmost (X from {LEFTMOST_MONITOR['left']} to {LEFTMOST_MONITOR['left'] + LEFTMOST_MONITOR['width']})")
        
        # Warm-up and save test image
        img = sct.grab(region)
        mss.tools.to_png(img.rgb, img.size, output="bizhawk_test.png")
        print("  Saved test capture to bizhawk_test.png - CHECK THIS!")
        
        # Benchmark
        for _ in range(iterations):
            start = time.perf_counter()
            img = sct.grab(region)
            end = time.perf_counter()
            timings.append((end - start) * 1000)
        
        # Stats
        print(f"\nResults ({iterations} captures):")
        print(f"  Average: {statistics.mean(timings):.3f} ms")
        print(f"  Min: {min(timings):.3f} ms")
        print(f"  Max: {max(timings):.3f} ms")
        print(f"  Median: {statistics.median(timings):.3f} ms")
        print(f"  StdDev: {statistics.stdev(timings):.3f} ms")
        print(f"  Potential FPS: {1000 / statistics.mean(timings):.1f}")
        
        return timings

def compare_different_sizes():
    """Compare capture speeds for different window sizes."""
    # Adjust these coordinates based on where your BizHawk window actually is!
    # These assume the window is 100 pixels from the left edge of the leftmost monitor
    window_left = -1920 + 100
    window_top = 76 + 100
    
    test_cases = [
        # Name, width, height
        ("Atari 2600 (160x192)", 160, 192),
        ("NES (256x240)", 256, 240),
        ("SNES (256x224)", 256, 224),
        ("Genesis (320x224)", 320, 224),
        ("Larger window (640x480)", 640, 480),
        ("HD window (1280x720)", 1280, 720),
    ]
    
    print("\nComparing capture speeds for different window sizes")
    print("=" * 60)
    
    results = []
    for name, width, height in test_cases:
        print(f"\nTesting {name}...")
        timings = benchmark_game_capture(window_left, window_top, width, height, iterations=50)
        avg_ms = statistics.mean(timings)
        results.append((name, avg_ms))
    
    print("\n" + "=" * 60)
    print("SUMMARY - Capture Speed by Resolution:")
    print("=" * 60)
    for name, avg_ms in sorted(results, key=lambda x: x[1]):
        print(f"{name:.<30} {avg_ms:.2f} ms ({1000/avg_ms:.0f} FPS)")

if __name__ == "__main__":
    print("BizHawk Capture Benchmark (Leftmost Monitor)")
    print("=" * 70)
    print(f"Your leftmost monitor: X={LEFTMOST_MONITOR['left']} to {LEFTMOST_MONITOR['left'] + LEFTMOST_MONITOR['width']}")
    print(f"                      Y={LEFTMOST_MONITOR['top']} to {LEFTMOST_MONITOR['top'] + LEFTMOST_MONITOR['height']}")
    
    print("\n" + "=" * 70)
    print("IMPORTANT: You need to know where your BizHawk window is positioned!")
    print("=" * 70)
    
    # Show examples
    examples = find_game_window_coordinates()
    
    # Test with an estimated position (ADJUST THESE!)
    # Assuming BizHawk is near the top-left of your leftmost monitor
    game_left = -1920 + 100  # 100 pixels from left edge of leftmost monitor
    game_top = 76 + 100      # 100 pixels from top
    game_width = 640         # Typical game window width
    game_height = 480        # Typical game window height
    
    print(f"\nUsing test coordinates: ({game_left}, {game_top}) with size {game_width}x{game_height}")
    print("If the capture is wrong, adjust the coordinates in the script!\n")
    
    # Single benchmark test
    benchmark_game_capture(game_left, game_top, game_width, game_height)
    
    # Compare different sizes
    print("\n" + "=" * 70)
    input("Press Enter to test different window sizes...")
    compare_different_sizes()
    
    print("\n" + "=" * 70)
    print("TIPS:")
    print("1. Check bizhawk_test.png to verify you're capturing the right area")
    print("2. Smaller capture regions are MUCH faster")
    print("3. For 60 FPS emulation, you need <16.67ms per capture")
    print("4. These times leave plenty of room for processing!")