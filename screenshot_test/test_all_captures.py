import mss
import time
import statistics
from collections import defaultdict

def test_single_capture(region, iterations=50):
    """Test capture speed for a specific region."""
    timings = []
    
    with mss.mss() as sct:
        # Warm-up
        sct.grab(region)
        
        for _ in range(iterations):
            start = time.perf_counter()
            img = sct.grab(region)
            end = time.perf_counter()
            timings.append((end - start) * 1000)
    
    return {
        "avg": statistics.mean(timings),
        "min": min(timings),
        "max": max(timings),
        "median": statistics.median(timings)
    }

def test_all_monitors():
    """Test capture from different positions on all monitors."""
    results = []
    
    with mss.mss() as sct:
        print("Available Monitors:")
        print("="*70)
        
        for i, monitor in enumerate(sct.monitors):
            if i == 0:  # Skip the combined virtual screen
                continue
                
            print(f"\nMonitor {i}:")
            print(f"  Position: ({monitor['left']}, {monitor['top']})")
            print(f"  Size: {monitor['width']}x{monitor['height']}")
            
            # Test different positions on this monitor
            test_cases = [
                # (name, left_offset, top_offset, width, height)
                ("Top-left corner", 0, 0, 160, 192),
                ("Center small", monitor['width']//2 - 80, monitor['height']//2 - 96, 160, 192),
                ("Bottom-right corner", monitor['width'] - 160, monitor['height'] - 192, 160, 192),
                ("Medium size center", monitor['width']//2 - 160, monitor['height']//2 - 120, 320, 240),
                ("Large size center", monitor['width']//2 - 320, monitor['height']//2 - 240, 640, 480),
                ("Full monitor", 0, 0, monitor['width'], monitor['height'])
            ]
            
            for test_name, left_off, top_off, width, height in test_cases:
                # Skip if region would go outside monitor bounds
                if left_off < 0 or top_off < 0 or left_off + width > monitor['width'] or top_off + height > monitor['height']:
                    continue
                    
                region = {
                    "left": monitor['left'] + left_off,
                    "top": monitor['top'] + top_off,
                    "width": width,
                    "height": height
                }
                
                print(f"\n  Testing: {test_name} ({width}x{height})")
                print(f"    Absolute position: ({region['left']}, {region['top']})")
                
                result = test_single_capture(region, iterations=30)
                
                results.append({
                    "monitor": i,
                    "monitor_pos": (monitor['left'], monitor['top']),
                    "test": test_name,
                    "region": region,
                    "size": f"{width}x{height}",
                    "stats": result
                })
                
                print(f"    Average: {result['avg']:.2f}ms")
                print(f"    Min/Max: {result['min']:.2f}ms / {result['max']:.2f}ms")
    
    return results

def analyze_results(results):
    """Analyze and summarize the results."""
    print("\n" + "="*70)
    print("CAPTURE SPEED ANALYSIS")
    print("="*70)
    
    # Group by size
    by_size = defaultdict(list)
    for r in results:
        by_size[r['size']].append((r['stats']['avg'], r))
    
    print("\nAverage capture time by size:")
    for size, items in sorted(by_size.items()):
        avg_times = [t[0] for t in items]
        overall_avg = statistics.mean(avg_times)
        print(f"  {size}: {overall_avg:.2f}ms")
    
    # Find fastest and slowest
    sorted_results = sorted(results, key=lambda x: x['stats']['avg'])
    
    print("\nFastest captures:")
    for r in sorted_results[:3]:
        print(f"  Monitor {r['monitor']} - {r['test']}: {r['stats']['avg']:.2f}ms")
        print(f"    Position: ({r['region']['left']}, {r['region']['top']})")
    
    print("\nSlowest captures:")
    for r in sorted_results[-3:]:
        print(f"  Monitor {r['monitor']} - {r['test']}: {r['stats']['avg']:.2f}ms")
        print(f"    Position: ({r['region']['left']}, {r['region']['top']})")
    
    # Check if negative coordinates are slower
    negative_x = [r for r in results if r['region']['left'] < 0]
    positive_x = [r for r in results if r['region']['left'] >= 0]
    
    if negative_x and positive_x:
        neg_avg = statistics.mean([r['stats']['avg'] for r in negative_x])
        pos_avg = statistics.mean([r['stats']['avg'] for r in positive_x])
        
        print(f"\nNegative X coordinate impact:")
        print(f"  Negative X average: {neg_avg:.2f}ms")
        print(f"  Positive X average: {pos_avg:.2f}ms")
        print(f"  Difference: {neg_avg - pos_avg:.2f}ms")
    
    # Find optimal game capture settings
    game_sized = [r for r in results if r['size'] in ['160x192', '320x240']]
    if game_sized:
        optimal = min(game_sized, key=lambda x: x['stats']['avg'])
        print(f"\nOptimal game capture:")
        print(f"  Monitor {optimal['monitor']}: {optimal['test']}")
        print(f"  Position: ({optimal['region']['left']}, {optimal['region']['top']})")
        print(f"  Speed: {optimal['stats']['avg']:.2f}ms ({1000/optimal['stats']['avg']:.0f} FPS)")

def test_continuous_at_position(region, duration=3):
    """Test continuous capture at a specific position."""
    print(f"\nTesting continuous capture for {duration} seconds...")
    print(f"Region: {region}")
    
    frame_count = 0
    start_time = time.time()
    timings = []
    
    with mss.mss() as sct:
        while time.time() - start_time < duration:
            frame_start = time.perf_counter()
            img = sct.grab(region)
            frame_time = (time.perf_counter() - frame_start) * 1000
            timings.append(frame_time)
            frame_count += 1
    
    elapsed = time.time() - start_time
    fps = frame_count / elapsed
    
    print(f"Results:")
    print(f"  Frames: {frame_count}")
    print(f"  FPS: {fps:.1f}")
    print(f"  Average: {statistics.mean(timings):.2f}ms")
    print(f"  Median: {statistics.median(timings):.2f}ms")
    
    return fps

def main():
    print("Comprehensive Screenshot Capture Test")
    print("="*70)
    print("This will test capture speeds from all monitors and positions\n")
    
    # Test all monitors and positions
    results = test_all_monitors()
    
    # Analyze results
    analyze_results(results)
    
    # Test continuous capture on primary monitor
    print("\n" + "="*70)
    print("CONTINUOUS CAPTURE TEST (Primary Monitor)")
    print("="*70)
    
    with mss.mss() as sct:
        primary = sct.monitors[1]  # Primary monitor
        test_region = {
            "left": primary['left'] + 100,
            "top": primary['top'] + 100,
            "width": 160,
            "height": 192
        }
        
        fps = test_continuous_at_position(test_region)
        
        if fps >= 60:
            print(f"\n✓ Primary monitor can achieve 60+ FPS!")
            print(f"Consider moving BizHawk to your primary monitor")
        else:
            print(f"\n✗ Even primary monitor is slow ({fps:.1f} FPS)")
            print(f"This suggests a system-wide issue with MSS")
    
    print("\n" + "="*70)
    print("RECOMMENDATIONS:")
    print("="*70)
    print("1. Use the monitor/position with fastest capture times")
    print("2. If all captures are ~16ms, you're hitting display refresh limits")
    print("3. Consider alternative capture methods (dxcam, Windows Graphics Capture)")
    print("4. For BizHawk, investigate Lua scripting for direct frame access")

if __name__ == "__main__":
    main()