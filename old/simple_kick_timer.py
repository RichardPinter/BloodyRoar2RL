#!/usr/bin/env python3
"""
Simple Kick Timer

Just sends kicks at configurable intervals with timestamps.
No health detection - pure timing focus.
"""

import time
from game_controller import BizHawkController

# === EASY TIMING CONTROL ===
KICK_INTERVAL = 0.5  # Seconds between kicks (CHANGE THIS TO TEST)
# Try: 0.2 (fast), 0.3 (medium-fast), 0.5 (medium), 0.7 (slow), 1.0 (very slow)

def main():
    print("⚡ Simple Kick Timer")
    print("=" * 40)
    print(f"Kick interval: {KICK_INTERVAL} seconds")
    print("Press Ctrl+C to stop")
    print("-" * 40)
    
    # Initialize controller
    try:
        controller = BizHawkController()
        print("✅ Controller initialized")
    except Exception as e:
        print(f"❌ Controller failed: {e}")
        return
    
    # Test one kick immediately
    print("🧪 Sending test kick...")
    try:
        controller.kick()
        print("✅ Test kick sent")
    except Exception as e:
        print(f"❌ Test kick failed: {e}")
        return
    
    print(f"\n🚀 Starting automatic kicks every {KICK_INTERVAL}s...")
    print("Format: [elapsed_time] KICK #N")
    print("-" * 40)
    
    # Timing variables
    start_time = time.time()
    next_kick_time = start_time + KICK_INTERVAL
    kicks_sent = 0
    
    try:
        while True:
            current_time = time.time()
            elapsed = current_time - start_time
            
            # Check if it's time for next kick
            if current_time >= next_kick_time:
                kicks_sent += 1
                
                # Send kick
                try:
                    controller.kick()
                    print(f"[{elapsed:7.3f}s] KICK #{kicks_sent:3d} ✅")
                except Exception as e:
                    print(f"[{elapsed:7.3f}s] KICK #{kicks_sent:3d} ❌ Error: {e}")
                
                # Schedule next kick
                next_kick_time = current_time + KICK_INTERVAL
            
            # Small sleep to prevent CPU spinning
            time.sleep(0.01)  # 10ms
    
    except KeyboardInterrupt:
        elapsed = time.time() - start_time
        
        print(f"\n\n⏹️  Timer stopped")
        print("-" * 40)
        print(f"Duration: {elapsed:.1f} seconds")
        print(f"Total kicks sent: {kicks_sent}")
        print(f"Average frequency: {kicks_sent/elapsed:.2f} kicks/second")
        print(f"Configured interval: {KICK_INTERVAL}s")
        print(f"Expected kicks: {int(elapsed/KICK_INTERVAL)}")
        
        # Timing analysis
        actual_interval = elapsed / kicks_sent if kicks_sent > 0 else 0
        print(f"Actual average interval: {actual_interval:.3f}s")
        
        if kicks_sent > 0:
            timing_accuracy = (KICK_INTERVAL / actual_interval) * 100
            print(f"Timing accuracy: {timing_accuracy:.1f}%")
            
            print(f"\n💡 To test different speeds:")
            print(f"   Faster: KICK_INTERVAL = {KICK_INTERVAL/2:.1f}")
            print(f"   Slower: KICK_INTERVAL = {KICK_INTERVAL*2:.1f}")
    
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()


def show_timing_examples():
    """Show examples of different timing settings"""
    print("Timing Examples for RL Training:")
    print("=" * 40)
    
    examples = [
        (0.1, "Very fast - 10 actions/sec"),
        (0.2, "Fast - 5 actions/sec"),  
        (0.3, "Medium-fast - 3.3 actions/sec"),
        (0.5, "Medium - 2 actions/sec"),
        (0.7, "Medium-slow - 1.4 actions/sec"),
        (1.0, "Slow - 1 action/sec"),
        (1.5, "Very slow - 0.7 actions/sec"),
    ]
    
    for interval, description in examples:
        ms = interval * 1000
        print(f"{interval:4.1f}s ({ms:4.0f}ms) - {description}")
    
    print(f"\nCurrent setting: {KICK_INTERVAL}s ({KICK_INTERVAL*1000:.0f}ms)")
    print(f"\nFor RL training, good ranges:")
    print(f"  • Fighting games: 0.2-0.5s (allows action execution)")
    print(f"  • Real-time feel: 0.3-0.4s (responsive but not rushed)")
    print(f"  • Training speed: 0.5-1.0s (stable learning)")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "examples":
        show_timing_examples()
    else:
        main()