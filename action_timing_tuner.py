#!/usr/bin/env python3
"""
Action Timing Tuner

Simple script to find the optimal action frequency.
Automatically sends punches at adjustable intervals while showing health changes.
"""

import time
from round_sub_episode import RoundStateMonitor
from game_controller import BizHawkController

# === EASY TIMING CONTROL ===
PUNCH_INTERVAL = 0.5  # Seconds between punches (CHANGE THIS TO TEST)
# Try: 0.2 (fast), 0.5 (medium), 1.0 (slow), 1.5 (very slow)

DISPLAY_EVERY_N_FRAMES = 3  # Show health every 3 frames (reduce spam)

def main():
    print("🥊 Action Timing Tuner")
    print("=" * 50)
    print(f"Punch interval: {PUNCH_INTERVAL} seconds")
    print(f"Health display: Every {DISPLAY_EVERY_N_FRAMES} frames")
    print("Press Ctrl+C to stop")
    print("-" * 50)
    
    # Initialize components
    try:
        round_monitor = RoundStateMonitor()
        controller = BizHawkController()
        print("✅ All components initialized")
        
        # Reset round monitor
        round_monitor.reset()
        
        # Test controller immediately
        print("🧪 Testing controller with one punch...")
        controller.punch()
        print("✅ Test punch sent")
        time.sleep(1)  # Wait to see if it worked
        
    except Exception as e:
        print(f"❌ Initialization failed: {e}")
        return
    
    # Timing variables
    start_time = time.time()
    last_punch_time = start_time - PUNCH_INTERVAL  # Start immediately ready to punch
    frame_count = 0
    punches_sent = 0
    
    # Health tracking
    last_p1_health = 100.0
    last_p2_health = 100.0
    
    print(f"\n[Starting at {time.strftime('%H:%M:%S')}]")
    
    try:
        while True:
            frame_count += 1
            current_time = time.time()
            elapsed = current_time - start_time
            
            # Get current health using the working RoundStateMonitor
            game_state = round_monitor.get_current_state()
            p1_health = game_state.p1_health
            p2_health = game_state.p2_health
            
            # Check if it's time to punch
            time_since_last_punch = current_time - last_punch_time
            if time_since_last_punch >= PUNCH_INTERVAL:
                punches_sent += 1
                print(f"\n[{elapsed:6.3f}s] *** PUNCH #{punches_sent} SENT *** (Frame {frame_count})")
                print(f"  Debug: time_since_last={time_since_last_punch:.3f}s, interval={PUNCH_INTERVAL}s")
                
                try:
                    controller.punch()
                    print("  ✅ controller.punch() completed")
                except Exception as e:
                    print(f"  ❌ controller.punch() failed: {e}")
                
                last_punch_time = current_time
            
            # Display health every N frames (reduce spam)
            if frame_count % DISPLAY_EVERY_N_FRAMES == 0:
                # Check for health changes
                p1_change = p1_health - last_p1_health
                p2_change = p2_health - last_p2_health
                
                # Build status line
                status_line = f"[{elapsed:6.3f}s] Frame {frame_count:04d} | P1: {p1_health:5.1f}% | P2: {p2_health:5.1f}%"
                
                # Add change indicators
                if abs(p1_change) > 0.1:
                    status_line += f" | P1 Δ{p1_change:+.1f}%"
                if abs(p2_change) > 0.1:
                    status_line += f" | P2 Δ{p2_change:+.1f}%"
                
                # Highlight significant health drops
                if p2_change < -1.0:  # P2 took damage
                    status_line += " ← HIT!"
                elif p1_change < -1.0:  # P1 took damage
                    status_line += " ← OUCH!"
                
                print(status_line)
                
                # Update last health for next comparison
                last_p1_health = p1_health
                last_p2_health = p2_health
            
            # Small delay to control frame rate (~60 FPS)
            time.sleep(1/60)
    
    except KeyboardInterrupt:
        elapsed = time.time() - start_time
        total_punches = punches_sent
        
        print(f"\n\n⏹️  Tuning session stopped")
        print(f"Duration: {elapsed:.1f}s")
        print(f"Total punches sent: {total_punches}")
        print(f"Average punch frequency: {total_punches/elapsed:.2f} punches/second")
        print(f"Configured interval: {PUNCH_INTERVAL}s")
        
        if total_punches > 0:
            print(f"\n💡 To test different timing, change PUNCH_INTERVAL:")
            print(f"   - Faster: PUNCH_INTERVAL = {PUNCH_INTERVAL/2:.1f}")
            print(f"   - Slower: PUNCH_INTERVAL = {PUNCH_INTERVAL*2:.1f}")
    
    except Exception as e:
        print(f"\n❌ Error during tuning: {e}")
        import traceback
        traceback.print_exc()


def show_timing_examples():
    """Show examples of different timing settings"""
    print("Timing Examples:")
    print("=" * 30)
    
    examples = [
        (0.1, "Very fast (10 punches/sec) - might be too fast"),
        (0.2, "Fast (5 punches/sec) - good for combos"),  
        (0.3, "Medium-fast (3.3 punches/sec) - responsive"),
        (0.5, "Medium (2 punches/sec) - balanced"),
        (0.7, "Medium-slow (1.4 punches/sec) - deliberate"),
        (1.0, "Slow (1 punch/sec) - strategic"),
        (1.5, "Very slow (0.7 punches/sec) - careful timing"),
    ]
    
    for interval, description in examples:
        print(f"{interval:4.1f}s - {description}")
    
    print(f"\nCurrent setting: {PUNCH_INTERVAL}s")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "examples":
        show_timing_examples()
    else:
        main()