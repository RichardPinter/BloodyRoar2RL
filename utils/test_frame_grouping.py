#!/usr/bin/env python3
"""
Simple 20-Frame Grouping Test

Collects 20 frames of game state and shows:
- Average health for both players
- Final frame coordinates for both players
- Timestamp
"""

import time
from datetime import datetime
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.round_sub_episode import RoundStateMonitor

def collect_20_frames():
    """Collect 20 frames of game state data"""
    print("Collecting 20 frames...")
    
    monitor = RoundStateMonitor()
    frames = []
    
    try:
        # Collect 20 frames
        for i in range(20):
            # Get current game state
            state = monitor.get_current_state()
            frames.append(state)
            
            # Wait one frame (~16.7ms at 60fps)
            time.sleep(1/60)
            
            # Show progress
            if i % 5 == 0:
                print(f"  Frame {i+1}/20...")
        
        return frames
    
    except Exception as e:
        print(f"Error collecting frames: {e}")
        return []
    
    finally:
        monitor.close()

def analyze_frames(frames):
    """Analyze the collected frames"""
    if not frames:
        return None
    
    # Calculate average health
    p1_healths = [f.p1_health for f in frames if f.p1_health is not None]
    p2_healths = [f.p2_health for f in frames if f.p2_health is not None]
    
    p1_avg_health = sum(p1_healths) / len(p1_healths) if p1_healths else 0
    p2_avg_health = sum(p2_healths) / len(p2_healths) if p2_healths else 0
    
    # Get final frame coordinates
    final_frame = frames[-1]
    p1_final_pos = final_frame.p1_position if final_frame.p1_position else (0, 0)
    p2_final_pos = final_frame.p2_position if final_frame.p2_position else (0, 0)
    
    return {
        'p1_avg_health': p1_avg_health,
        'p2_avg_health': p2_avg_health,
        'p1_final_pos': p1_final_pos,
        'p2_final_pos': p2_final_pos
    }

def main():
    """Main function"""
    print("🎮 20-Frame Analysis Test")
    print("=" * 40)
    
    # Collect frames
    frames = collect_20_frames()
    
    if not frames:
        print("❌ Failed to collect frames")
        return
    
    # Analyze frames
    analysis = analyze_frames(frames)
    
    if not analysis:
        print("❌ Failed to analyze frames")
        return
    
    # Print results with timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n[{timestamp}] 20-Frame Analysis:")
    print(f"  P1 Avg Health: {analysis['p1_avg_health']:.1f}%")
    print(f"  P2 Avg Health: {analysis['p2_avg_health']:.1f}%")
    print(f"  Final P1 Position: {analysis['p1_final_pos']}")
    print(f"  Final P2 Position: {analysis['p2_final_pos']}")

if __name__ == "__main__":
    main()