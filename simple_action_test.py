#!/usr/bin/env python3
"""
Simple Action Timing Test

Type actions like kick, punch, etc. and watch health values to see timing.
"""

import time
import msvcrt
from window_capture import WindowCapture
from health_detector import HealthDetector
from game_controller import BizHawkController

def main():
    print("Simple Action Timing Test")
    print("Type: kick, punch, throw, special, block")
    print("Press 'q' to quit")
    print("-" * 40)
    
    # Initialize
    capture = WindowCapture("Bloody Roar II (USA) [PlayStation] - BizHawk")
    health_detector = HealthDetector()
    controller = BizHawkController()
    
    command = ""
    frame = 0
    last_p2_health = 100.0
    action_frame = None
    action_name = None
    
    while True:
        frame += 1
        
        # Get health
        health_state = health_detector.detect(capture)
        if health_state:
            p1_health = health_state.p1_health
            p2_health = health_state.p2_health
        else:
            p1_health = p2_health = 0.0
        
        # Check if P2 health decreased
        if p2_health < last_p2_health and action_frame is not None:
            frames_to_hit = frame - action_frame
            print(f"\n>>> P2 HIT! {action_name} took {frames_to_hit} frames to connect <<<")
            action_frame = None  # Reset
        
        # Print health
        print(f"\rFrame {frame:04d} | P1: {p1_health:5.1f}% | P2: {p2_health:5.1f}% | {command}", end='', flush=True)
        
        # Update last health
        last_p2_health = p2_health
        
        # Check for keyboard input (non-blocking)
        if msvcrt.kbhit():
            key = msvcrt.getch().decode('utf-8', errors='ignore').lower()
            
            if key == '\r':  # Enter pressed
                # Execute command
                print()  # New line
                if command == "kick":
                    controller.kick()
                    print(f"*** KICK at frame {frame} ***")
                    action_frame = frame
                    action_name = "KICK"
                elif command == "punch":
                    controller.punch()
                    print(f"*** PUNCH at frame {frame} ***")
                    action_frame = frame
                    action_name = "PUNCH"
                elif command == "throw":
                    controller.throw()
                    print(f"*** THROW at frame {frame} ***")
                    action_frame = frame
                    action_name = "THROW"
                elif command == "special":
                    controller.special()
                    print(f"*** SPECIAL at frame {frame} ***")
                    action_frame = frame
                    action_name = "SPECIAL"
                elif command == "block":
                    controller.block()
                    print(f"*** BLOCK at frame {frame} ***")
                    action_frame = frame
                    action_name = "BLOCK"
                command = ""
            elif key == 'q':
                break
            elif key == '\x08':  # Backspace
                command = command[:-1]
            else:
                command += key
        
        time.sleep(0.05)  # ~20 FPS

if __name__ == "__main__":
    main()