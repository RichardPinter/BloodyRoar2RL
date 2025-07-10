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
    
    while True:
        frame += 1
        
        # Get health
        health_state = health_detector.detect(capture)
        if health_state:
            p1_health = health_state.p1_health
            p2_health = health_state.p2_health
        else:
            p1_health = p2_health = 0.0
        
        # Print health
        print(f"\rFrame {frame:04d} | P1: {p1_health:5.1f}% | P2: {p2_health:5.1f}% | {command}", end='', flush=True)
        
        # Check for keyboard input (non-blocking)
        if msvcrt.kbhit():
            key = msvcrt.getch().decode('utf-8', errors='ignore').lower()
            
            if key == '\r':  # Enter pressed
                # Execute command
                print()  # New line
                if command == "kick":
                    controller.kick()
                    print(f"*** KICK at frame {frame} ***")
                elif command == "punch":
                    controller.punch()
                    print(f"*** PUNCH at frame {frame} ***")
                elif command == "throw":
                    controller.throw()
                    print(f"*** THROW at frame {frame} ***")
                elif command == "special":
                    controller.special()
                    print(f"*** SPECIAL at frame {frame} ***")
                elif command == "block":
                    controller.block()
                    print(f"*** BLOCK at frame {frame} ***")
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