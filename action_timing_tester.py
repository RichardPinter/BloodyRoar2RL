#!/usr/bin/env python3
"""
Interactive Action Timing Tester

Type commands like kick(), punch(), etc. and see real-time health changes
to understand action-to-reward timing delays.
"""

import time
import threading
from typing import Optional, Dict, Any, Tuple
from collections import deque

from round_sub_episode import RoundStateMonitor
from game_controller import BizHawkController

class ActionTimingTester:
    """Interactive tester to measure action-to-effect delays"""
    
    def __init__(self):
        # Initialize components
        print("Initializing Action Timing Tester...")
        
        self.monitor = RoundStateMonitor()
        self.controller = BizHawkController()
        
        # State tracking
        self.frame_count = 0
        self.running = True
        self.monitoring_thread = None
        
        # Action tracking
        self.last_action = None
        self.last_action_frame = None
        self.last_action_name = None
        
        # Health history for change detection
        self.health_history = deque(maxlen=5)  # Keep last 5 frames
        self.last_p1_health = 100.0
        self.last_p2_health = 100.0
        
        # Available actions
        self.action_map = {
            'punch': lambda: self.controller.punch(),
            'kick': lambda: self.controller.kick(),
            'throw': lambda: self.controller.throw(),
            'special': lambda: self.controller.special(),
            'transform': lambda: self.controller.transform(),
            'left': lambda: self.controller.move_left(),
            'right': lambda: self.controller.move_right(),
            'jump': lambda: self.controller.jump(),
            'squat': lambda: self.controller.squat(),
            'block': lambda: self.controller.block(),
            'stop': lambda: self.controller.send_action('stop'),
        }
        
        print("✅ Tester initialized!")
        print("Available commands:", ", ".join([f"{cmd}()" for cmd in self.action_map.keys()]))
    
    def start_monitoring(self):
        """Start the health monitoring thread"""
        self.monitoring_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitoring_thread.start()
    
    def _monitor_loop(self):
        """Continuous monitoring loop"""
        self.monitor.reset()
        
        while self.running:
            try:
                # Get current state
                state = self.monitor.get_current_state()
                self.frame_count = state.frame_count
                
                # Detect health changes
                health_changed = self._detect_health_change(state.p1_health, state.p2_health)
                
                # Build output string
                output = f"Frame {self.frame_count:04d} | "
                output += f"P1: {state.p1_health:5.1f}% | "
                output += f"P2: {state.p2_health:5.1f}% | "
                
                # Add action timing info
                if self.last_action_frame is not None:
                    frames_since = self.frame_count - self.last_action_frame
                    output += f"[{self.last_action_name}+{frames_since}]"
                    
                    # Highlight if health changed after our action
                    if health_changed and frames_since > 0:
                        output += " ← HIT DETECTED!"
                
                print(output)
                
                # Update health history
                self.last_p1_health = state.p1_health
                self.last_p2_health = state.p2_health
                
                # Small delay for readability
                time.sleep(0.05)  # ~20 FPS display
                
            except Exception as e:
                print(f"Monitor error: {e}")
                time.sleep(0.1)
    
    def _detect_health_change(self, p1_health: float, p2_health: float) -> bool:
        """Detect if health changed from last frame"""
        p1_changed = abs(p1_health - self.last_p1_health) > 0.1
        p2_changed = abs(p2_health - self.last_p2_health) > 0.1
        return p1_changed or p2_changed
    
    def execute_action(self, action_name: str):
        """Execute an action and mark timing"""
        if action_name not in self.action_map:
            print(f"❌ Unknown action: {action_name}")
            return
        
        # Record action timing
        self.last_action_frame = self.frame_count
        self.last_action_name = action_name.upper()
        
        # Execute action
        print(f"\n*** {self.last_action_name} ACTION SENT at Frame {self.frame_count:04d} ***\n")
        self.action_map[action_name]()
    
    def run_interactive(self):
        """Run the interactive command loop"""
        print("\n" + "="*60)
        print("ACTION TIMING TESTER")
        print("="*60)
        print("Health monitoring started. Type commands and press Enter:")
        print("Commands: punch(), kick(), throw(), special(), transform(), etc.")
        print("Type 'quit' to exit")
        print("-"*60 + "\n")
        
        # Start monitoring
        self.start_monitoring()
        
        # Give monitoring a moment to start
        time.sleep(0.5)
        
        # Interactive command loop
        while self.running:
            try:
                # Get user input
                user_input = input("\n> ").strip().lower()
                
                # Check for quit
                if user_input in ['quit', 'exit', 'q']:
                    print("\nExiting...")
                    self.running = False
                    break
                
                # Parse command (remove parentheses)
                if user_input.endswith('()'):
                    action = user_input[:-2]
                    self.execute_action(action)
                elif user_input in self.action_map:
                    self.execute_action(user_input)
                else:
                    print(f"Unknown command: {user_input}")
                    print("Try: punch(), kick(), throw(), etc.")
                
            except KeyboardInterrupt:
                print("\n\nInterrupted by user")
                self.running = False
                break
            except Exception as e:
                print(f"Error: {e}")
        
        # Cleanup
        self.running = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=1.0)
        print("\nTester closed")


def test_single_action(action: str = "punch", duration: float = 3.0):
    """Quick test mode - send one action and monitor for a few seconds"""
    print(f"\n🧪 Testing '{action}' action timing...")
    
    tester = ActionTimingTester()
    tester.start_monitoring()
    
    # Wait for monitoring to stabilize
    time.sleep(1.0)
    
    # Execute the action
    print(f"\nSending {action} in 3... 2... 1...")
    time.sleep(1.0)
    tester.execute_action(action)
    
    # Monitor for specified duration
    print(f"\nMonitoring for {duration} seconds...")
    time.sleep(duration)
    
    tester.running = False
    print("\nTest complete!")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        # Quick test mode
        action = sys.argv[1]
        duration = float(sys.argv[2]) if len(sys.argv) > 2 else 3.0
        test_single_action(action, duration)
    else:
        # Interactive mode
        tester = ActionTimingTester()
        tester.run_interactive()