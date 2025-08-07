#!/usr/bin/env python3
"""
Real-time Debug Monitor for your RL Game Agent
Run this in a separate terminal to monitor the game state in real-time
"""
import time
import os
from collections import deque
from datetime import datetime

# Configuration
ACTIONS_FILE = "actions.txt"
LOG_FILE = "logs/game_debug_*.log"  # Will find the latest
HEALTH_CSV = "health_results.csv"

class GameMonitor:
    def __init__(self):
        self.last_action = None
        self.last_action_time = time.time()
        self.action_history = deque(maxlen=20)
        self.state = "Unknown"
        self.match_info = {"match": 0, "p1_rounds": 0, "p2_rounds": 0}
        self.health = {"p1": 0.0, "p2": 0.0}
        
        # Snapshot logging
        self.snapshot_file = f"monitor_snapshots_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        self.last_snapshot_time = 0
        self.snapshot_interval = 30  # Save snapshot every 30 seconds
        self.error_count = 0
        
    def save_snapshot(self):
        """Save a snapshot of current state to file"""
        current_time = time.time()
        if current_time - self.last_snapshot_time < self.snapshot_interval:
            return
            
        self.last_snapshot_time = current_time
        
        with open(self.snapshot_file, 'a') as f:
            f.write(f"\n{'='*60}\n")
            f.write(f"SNAPSHOT: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"State: {self.state}\n")
            f.write(f"Match: #{self.match_info['match']} | P1:{self.match_info['p1_rounds']} P2:{self.match_info['p2_rounds']}\n")
            f.write(f"Health: P1={self.health['p1']:.1f}% P2={self.health['p2']:.1f}%\n")
            f.write(f"Current Action: {self.last_action} (age: {time.time() - self.last_action_time:.1f}s)\n")
            
            # Log any issues
            if self.state == "POST_MATCH_WAITING" and (time.time() - self.last_action_time) > 0.5:
                f.write("⚠️  WARNING: Actions not updating during POST_MATCH_WAITING!\n")
            
            # Last few actions
            f.write("Recent actions: ")
            recent = list(self.action_history)[-5:]
            f.write(", ".join([a[1] for a in recent]) + "\n")
        
    def get_latest_log_file(self):
        """Find the most recent log file"""
        import glob
        files = glob.glob("logs/game_debug_*.log")
        if files:
            return max(files, key=os.path.getctime)
        return None
        
    def read_last_action(self):
        """Read the current action from actions.txt"""
        try:
            with open(ACTIONS_FILE, 'r') as f:
                action = f.read().strip()
                if action != self.last_action:
                    self.last_action = action
                    self.last_action_time = time.time()
                    self.action_history.append((time.time(), action))
                return action
        except:
            return "NO FILE"
    
    def parse_recent_logs(self, log_file, num_lines=100):
        """Parse recent log entries for state info"""
        try:
            with open(log_file, 'r') as f:
                lines = deque(f, num_lines)
                
            for line in reversed(lines):
                # Check for state changes
                if "Entering post-match navigation mode" in line:
                    self.state = "POST_MATCH_WAITING"
                elif "ROUND STARTED:" in line:
                    self.state = "ACTIVE"
                elif "NEW ROUND DETECTED:" in line and "Starting Match" in line:
                    self.state = "NEW_MATCH_STARTING"
                    # Extract match number
                    if "Match #" in line:
                        try:
                            match_num = int(line.split("Match #")[1].split()[0])
                            self.match_info["match"] = match_num
                        except:
                            pass
                
                # Check for round info
                if "ROUND CONFIRMED:" in line:
                    if "(P1:" in line and "P2:" in line:
                        try:
                            p1 = int(line.split("(P1:")[1].split()[0])
                            p2 = int(line.split("P2:")[1].split(")")[0])
                            self.match_info["p1_rounds"] = p1
                            self.match_info["p2_rounds"] = p2
                        except:
                            pass
                
                # Check for match end
                if "MATCH #" in line and "OVER:" in line:
                    self.state = "MATCH_ENDED"
                    
        except Exception as e:
            pass
    
    def get_health_from_csv(self):
        """Get latest health values from CSV"""
        try:
            with open(HEALTH_CSV, 'r') as f:
                lines = f.readlines()
                if len(lines) > 1:
                    last_line = lines[-1].strip()
                    parts = last_line.split(',')
                    if len(parts) == 3:
                        self.health["p1"] = float(parts[1])
                        self.health["p2"] = float(parts[2])
        except:
            pass
    
    def display(self):
        """Display current game state"""
        os.system('cls' if os.name == 'nt' else 'clear')
        
        print("="*60)
        print(f" GAME STATE MONITOR - {datetime.now().strftime('%H:%M:%S')}")
        print("="*60)
        
        # Current State
        state_color = {
            "ACTIVE": "\033[92m",  # Green
            "POST_MATCH_WAITING": "\033[93m",  # Yellow
            "MATCH_ENDED": "\033[91m",  # Red
            "NEW_MATCH_STARTING": "\033[94m",  # Blue
            "Unknown": "\033[90m"  # Gray
        }.get(self.state, "\033[0m")
        
        print(f"\nCURRENT STATE: {state_color}{self.state}\033[0m")
        print(f"Match #{self.match_info['match']} | Rounds: P1={self.match_info['p1_rounds']} P2={self.match_info['p2_rounds']}")
        print(f"Health: P1={self.health['p1']:.1f}% | P2={self.health['p2']:.1f}%")
        
        # Current Action
        action_age = time.time() - self.last_action_time
        if action_age > 2.0:
            action_color = "\033[91m"  # Red if stale
        else:
            action_color = "\033[92m"  # Green if fresh
        
        print(f"\nCURRENT ACTION: {action_color}{self.last_action}\033[0m (age: {action_age:.1f}s)")
        
        # Action History
        print("\nACTION HISTORY (last 10):")
        for ts, action in list(self.action_history)[-10:]:
            age = time.time() - ts
            print(f"  {age:6.1f}s ago: {action}")
        
        # State-specific warnings
        print("\nWARNINGS:")
        if self.state == "POST_MATCH_WAITING":
            if action_age > 0.5:
                print("  ⚠️  Actions not updating during POST_MATCH_WAITING!")
            if self.last_action not in ["start", "kick"]:
                print("  ⚠️  Wrong action during POST_MATCH_WAITING! Expected start/kick")
        
        if self.state == "ACTIVE" and action_age > 1.0:
            print("  ⚠️  No recent actions during ACTIVE state!")
        
        # Action Statistics
        if len(self.action_history) > 0:
            action_counts = {}
            for _, action in self.action_history:
                action_counts[action] = action_counts.get(action, 0) + 1
            
            print("\nRECENT ACTION DISTRIBUTION:")
            for action, count in sorted(action_counts.items(), key=lambda x: x[1], reverse=True):
                pct = count / len(self.action_history) * 100
                print(f"  {action:10s}: {count:3d} ({pct:5.1f}%)")
    
    def run(self):
        """Main monitoring loop"""
        print("Starting Game Monitor...")
        print("Looking for log files...")
        
        while True:
            try:
                # Find latest log file
                log_file = self.get_latest_log_file()
                
                if log_file:
                    # Update all information
                    self.read_last_action()
                    self.parse_recent_logs(log_file)
                    self.get_health_from_csv()
                    
                    # Display
                    self.display()
                    
                    # Save periodic snapshot
                    self.save_snapshot()
                else:
                    print("No log files found in logs/ directory")
                
                time.sleep(0.1)  # Update 10 times per second
                
            except KeyboardInterrupt:
                print("\nMonitor stopped.")
                print(f"Snapshots saved to: {self.snapshot_file}")
                
                # Save final summary
                with open(self.snapshot_file, 'a') as f:
                    f.write(f"\n{'='*60}\n")
                    f.write(f"MONITOR SESSION ENDED: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write(f"Total snapshots: {int(self.last_snapshot_time > 0)}\n")
                break
            except Exception as e:
                print(f"Error: {e}")
                time.sleep(1)

if __name__ == "__main__":
    monitor = GameMonitor()
    monitor.run()