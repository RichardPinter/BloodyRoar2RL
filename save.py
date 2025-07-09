import cv2
import numpy as np
from mss import mss
import win32gui
import time
from ultralytics import YOLO

class CompleteGameMonitor:
    def __init__(self, window_title):
        self.window_title = window_title
        self.sct = mss()
        
        # YOLO for fighter detection
        print("Loading YOLO model...")
        self.model = YOLO('yolov8n.pt')  # Will download first time
        
        # Health bar parameters (from your working code)
        self.health_params = {
            'p1_x': 505,
            'p2_x': 1421,
            'bar_len': 400,
            'y': 155,
            'lower_bgr': np.array([0, 160, 190], dtype=np.uint8),
            'upper_bgr': np.array([20, 180, 220], dtype=np.uint8),
            'drop_per_px': 0.25
        }
        
        # Get window handle
        self.hwnd = win32gui.FindWindow(None, self.window_title)
        if not self.hwnd:
            raise RuntimeError(f"Window not found: {self.window_title}")
            
        # Window dimensions
        rect = win32gui.GetClientRect(self.hwnd)
        self.left, self.top = win32gui.ClientToScreen(self.hwnd, (0, 0))
        self.width = rect[2]
        self.height = rect[3]
        
        # Position tracking for stability
        self.last_p1_pos = None
        self.last_p2_pos = None
        
    def detect_health(self):
        """Detect health bars using your existing method."""
        # ROIs for health bars
        roi_p1 = {
            'left': self.left + self.health_params['p1_x'],
            'top': self.top + self.health_params['y'],
            'width': self.health_params['bar_len'],
            'height': 1,
        }
        
        roi_p2 = {
            'left': self.left + (self.health_params['p2_x'] - self.health_params['bar_len']),
            'top': self.top + self.health_params['y'],
            'width': self.health_params['bar_len'],
            'height': 1,
        }
        
        # Capture health bars
        raw_p1 = self.sct.grab(roi_p1)
        strip_p1 = np.array(raw_p1)[:, :, :3]
        b1, g1, r1 = strip_p1[0].T
        
        raw_p2 = self.sct.grab(roi_p2)
        strip_p2 = np.array(raw_p2)[:, :, :3]
        b2, g2, r2 = strip_p2[0].T
        
        # Process Player 1 (left to right)
        mask_p1 = (
            (r1 >= self.health_params['lower_bgr'][2]) & 
            (r1 <= self.health_params['upper_bgr'][2]) &
            (g1 >= self.health_params['lower_bgr'][1]) & 
            (g1 <= self.health_params['upper_bgr'][1]) &
            (b1 >= self.health_params['lower_bgr'][0]) & 
            (b1 <= self.health_params['upper_bgr'][0])
        )
        non_yellow_p1 = np.nonzero(~mask_p1)[0]
        last_idx_p1 = non_yellow_p1.max() if non_yellow_p1.size else -1
        drop_pixels_p1 = max(0, last_idx_p1 + 1)
        life_pct_p1 = 100.0 - (drop_pixels_p1 * self.health_params['drop_per_px'])
        life_pct_p1 = np.clip(life_pct_p1, 0.0, 100.0)
        
        # Process Player 2 (right to left)
        mask_p2 = (
            (r2 >= self.health_params['lower_bgr'][2]) & 
            (r2 <= self.health_params['upper_bgr'][2]) &
            (g2 >= self.health_params['lower_bgr'][1]) & 
            (g2 <= self.health_params['upper_bgr'][1]) &
            (b2 >= self.health_params['lower_bgr'][0]) & 
            (b2 <= self.health_params['upper_bgr'][0])
        )
        non_yellow_p2 = np.nonzero(~mask_p2)[0]
        last_idx_p2 = non_yellow_p2.min() if non_yellow_p2.size else self.health_params['bar_len']
        drop_pixels_p2 = self.health_params['bar_len'] - last_idx_p2
        life_pct_p2 = 100.0 - (drop_pixels_p2 * self.health_params['drop_per_px'])
        life_pct_p2 = np.clip(life_pct_p2, 0.0, 100.0)
        
        return life_pct_p1, life_pct_p2
    
    def detect_fighters(self, image):
        """Detect fighter positions using YOLO."""
        # Run YOLO detection
        results = self.model(image, classes=[0], conf=0.3, verbose=False)  # class 0 = person
        
        fighters = []
        for r in results:
            if r.boxes is not None:
                for box in r.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    center_x = int((x1 + x2) / 2)
                    center_y = int((y1 + y2) / 2)
                    conf = box.conf[0].item()
                    
                    # Skip if too high (might be UI elements)
                    if center_y > 100:  # Below health bars
                        fighters.append({
                            'center': (center_x, center_y),
                            'box': (int(x1), int(y1), int(x2), int(y2)),
                            'conf': conf
                        })
        
        # Sort by x position to identify P1 (left) and P2 (right)
        fighters.sort(key=lambda f: f['center'][0])
        
        # Assign to players
        p1_pos = None
        p2_pos = None
        
        if len(fighters) >= 2:
            p1_pos = fighters[0]['center']
            p2_pos = fighters[1]['center']
        elif len(fighters) == 1:
            # Use last known positions to determine which player
            pos = fighters[0]['center']
            if self.last_p1_pos and self.last_p2_pos:
                dist_to_p1 = abs(pos[0] - self.last_p1_pos[0])
                dist_to_p2 = abs(pos[0] - self.last_p2_pos[0])
                if dist_to_p1 < dist_to_p2:
                    p1_pos = pos
                else:
                    p2_pos = pos
            else:
                # Guess based on position
                if pos[0] < self.width // 2:
                    p1_pos = pos
                else:
                    p2_pos = pos
        
        # Update last known positions
        if p1_pos:
            self.last_p1_pos = p1_pos
        if p2_pos:
            self.last_p2_pos = p2_pos
        
        return {
            'p1': p1_pos,
            'p2': p2_pos,
            'distance': abs(p1_pos[0] - p2_pos[0]) if p1_pos and p2_pos else None,
            'all_detections': fighters
        }
    
    def get_game_state(self):
        """Get complete game state including positions and health."""
        # Capture full screen
        monitor = {
            'left': self.left,
            'top': self.top,
            'width': self.width,
            'height': self.height
        }
        
        screenshot = np.array(self.sct.grab(monitor))
        image = cv2.cvtColor(screenshot, cv2.COLOR_BGRA2BGR)
        
        # Get fighter positions
        fighter_data = self.detect_fighters(image)
        
        # Get health
        p1_health, p2_health = self.detect_health()
        
        # Combine all data
        game_state = {
            'p1': {
                'position': fighter_data['p1'],
                'health': p1_health
            },
            'p2': {
                'position': fighter_data['p2'],
                'health': p2_health
            },
            'distance': fighter_data['distance'],
            'frame': image,
            'detections': fighter_data['all_detections']
        }
        
        return game_state
    
    def visualize_state(self, show_window=True):
        """Run the monitor with visualization."""
        cv2.namedWindow('Game Monitor', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Game Monitor', self.width // 2, self.height // 2)
        
        fps = 30
        interval = 1.0 / fps
        
        print("Game monitor running. Press 'q' to quit\n")
        
        try:
            while True:
                t0 = time.perf_counter()
                
                # Get game state
                state = self.get_game_state()
                
                # Draw visualization
                display = state['frame'].copy()
                
                # Draw fighter positions
                if state['p1']['position']:
                    x, y = state['p1']['position']
                    cv2.circle(display, (x, y), 10, (0, 255, 0), -1)
                    cv2.putText(display, f"P1: {state['p1']['health']:.1f}%", 
                               (x-50, y-30), cv2.FONT_HERSHEY_SIMPLEX, 
                               0.7, (0, 255, 0), 2)
                
                if state['p2']['position']:
                    x, y = state['p2']['position']
                    cv2.circle(display, (x, y), 10, (0, 0, 255), -1)
                    cv2.putText(display, f"P2: {state['p2']['health']:.1f}%", 
                               (x-50, y-30), cv2.FONT_HERSHEY_SIMPLEX, 
                               0.7, (0, 0, 255), 2)
                
                # Draw health bars with outlines
                # P1 health bar
                cv2.rectangle(display, 
                             (self.health_params['p1_x'], self.health_params['y'] - 5),
                             (self.health_params['p1_x'] + self.health_params['bar_len'], 
                              self.health_params['y'] + 5),
                             (0, 255, 0), 2)
                
                # P2 health bar
                cv2.rectangle(display, 
                             (self.health_params['p2_x'] - self.health_params['bar_len'], 
                              self.health_params['y'] - 5),
                             (self.health_params['p2_x'], self.health_params['y'] + 5),
                             (0, 0, 255), 2)
                
                # Draw distance
                if state['distance']:
                    cv2.putText(display, f"Distance: {state['distance']}px", 
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                               1, (255, 255, 0), 2)
                
                # Draw all YOLO detections
                for det in state['detections']:
                    x1, y1, x2, y2 = det['box']
                    cv2.rectangle(display, (x1, y1), (x2, y2), (128, 128, 128), 1)
                    cv2.putText(display, f"{det['conf']:.2f}", 
                               (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 
                               0.4, (128, 128, 128), 1)
                
                # Print to console
                p1_str = f"P1: pos={state['p1']['position']}, hp={state['p1']['health']:.1f}%"
                p2_str = f"P2: pos={state['p2']['position']}, hp={state['p2']['health']:.1f}%"
                dist_str = f"Dist: {state['distance']}px" if state['distance'] else "Dist: ---"
                print(f"\r{p1_str} | {p2_str} | {dist_str}", end='', flush=True)
                
                # Show window
                if show_window:
                    cv2.imshow('Game Monitor', display)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                
                # Maintain FPS
                dt = time.perf_counter() - t0
                if dt < interval:
                    time.sleep(interval - dt)
                    
        except KeyboardInterrupt:
            print("\nStopped.")
        finally:
            cv2.destroyAllWindows()


# Usage for RL environment
class GameEnvironment:
    def __init__(self, window_title):
        self.monitor = CompleteGameMonitor(window_title)
        
    def get_observation(self):
        """Get current game state for RL agent."""
        state = self.monitor.get_game_state()
        
        # Format for RL (example)
        obs = {
            'p1_x': state['p1']['position'][0] if state['p1']['position'] else 0,
            'p1_y': state['p1']['position'][1] if state['p1']['position'] else 0,
            'p1_health': state['p1']['health'],
            'p2_x': state['p2']['position'][0] if state['p2']['position'] else 0,
            'p2_y': state['p2']['position'][1] if state['p2']['position'] else 0,
            'p2_health': state['p2']['health'],
            'distance': state['distance'] if state['distance'] else 0
        }
        
        return obs


if __name__ == "__main__":
    WINDOW_TITLE = "Bloody Roar II (USA) [PlayStation] - BizHawk"
    
    # Run with visualization
    monitor = CompleteGameMonitor(WINDOW_TITLE)
    monitor.visualize_state()
    
    # Or use for RL
    # env = GameEnvironment(WINDOW_TITLE)
    # while True:
    #     obs = env.get_observation()
    #     print(obs)
    #     time.sleep(0.033)  # 30 FPS






# test_all_controls_complete.py
import time
import os

class BizHawkController:
    def __init__(self):
        self.actions_file = r"C:\Users\richa\Desktop\Personal\Uni\ShenLong\actions.txt"
        self.action_delay = 0.3  # 300ms between actions
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(self.actions_file), exist_ok=True)
        
        # Clear any old actions
        self.clear_file()
        print(f"Controller ready. Writing to: {self.actions_file}")
        
    def clear_file(self):
        """Clear the actions file"""
        try:
            with open(self.actions_file, 'w') as f:
                f.write("")
        except Exception as e:
            print(f"Error clearing file: {e}")
    
    def send_action(self, action):
        """Write action and wait for it to be processed"""
        try:
            with open(self.actions_file, 'w') as f:
                f.write(action)
            print(f"Sent: {action}")
            time.sleep(self.action_delay)  # Give Lua time to read and clear
        except Exception as e:
            print(f"Error sending action: {e}")
    
    # Basic actions
    def kick(self):
        self.send_action("kick")
    
    def punch(self):
        self.send_action("punch")
    
    def move_right(self):
        self.send_action("right")
    
    def move_left(self):
        self.send_action("left")
    
    def move_up(self):
        self.send_action("up")
    
    def move_down(self):
        self.send_action("down")
    
    def stop(self):
        self.send_action("stop")
    
    # Additional button methods
    def circle(self):
        self.send_action("circle")
    
    def x(self):
        self.send_action("x")
    
    def square(self):
        self.send_action("square")
    
    def triangle(self):
        self.send_action("triangle")
    
    def l1(self):
        self.send_action("l1")
    
    def l2(self):
        self.send_action("l2")
    
    def r1(self):
        self.send_action("r1")
    
    def r2(self):
        self.send_action("r2")
    
    def start(self):
        self.send_action("start")
    
    def select(self):
        self.send_action("select")
    
    # Hold actions
    def hold_right(self, frames=60):
        self.send_action(f"right:{frames}")
    
    def hold_left(self, frames=60):
        self.send_action(f"left:{frames}")
    
    def hold_up(self, frames=60):
        self.send_action(f"up:{frames}")
    
    def hold_down(self, frames=60):
        self.send_action(f"down:{frames}")


# Test functions
def test_all_controls():
    controller = BizHawkController()
    
    print("=== TESTING ALL CONTROLS ===")
    print("Watch BizHawk to see which buttons work!")
    print("-" * 40)
    
    # Test each button
    tests = [
        ("Circle (○) - Kick", "circle"),
        ("X - Punch", "x"),
        ("Square (□)", "square"),
        ("Triangle (△)", "triangle"),
        ("D-Pad Right", "right"),
        ("D-Pad Left", "left"),
        ("D-Pad Up", "up"),
        ("D-Pad Down", "down"),
        ("L1", "l1"),
        ("L2", "l2"),
        ("R1", "r1"),
        ("R2", "r2"),
        ("Start", "start"),
        ("Select", "select"),
    ]
    
    for button_name, action in tests:
        print(f"\nTesting {button_name}...")
        controller.send_action(action)
        time.sleep(1)  # Wait 1 second between tests
        controller.send_action("stop")  # Release button
        time.sleep(0.5)
    
    print("\n=== TEST COMPLETE ===")
    print("\nWhich buttons worked? Make note of any that didn't respond.")

def test_movement():
    controller = BizHawkController()
    
    print("\n=== TESTING MOVEMENT ===")
    print("Character should move in a square pattern")
    
    # Move in a square
    movements = [
        ("Moving right...", "right:60"),  # Hold for 60 frames (1 second)
        ("Moving down...", "down:60"),
        ("Moving left...", "left:60"),
        ("Moving up...", "up:60"),
    ]
    
    for description, action in movements:
        print(description)
        controller.send_action(action)
        time.sleep(1.5)
    
    controller.send_action("stop")
    print("Movement test complete!")

def test_combos():
    controller = BizHawkController()
    
    print("\n=== TESTING COMBOS ===")
    
    # Test some fighting game combos
    combos = [
        {
            "name": "Basic Punch Combo",
            "moves": ["x", "x", "x"],
            "delay": 0.3
        },
        {
            "name": "Kick Combo",
            "moves": ["circle", "circle", "circle"],
            "delay": 0.3
        },
        {
            "name": "Mixed Combo",
            "moves": ["x", "circle", "square"],
            "delay": 0.3
        },
        {
            "name": "Movement + Attack",
            "moves": ["right:30", "x", "right:30", "circle"],
            "delay": 0.5
        }
    ]
    
    for combo in combos:
        print(f"\nTesting: {combo['name']}")
        for move in combo['moves']:
            controller.send_action(move)
            time.sleep(combo['delay'])
        
        controller.send_action("stop")
        time.sleep(1)

def interactive_test():
    controller = BizHawkController()
    
    print("\n=== INTERACTIVE TEST MODE ===")
    print("Type commands to test them directly")
    print("Available commands: circle, x, square, triangle, up, down, left, right")
    print("                   l1, l2, r1, r2, start, select, stop")
    print("Hold commands: right:60 (holds right for 60 frames)")
    print("Type 'quit' to exit")
    print("-" * 40)
    
    while True:
        command = input("\nEnter command: ").strip().lower()
        
        if command == 'quit':
            break
        elif command == '':
            continue
        else:
            controller.send_action(command)

def debug_movement():
    controller = BizHawkController()
    
    print("=== MOVEMENT DEBUG ===")
    print("This will test each direction individually")
    print("Watch both the Lua console and the game")
    print("-" * 40)
    
    # Test each direction with different methods
    directions = [
        ("right", "Simple right"),
        ("left", "Simple left"),
        ("up", "Simple up"),
        ("down", "Simple down"),
        ("right:120", "Hold right for 2 seconds"),
        ("left:120", "Hold left for 2 seconds"),
    ]
    
    for command, description in directions:
        print(f"\nTesting: {description}")
        print(f"Sending: '{command}'")
        controller.send_action(command)
        
        input("Press Enter to continue to next test...")
        
        # Make sure to stop between tests
        controller.send_action("stop")
        time.sleep(0.5)
    
    print("\nDebug complete!")
    
    # Check what's in the file
    print("\nChecking actions file...")
    try:
        with open(controller.actions_file, 'r') as f:
            content = f.read()
            print(f"Current file content: '{content}'")
    except Exception as e:
        print(f"Error reading file: {e}")

# Main program
if __name__ == "__main__":
    print("BizHawk Complete Control Test")
    print("=" * 50)
    print("Make sure:")
    print("1. BizHawk is running with your game")
    print("2. The updated Lua script is running")
    print("3. You see 'Ready for Python commands...' in Lua console")
    print("=" * 50)
    
    while True:
        print("\nSelect test:")
        print("1. Test all buttons one by one")
        print("2. Test movement patterns")
        print("3. Test combat combos")
        print("4. Interactive mode (type commands)")
        print("5. Debug movement")
        print("6. Simple test (kick only)")
        print("7. Exit")
        
        choice = input("\nEnter choice (1-7): ").strip()
        
        if choice == '1':
            test_all_controls()
        elif choice == '2':
            test_movement()
        elif choice == '3':
            test_combos()
        elif choice == '4':
            interactive_test()
        elif choice == '5':
            debug_movement()
        elif choice == '6':
            controller = BizHawkController()
            print("\nTesting single kick...")
            controller.kick()
            print("If you saw the kick, it's working!")
        elif choice == '7':
            break
        else:
            print("Invalid choice, try again")
    
    print("\nTest session complete!")