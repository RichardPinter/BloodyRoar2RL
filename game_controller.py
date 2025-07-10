# test_all_controls_complete.py
import time
import os

class BizHawkController:
    def __init__(self):
        self.actions_file = r"C:\Users\richa\Desktop\Personal\Uni\ShenLong\actions.txt"
        self.action_delay = 0.05  # 50ms between actions for faster training
        
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
    
    # Compound fighting actions for RL environment
    def heavy_punch(self):
        """Heavy punch - typically triangle"""
        self.send_action("triangle")
    
    def heavy_kick(self):
        """Heavy kick - typically square"""
        self.send_action("square")
    
    def grab(self):
        """Grab/throw - typically R1"""
        self.send_action("r1")
    
    def jump_punch(self):
        """Jump + punch combo"""
        self.send_action("up+x")
    
    def jump_kick(self):
        """Jump + kick combo"""
        self.send_action("up+circle")
    
    def crouch_punch(self):
        """Crouch + punch combo"""
        self.send_action("down+x")
    
    def crouch_kick(self):
        """Crouch + kick combo"""
        self.send_action("down+circle")
    
    def beast(self):
        """Beast transformation"""
        self.send_action("l1+l2")


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