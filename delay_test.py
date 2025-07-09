# delay_test.py
import time
import os
from datetime import datetime

class BizHawkController:
    def __init__(self, action_delay=0.3):
        self.actions_file = r"C:\Users\richa\Desktop\Personal\Uni\ShenLong\actions.txt"
        self.action_delay = action_delay  # Configurable delay
        os.makedirs(os.path.dirname(self.actions_file), exist_ok=True)
        self.clear_file()
        
    def clear_file(self):
        try:
            with open(self.actions_file, 'w') as f:
                f.write("")
        except Exception as e:
            print(f"Error clearing file: {e}")
    
    def send_action(self, action, custom_delay=None):
        try:
            with open(self.actions_file, 'w') as f:
                f.write(action)
            # Use custom delay if provided, otherwise use default
            delay = custom_delay if custom_delay is not None else self.action_delay
            time.sleep(delay)
            return True
        except Exception as e:
            print(f"Error sending action: {e}")
            return False

def test_minimum_delay():
    """Find the minimum reliable delay"""
    print("=== TESTING MINIMUM DELAY ===")
    print("This will find the fastest reliable delay setting")
    print("-" * 50)
    
    controller = BizHawkController(action_delay=0)
    
    # Test different delays
    delays = [0.5, 0.4, 0.3, 0.2, 0.15, 0.1, 0.08, 0.05, 0.03, 0.02, 0.01, 0.005, 0]
    
    for delay in delays:
        print(f"\nTesting delay: {delay*1000:.0f}ms")
        
        success_count = 0
        test_count = 10
        
        for i in range(test_count):
            # Alternate between two different actions to ensure change
            action = "x" if i % 2 == 0 else "circle"
            
            start = time.perf_counter()
            controller.send_action(action, custom_delay=delay)
            elapsed = time.perf_counter() - start
            
            # Check if file was cleared (meaning Lua read it)
            time.sleep(0.05)  # Small wait to let Lua process
            try:
                with open(controller.actions_file, 'r') as f:
                    content = f.read()
                    if content == "":
                        success_count += 1
            except:
                pass
        
        success_rate = (success_count / test_count) * 100
        print(f"Success rate: {success_rate}% ({success_count}/{test_count})")
        
        if success_rate < 90:
            print(f">>> Minimum reliable delay: {delays[delays.index(delay)-1]*1000:.0f}ms")
            return delays[delays.index(delay)-1]
    
    print(">>> Even 0ms delay works! (File I/O is the limiting factor)")
    return 0

def test_action_throughput():
    """Test how many actions per second can be processed"""
    print("\n=== TESTING ACTION THROUGHPUT ===")
    print("How many actions can we send per second?")
    print("-" * 50)
    
    test_delays = [0.3, 0.2, 0.1, 0.05, 0.02, 0.01, 0]
    
    for delay in test_delays:
        controller = BizHawkController(action_delay=delay)
        
        print(f"\nTesting with {delay*1000:.0f}ms delay:")
        
        actions = ["x", "circle", "square", "triangle"]
        start_time = time.perf_counter()
        action_count = 0
        
        # Run for 2 seconds
        while time.perf_counter() - start_time < 2.0:
            action = actions[action_count % len(actions)]
            controller.send_action(action)
            action_count += 1
        
        elapsed = time.perf_counter() - start_time
        actions_per_second = action_count / elapsed
        
        print(f"Sent {action_count} actions in {elapsed:.2f} seconds")
        print(f"Rate: {actions_per_second:.1f} actions/second")
        print(f"Average time per action: {(elapsed/action_count)*1000:.1f}ms")

def test_response_time():
    """Measure actual response time including Lua processing"""
    print("\n=== TESTING RESPONSE TIME ===")
    print("Measuring time from Python send to Lua execution")
    print("-" * 50)
    
    controller = BizHawkController(action_delay=0.05)
    
    print("\nInstructions:")
    print("1. Watch the Lua console for 'PUNCH!' messages")
    print("2. This will send 10 punches and measure timing")
    input("\nPress Enter when ready...")
    
    timings = []
    
    for i in range(10):
        # Write timestamp to file with action
        timestamp = str(time.perf_counter())
        action_with_time = f"x|{timestamp}"
        
        start = time.perf_counter()
        
        with open(controller.actions_file, 'w') as f:
            f.write(action_with_time)
        
        # Wait for file to be cleared
        while True:
            time.sleep(0.001)  # 1ms polling
            try:
                with open(controller.actions_file, 'r') as f:
                    if f.read() == "":
                        break
            except:
                pass
            
            if time.perf_counter() - start > 1.0:  # 1 second timeout
                print("Timeout!")
                break
        
        elapsed = time.perf_counter() - start
        timings.append(elapsed * 1000)  # Convert to ms
        
        print(f"Action {i+1}: {elapsed*1000:.1f}ms")
        time.sleep(0.1)  # Small delay between tests
    
    if timings:
        avg_time = sum(timings) / len(timings)
        min_time = min(timings)
        max_time = max(timings)
        
        print(f"\nResults:")
        print(f"Average response time: {avg_time:.1f}ms")
        print(f"Minimum: {min_time:.1f}ms")
        print(f"Maximum: {max_time:.1f}ms")

def test_lua_polling_rate():
    """Help determine optimal Lua polling rate"""
    print("\n=== LUA POLLING RATE TEST ===")
    print("This helps optimize the check_interval in your Lua script")
    print("-" * 50)
    
    controller = BizHawkController(action_delay=0)
    
    print("\nCurrent Lua setting: checking every 10 frames")
    print("At 60 FPS: 10 frames = 166ms between checks")
    print("\nRecommended settings:")
    print("- Fast (every 3 frames): ~50ms latency, higher CPU")
    print("- Balanced (every 5 frames): ~83ms latency, moderate CPU")
    print("- Default (every 10 frames): ~166ms latency, low CPU")
    print("- Slow (every 30 frames): ~500ms latency, minimal CPU")
    
    input("\nPress Enter to test current settings...")
    
    # Send rapid actions to test polling
    print("\nSending 5 rapid actions...")
    for i in range(5):
        start = time.perf_counter()
        controller.send_action("x")
        # Wait for clear
        while True:
            with open(controller.actions_file, 'r') as f:
                if f.read() == "":
                    break
            time.sleep(0.001)
        elapsed = time.perf_counter() - start
        print(f"Action {i+1} processed in: {elapsed*1000:.0f}ms")

def optimize_settings():
    """Run all tests and recommend optimal settings"""
    print("=== OPTIMIZING BIZHAWK CONTROLLER SETTINGS ===")
    print("=" * 50)
    
    # Test 1: Find minimum delay
    print("\n1. Finding minimum delay...")
    min_delay = test_minimum_delay()
    
    # Test 2: Throughput test
    test_action_throughput()
    
    # Test 3: Response time
    test_response_time()
    
    # Test 4: Lua polling
    test_lua_polling_rate()
    
    # Recommendations
    print("\n" + "=" * 50)
    print("RECOMMENDED SETTINGS:")
    print("=" * 50)
    print(f"\nFor Python (action_delay):")
    print(f"  - Minimum safe: {min_delay}s")
    print(f"  - Recommended: {max(min_delay, 0.05)}s")
    print(f"  - Conservative: 0.1s")
    
    print(f"\nFor Lua script (check_interval):")
    print(f"  - Fast response: 3 frames (~50ms)")
    print(f"  - Balanced: 5 frames (~83ms)")
    print(f"  - Low CPU: 10 frames (~166ms)")
    
    print(f"\nExpected total latency:")
    print(f"  - Best case: ~{(max(min_delay, 0.05) + 0.05)*1000:.0f}ms")
    print(f"  - Average: ~{(max(min_delay, 0.05) + 0.083)*1000:.0f}ms")

if __name__ == "__main__":
    while True:
        print("\n=== BIZHAWK DELAY TESTER ===")
        print("1. Test minimum delay")
        print("2. Test action throughput")
        print("3. Test response time")
        print("4. Test Lua polling rate")
        print("5. Run all tests (recommended)")
        print("6. Exit")
        
        choice = input("\nSelect option (1-6): ").strip()
        
        if choice == '1':
            test_minimum_delay()
        elif choice == '2':
            test_action_throughput()
        elif choice == '3':
            test_response_time()
        elif choice == '4':
            test_lua_polling_rate()
        elif choice == '5':
            optimize_settings()
        elif choice == '6':
            break
        else:
            print("Invalid choice")
        
        input("\nPress Enter to continue...")