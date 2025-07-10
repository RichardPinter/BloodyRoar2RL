import time
import numpy as np
from br2_env import BR2Environment

def test_controlled_actions():
    """Test specific actions to validate timing and reward attribution"""
    
    env = BR2Environment()
    
    # Test scenarios
    test_scenarios = [
        {
            "name": "Single Punch Test",
            "actions": [5],  # Just punch
            "description": "Test if punch reward is attributed correctly"
        },
        {
            "name": "Single Kick Test", 
            "actions": [6],  # Just kick
            "description": "Test if kick reward is attributed correctly"
        },
        {
            "name": "Movement Test",
            "actions": [1, 2, 1, 2],  # Left, right, left, right
            "description": "Test movement actions (should have minimal/no reward)"
        },
        {
            "name": "Combo Test",
            "actions": [5, 6, 5],  # Punch, kick, punch
            "description": "Test if combo actions are attributed correctly"
        },
        {
            "name": "Transform Test",
            "actions": [8],  # Transform
            "description": "Test transform action timing"
        }
    ]
    
    print("=== TIMING FIX VALIDATION ===")
    print("Testing action-specific delays and reward attribution")
    print("-" * 50)
    
    try:
        for scenario in test_scenarios:
            print(f"\n--- {scenario['name']} ---")
            print(f"Description: {scenario['description']}")
            
            # Reset environment
            observation = env.reset()
            print(f"Initial state - P1: {env.current_state.player1.health:.1f}%, P2: {env.current_state.player2.health:.1f}%")
            
            total_reward = 0
            step_results = []
            
            # Execute each action in the scenario
            for i, action in enumerate(scenario['actions']):
                action_name = env.action_map.get(action, 'unknown')
                expected_delay = env.action_delays.get(action_name, 0.05)
                
                print(f"  Step {i+1}: Executing {action_name} (expected delay: {expected_delay:.2f}s)")
                
                # Record time before action
                start_time = time.time()
                
                # Execute action
                obs, reward, done, info = env.step(action)
                
                # Record time after action
                end_time = time.time()
                actual_delay = end_time - start_time
                
                # Track results
                step_results.append({
                    'action': action_name,
                    'expected_delay': expected_delay,
                    'actual_delay': actual_delay,
                    'reward': reward,
                    'info': info
                })
                
                total_reward += reward
                
                # Print step details
                delay_diff = abs(actual_delay - expected_delay)
                delay_ok = delay_diff < 0.05  # Allow 50ms tolerance
                
                print(f"    → Delay: {actual_delay:.2f}s (expected: {expected_delay:.2f}s) {'✓' if delay_ok else '✗'}")
                print(f"    → Reward: {reward:.2f} (attributed: {'Yes' if info.get('reward_attributed', False) else 'No'})")
                
                if 'p1_health' in info and 'p2_health' in info:
                    print(f"    → Health: P1: {info['p1_health']:.1f}%, P2: {info['p2_health']:.1f}%")
                
                if done:
                    print(f"    → Episode ended")
                    break
            
            # Scenario summary
            print(f"\n  Summary:")
            print(f"    Actions executed: {len(step_results)}")
            print(f"    Total reward: {total_reward:.2f}")
            print(f"    Significant rewards: {sum(1 for r in step_results if abs(r['reward']) > 0.01)}")
            
            # Check reward attribution
            if len(env.reward_attribution) > 0:
                print(f"    Reward attribution entries:")
                for step, attr in env.reward_attribution.items():
                    print(f"      Step {step}: {attr['action']} → {attr['reward']:.2f} (delay: {attr['delay_used']:.2f}s)")
            else:
                print(f"    No significant rewards to attribute")
            
            # Wait between scenarios
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\nValidation interrupted by user")
    except Exception as e:
        print(f"\nError during validation: {e}")
        import traceback
        traceback.print_exc()
    finally:
        env.close()

def test_reward_timing():
    """Test reward timing by executing a single action and monitoring changes"""
    
    env = BR2Environment()
    
    print("\n=== REWARD TIMING TEST ===")
    print("Executing single punch and monitoring reward over time")
    print("-" * 50)
    
    try:
        # Reset environment
        observation = env.reset()
        print(f"Initial state - P1: {env.current_state.player1.health:.1f}%, P2: {env.current_state.player2.health:.1f}%")
        
        # Execute a single punch
        print("\nExecuting punch...")
        obs, reward, done, info = env.step(5)  # Punch action
        
        print(f"Immediate reward: {reward:.2f}")
        print(f"Action delay used: {info.get('action_delay', 'unknown'):.2f}s")
        print(f"Reward attributed: {'Yes' if info.get('reward_attributed', False) else 'No'}")
        
        # Monitor for a few more steps to see if any delayed effects occur
        print("\nMonitoring for delayed effects...")
        for i in range(5):
            obs, reward, done, info = env.step(0)  # No action
            print(f"  Step +{i+1}: Reward {reward:.2f}, P1: {info.get('p1_health', 'N/A'):.1f}%, P2: {info.get('p2_health', 'N/A'):.1f}%")
            
            if done:
                break
        
        # Show final attribution
        if len(env.reward_attribution) > 0:
            print("\nFinal reward attribution:")
            for step, attr in env.reward_attribution.items():
                print(f"  Step {step}: {attr['action']} → {attr['reward']:.2f}")
        
    except Exception as e:
        print(f"Error during reward timing test: {e}")
        import traceback
        traceback.print_exc()
    finally:
        env.close()

if __name__ == "__main__":
    print("Bloody Roar 2 - Timing Fix Validation")
    print("=" * 50)
    print("This will validate the action-specific delays and reward attribution")
    print("Make sure the game is running and ready for combat")
    print("=" * 50)
    
    test_choice = input("Choose test:\n1. Controlled action scenarios\n2. Reward timing test\n3. Both\nEnter choice (1-3): ").strip()
    
    if test_choice == '1':
        test_controlled_actions()
    elif test_choice == '2':
        test_reward_timing()
    elif test_choice == '3':
        test_controlled_actions()
        test_reward_timing()
    else:
        print("Invalid choice, running controlled actions test...")
        test_controlled_actions()