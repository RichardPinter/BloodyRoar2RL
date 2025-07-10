import time
import numpy as np
from br2_env import BR2Environment

def debug_action_timing():
    """Debug the timing of different actions to understand when they take effect"""
    
    env = BR2Environment()
    
    # Test actions with their expected timing
    test_actions = [
        ("punch", 5),
        ("kick", 6), 
        ("throw", 7),
        ("left", 1),
        ("right", 2),
        ("jump", 3),
        ("squat", 4),
        ("transform", 8),
        ("special", 9)
    ]
    
    print("=== ACTION TIMING DEBUG ===")
    print("This will test each action and track when health changes occur")
    print("Watch for the timing between action execution and reward")
    print("-" * 50)
    
    try:
        for action_name, action_id in test_actions:
            print(f"\n--- Testing {action_name.upper()} (ID: {action_id}) ---")
            
            # Reset environment
            observation = env.reset()
            print(f"Initial - P1: {env.current_state.player1.health:.1f}%, P2: {env.current_state.player2.health:.1f}%")
            
            # Track health over time after action
            health_history = []
            reward_history = []
            
            # Execute the action
            start_time = time.time()
            print(f"Executing {action_name} at time 0...")
            
            # Take the action
            obs, reward, done, info = env.step(action_id)
            
            # Track health changes for the next 30 frames (~0.5 seconds)
            for frame in range(30):
                frame_time = time.time() - start_time
                
                # Capture state without taking action
                prev_state = env.current_state
                time.sleep(0.016)  # ~1 frame at 60fps
                env.current_state = env.monitor.capture_state()
                
                if env.current_state is not None and prev_state is not None:
                    # Calculate reward for this frame
                    frame_reward = env.calculate_reward(prev_state, env.current_state)
                    
                    health_history.append({
                        'frame': frame,
                        'time': frame_time,
                        'p1_health': env.current_state.player1.health,
                        'p2_health': env.current_state.player2.health,
                        'reward': frame_reward
                    })
                    
                    # Print significant changes
                    if abs(frame_reward) > 0.1:
                        print(f"  Frame {frame:2d} ({frame_time:.3f}s): Reward {frame_reward:+.2f} - P1: {env.current_state.player1.health:.1f}%, P2: {env.current_state.player2.health:.1f}%")
                
                # Stop if done
                if done:
                    break
            
            # Summary for this action
            total_reward = sum(h['reward'] for h in health_history)
            max_reward_frame = max(health_history, key=lambda h: abs(h['reward']), default={'frame': 0, 'reward': 0})
            
            print(f"  Summary: Total reward = {total_reward:.2f}")
            if abs(max_reward_frame['reward']) > 0.1:
                print(f"  Peak effect at frame {max_reward_frame['frame']} ({max_reward_frame['frame']/60:.2f}s)")
            else:
                print(f"  No significant health changes detected")
            
            # Small delay between tests
            time.sleep(0.5)
            
    except KeyboardInterrupt:
        print("\nTiming debug interrupted by user")
    except Exception as e:
        print(f"\nError during timing debug: {e}")
        import traceback
        traceback.print_exc()
    finally:
        env.close()

if __name__ == "__main__":
    print("Bloody Roar 2 - Action Timing Debug")
    print("=" * 50)
    print("This will help determine the correct timing for each action type")
    print("Make sure the game is running and ready for combat")
    print("=" * 50)
    
    input("Press Enter to start timing debug...")
    debug_action_timing()