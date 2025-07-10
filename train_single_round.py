import time
import numpy as np
import random
from br2_env import BR2Environment

def train_single_round():
    """Train the agent for a single round with timer disabled"""
    
    # Initialize environment
    env = BR2Environment()
    
    # Training parameters
    max_steps = 3600  # 60 seconds at 60fps (longer for no timer)
    
    print("Starting single round training...")
    print("Actions available:")
    for i, action in enumerate(env.get_action_meanings()):
        print(f"  {i}: {action}")
    print("-" * 50)
    
    try:
        # Reset environment
        observation = env.reset()
        total_reward = 0
        steps = 0
        
        print(f"\nStarting single round")
        print(f"Initial state - P1 HP: {env.current_state.player1.health:.1f}%, P2 HP: {env.current_state.player2.health:.1f}%")
        
        # Single round loop
        done = False
        while not done and steps < max_steps:
            # Simple action selection (random for now, you can implement your agent logic here)
            action = random.randint(0, env.action_space.n - 1)
            
            # Take action in environment
            next_observation, reward, done, info = env.step(action)
            
            # Track rewards
            total_reward += reward
            
            # Update observation
            observation = next_observation
            steps += 1
            
            # Print step info every 30 steps (0.5 seconds)
            if steps % 30 == 0:
                if 'p1_health' in info and 'p2_health' in info:
                    print(f"  Step {steps}: P1 HP: {info['p1_health']:.1f}%, "
                          f"P2 HP: {info['p2_health']:.1f}%, "
                          f"Action: {info['action_executed']}, "
                          f"Reward this step: {reward:.2f}, "
                          f"Total reward: {total_reward:.2f}")
            
            # Detailed logging for significant events
            if abs(reward) > 0:
                print(f"    >>> Step {steps}: Significant event! Reward: {reward:.2f} "
                      f"(Action: {info['action_executed']})")
        
        # Round finished
        print(f"\nRound finished!")
        print(f"Total steps: {steps}")
        print(f"Total reward: {total_reward:.2f}")
        
        # Check if we have a valid final state
        if env.current_state is not None:
            print(f"Final state - P1 HP: {env.current_state.player1.health:.1f}%, P2 HP: {env.current_state.player2.health:.1f}%")
            
            if env.current_state.player1.health > env.current_state.player2.health:
                print("Result: PLAYER 1 WINS!")
            elif env.current_state.player2.health > env.current_state.player1.health:
                print("Result: PLAYER 2 WINS!")
            else:
                print("Result: DRAW!")
        else:
            print("Final state: Could not capture final game state")
            
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    except Exception as e:
        print(f"\nError during training: {e}")
        import traceback
        traceback.print_exc()
    finally:
        env.close()
        print("\nTraining session ended")

if __name__ == "__main__":
    print("Bloody Roar 2 RL Training - Single Round Mode")
    print("=" * 50)
    print("Make sure:")
    print("1. BizHawk is running with Bloody Roar 2")
    print("2. Game is in versus mode with timer disabled")
    print("3. The Lua script is active")
    print("4. Both characters are selected and round is ready to start")
    print("=" * 50)
    
    input("Press Enter to start single round training...")
    
    train_single_round()