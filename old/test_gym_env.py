# test_gym_env.py
import numpy as np
import time
from br2_env import BR2Environment

def test_environment():
    """Test the BR2 Gym environment"""
    
    print("=" * 60)
    print("BLOODY ROAR 2 GYM ENVIRONMENT TEST")
    print("=" * 60)
    
    # Initialize environment
    try:
        env = BR2Environment()
        print("✓ Environment initialized successfully")
    except Exception as e:
        print(f"✗ Failed to initialize environment: {e}")
        return
    
    # Test action meanings
    print(f"\nAction Space: {env.action_space.n} discrete actions")
    action_meanings = env.get_action_meanings()
    for i, action in enumerate(action_meanings):
        print(f"  {i}: {action}")
    
    print(f"\nObservation Space: {env.observation_space}")
    
    # Reset environment
    print("\n" + "-" * 40)
    print("RESETTING ENVIRONMENT")
    print("-" * 40)
    
    try:
        initial_obs = env.reset()
        print(f"✓ Environment reset successfully")
        print(f"Initial observation shape: {initial_obs.shape}")
        print(f"Initial observation range: [{initial_obs.min():.3f}, {initial_obs.max():.3f}]")
        
        if env.current_state:
            print(f"Initial state:")
            print(f"  P1 Health: {env.current_state.player1.health:.1f}%")
            print(f"  P2 Health: {env.current_state.player2.health:.1f}%")
            print(f"  Distance: {env.current_state.distance:.0f}")
    except Exception as e:
        print(f"✗ Failed to reset environment: {e}")
        return
    
    # Test stepping through environment
    print("\n" + "-" * 40)
    print("TESTING ENVIRONMENT STEPS")
    print("-" * 40)
    
    total_reward = 0
    step_count = 0
    
    # Test different actions - focus on fighting moves
    test_actions = [0, 5, 6, 7, 8, 10, 11, 12]  # none, punch, kick, heavy_punch, heavy_kick, grab, jump_punch, jump_kick
    
    for action in test_actions:
        try:
            print(f"\nStep {step_count + 1}: Executing action {action} ({action_meanings[action]})")
            
            # Store previous state for comparison
            prev_health_p1 = env.current_state.player1.health if env.current_state else 0
            prev_health_p2 = env.current_state.player2.health if env.current_state else 0
            
            # Take step
            obs, reward, done, info = env.step(action)
            
            step_count += 1
            total_reward += reward
            
            # Print results
            print(f"  Observation shape: {obs.shape}")
            print(f"  Reward: {reward:.3f}")
            print(f"  Done: {done}")
            
            if 'p1_health' in info and 'p2_health' in info:
                health_change_p1 = info['p1_health'] - prev_health_p1
                health_change_p2 = info['p2_health'] - prev_health_p2
                
                print(f"  P1 Health: {prev_health_p1:.1f}% → {info['p1_health']:.1f}% (Δ{health_change_p1:+.1f})")
                print(f"  P2 Health: {prev_health_p2:.1f}% → {info['p2_health']:.1f}% (Δ{health_change_p2:+.1f})")
                print(f"  Distance: {info.get('distance', 0):.0f}")
                
                # Explain reward calculation
                if step_count > 1:  # Skip first step (no previous state)
                    expected_reward = health_change_p1 - health_change_p2
                    print(f"  Reward calculation: {health_change_p1:.1f} - ({health_change_p2:.1f}) = {expected_reward:.1f}")
            
            if done:
                print(f"  Episode ended after {step_count} steps")
                break
                
        except Exception as e:
            print(f"  ✗ Error during step: {e}")
            break
    
    print(f"\nTotal steps: {step_count}")
    print(f"Total reward: {total_reward:.3f}")
    print(f"Average reward per step: {total_reward/max(1,step_count):.3f}")
    
    # Test random episode
    print("\n" + "-" * 40)
    print("TESTING RANDOM EPISODE (10 steps)")
    print("-" * 40)
    
    try:
        obs = env.reset()
        episode_reward = 0
        
        for step in range(10):
            action = env.action_space.sample()  # Random action
            obs, reward, done, info = env.step(action)
            episode_reward += reward
            
            print(f"Step {step+1}: Action={action_meanings[action]:<8} "
                  f"Reward={reward:+6.2f} "
                  f"P1={info.get('p1_health', 0):5.1f}% "
                  f"P2={info.get('p2_health', 0):5.1f}%")
            
            if done:
                print(f"Episode ended early at step {step+1}")
                break
        
        print(f"Episode total reward: {episode_reward:.3f}")
        
    except Exception as e:
        print(f"✗ Error during random episode: {e}")
    
    # Test observation vector interpretation
    print("\n" + "-" * 40)
    print("OBSERVATION VECTOR BREAKDOWN")
    print("-" * 40)
    
    if env.current_state:
        obs = env.monitor.get_normalized_observation(player_perspective=1)
        if obs is not None:
            labels = [
                "Agent Health", "Agent X", "Agent Y", "Agent VelX", "Agent VelY",
                "Opp Health", "Opp RelX", "Opp RelY", "Opp VelX", "Opp VelY", 
                "Distance", "Facing"
            ]
            
            print("Observation vector values:")
            for i, (label, value) in enumerate(zip(labels, obs)):
                print(f"  [{i:2d}] {label:<12}: {value:+7.3f}")
    
    # Clean up
    print("\n" + "-" * 40)
    print("CLEANUP")
    print("-" * 40)
    
    try:
        env.close()
        print("✓ Environment closed successfully")
    except Exception as e:
        print(f"✗ Error closing environment: {e}")
    
    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    test_environment()