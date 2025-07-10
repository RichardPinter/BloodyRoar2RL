import time
import numpy as np
from round_sub_episode import RoundSubEpisode, RoundOutcome

def test_round_win_detection():
    """
    Test the integrated win detection in RoundSubEpisode
    """
    
    print("Testing RoundSubEpisode with Integrated Win Detection")
    print("=" * 60)
    print("This will test the pixel-based win detection built into RoundSubEpisode")
    print("Press Ctrl+C to stop")
    print("-" * 60)
    
    try:
        # Initialize RoundSubEpisode
        round_env = RoundSubEpisode()
        
        # Reset for new round
        obs = round_env.reset()
        print(f"Round started. Observation shape: {obs.shape}")
        print(f"Win detection available: {round_env.health_detection_available}")
        
        step_count = 0
        done = False
        
        print("\nStarting round with random actions...")
        print("Health % | Zero Frames | Step | Action | Reward | Done | Outcome")
        print("-" * 70)
        
        while not done and step_count < 1000:  # Max 1000 steps for testing
            # Take random action
            action = np.random.randint(0, round_env.env.action_space.n)
            
            # Step the environment
            obs, reward, done, info = round_env.step(action)
            step_count += 1
            
            # Extract win detection info
            p1_health = info.get('p1_health_percentage', 0.0)
            p2_health = info.get('p2_health_percentage', 0.0)
            p1_zeros = info.get('p1_zero_frames', 0)
            p2_zeros = info.get('p2_zero_frames', 0)
            outcome = info.get('round_outcome', 'ongoing')
            
            # Print every 10 steps or when significant events occur
            if (step_count % 10 == 0 or 
                abs(reward) > 0.01 or 
                p1_zeros > 0 or p2_zeros > 0 or 
                done):
                
                print(f"P1:{p1_health:5.1f}% P2:{p2_health:5.1f}% | "
                      f"P1:{p1_zeros:2d} P2:{p2_zeros:2d} | "
                      f"{step_count:4d} | {action:6d} | {reward:6.2f} | "
                      f"{str(done):5s} | {outcome}")
            
            # Show danger warnings
            zero_threshold = info.get('zero_threshold', 10)
            if p1_zeros >= zero_threshold - 3 and p1_zeros < zero_threshold:
                print(f"    ⚠️  P1 DANGER! {zero_threshold - p1_zeros} frames until death")
            if p2_zeros >= zero_threshold - 3 and p2_zeros < zero_threshold:
                print(f"    ⚠️  P2 DANGER! {zero_threshold - p2_zeros} frames until death")
            
            if done:
                break
                
            # Small delay to avoid overwhelming output
            time.sleep(0.05)
        
        # Show final results
        print("\n" + "=" * 60)
        print("ROUND FINISHED!")
        
        stats = round_env.get_stats()
        if stats:
            print(f"Outcome: {stats.outcome.value}")
            print(f"Duration: {stats.duration:.1f} seconds")
            print(f"Steps: {stats.steps_taken}")
            print(f"Total reward: {stats.total_reward:.2f}")
            print(f"Final health: P1={stats.final_p1_health:.1f}%, P2={stats.final_p2_health:.1f}%")
        
        # Show win detection final state
        print(f"\nWin Detection State:")
        print(f"P1 zero frames: {round_env.p1_zero_frames}")
        print(f"P2 zero frames: {round_env.p2_zero_frames}")
        print(f"Zero threshold: {round_env.zero_threshold}")
        
        if stats and stats.outcome == RoundOutcome.PLAYER_WIN:
            print("🎉 PLAYER 1 WINS!")
        elif stats and stats.outcome == RoundOutcome.PLAYER_LOSS:
            print("💀 PLAYER 1 LOSES!")
        elif stats and stats.outcome == RoundOutcome.DRAW:
            print("🤝 DRAW!")
        elif stats and stats.outcome == RoundOutcome.TIMEOUT:
            print("⏰ TIMEOUT!")
        
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
    except Exception as e:
        print(f"\nError during test: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if 'round_env' in locals():
            round_env.close()
        print("\nTest completed")

def test_multiple_rounds():
    """Test multiple consecutive rounds"""
    
    print("Testing Multiple Consecutive Rounds")
    print("=" * 50)
    
    try:
        round_env = RoundSubEpisode()
        
        for round_num in range(3):
            print(f"\n--- ROUND {round_num + 1} ---")
            
            obs = round_env.reset()
            done = False
            step_count = 0
            
            while not done and step_count < 100:  # Shorter test rounds
                action = np.random.randint(0, round_env.env.action_space.n)
                obs, reward, done, info = round_env.step(action)
                step_count += 1
                
                if step_count % 20 == 0 or done:
                    p1_health = info.get('p1_health_percentage', 0.0)
                    p2_health = info.get('p2_health_percentage', 0.0)
                    print(f"  Step {step_count}: P1={p1_health:.1f}%, P2={p2_health:.1f}%, Done={done}")
            
            # Show round result
            stats = round_env.get_stats()
            if stats:
                print(f"  Result: {stats.outcome.value} in {stats.steps_taken} steps")
                
    except Exception as e:
        print(f"Error: {e}")
    finally:
        if 'round_env' in locals():
            round_env.close()

if __name__ == "__main__":
    print("RoundSubEpisode Win Detection Tests")
    print("=" * 50)
    print("1. Single round test with detailed output")
    print("2. Multiple rounds test")
    print("3. Exit")
    
    choice = input("\nEnter choice (1-3): ").strip()
    
    if choice == '1':
        test_round_win_detection()
    elif choice == '2':
        test_multiple_rounds()
    elif choice == '3':
        print("Exiting...")
    else:
        print("Invalid choice, running single round test...")
        test_round_win_detection()