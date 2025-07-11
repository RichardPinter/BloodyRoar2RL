#!/usr/bin/env python3
"""
Test Arcade Logic Script

Simple test of arcade progression logic without RL training.
Uses random actions to verify match transitions, win/loss detection,
and arcade completion/failure scenarios.
"""

import numpy as np
import time
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.arcade_rl_environment import ArcadeRLEnvironment


class ArcadeLogicTester:
    """Simple tester for arcade logic without training"""
    
    def __init__(self):
        # Use shorter settings for faster testing
        self.env = ArcadeRLEnvironment(
            matches_to_win=3,  # Shorter arcade for testing
            match_transition_delay=2.0  # Shorter delay
        )
        self.episode_count = 0
        
    def test_arcade_logic(self, max_episodes: int = 30):
        """Test arcade logic with random actions"""
        print("🧪 TESTING ARCADE LOGIC")
        print("="*60)
        print("Testing arcade progression without training")
        print(f"Target: Win {self.env.matches_to_win} matches for arcade completion")
        print(f"Max episodes: {max_episodes}")
        print("="*60)
        
        try:
            for episode in range(max_episodes):
                self.run_test_episode(episode)
                
                # Check if we've seen both success and failure scenarios
                if len(self.env.arcade_history) >= 3:
                    successes = sum(1 for h in self.env.arcade_history if h['success'])
                    failures = len(self.env.arcade_history) - successes
                    
                    print(f"\n📊 Testing Progress: {successes} successes, {failures} failures")
                    
                    if successes >= 1 and failures >= 1:
                        print("✅ Tested both success and failure scenarios!")
                        break
                        
        except KeyboardInterrupt:
            print("\n⏹️ Test interrupted by user")
        except Exception as e:
            print(f"\n❌ Error during test: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.print_test_summary()
            self.env.close()
    
    def run_test_episode(self, episode_num: int):
        """Run a single test episode with random actions"""
        self.episode_count += 1
        
        print(f"\n{'='*50}")
        print(f"🎮 TEST EPISODE {self.episode_count}")
        print(f"{'='*50}")
        
        # Reset environment
        state = self.env.reset()
        self.print_state_info("RESET", state)
        
        episode_reward = 0
        step_count = 0
        max_steps_per_episode = 50  # Prevent infinite episodes
        
        done = False
        while not done and step_count < max_steps_per_episode:
            step_count += 1
            
            # Use random action
            action = np.random.randint(0, self.env.get_action_space_size())
            action_name = self.env.get_actions()[action]
            
            print(f"\n  Step {step_count}: Action = {action_name}")
            
            # Take step
            next_state, reward, done, info = self.env.step(action)
            episode_reward += reward
            
            # Print step results
            if reward != 0:
                print(f"    💰 Reward: {reward:+.1f} (Total: {episode_reward:+.1f})")
            
            # Check for important events
            self.check_step_events(info)
            
            # Update state info if arcade context changed
            if self.state_changed(state, next_state):
                self.print_state_info("UPDATE", next_state)
            
            state = next_state
            
            # Prevent runaway episodes
            if step_count >= max_steps_per_episode:
                print(f"    ⏰ Episode ended after {max_steps_per_episode} steps (timeout)")
                break
        
        print(f"\n  📊 Episode {self.episode_count} complete:")
        print(f"      Total reward: {episode_reward:+.1f}")
        print(f"      Steps taken: {step_count}")
    
    def state_changed(self, old_state: np.ndarray, new_state: np.ndarray) -> bool:
        """Check if arcade-relevant state changed"""
        # Check arcade features (last 3 elements)
        old_arcade = old_state[-3:]
        new_arcade = new_state[-3:]
        return not np.array_equal(old_arcade, new_arcade)
    
    def print_state_info(self, label: str, state: np.ndarray):
        """Print current arcade state information"""
        # Extract arcade context (checkpoint system)
        current_opponent = int(state[19])
        total_wins = int(state[20])
        is_final = bool(state[21])
        
        # Extract match context  
        current_round = int(state[11])
        p1_rounds = int(state[12])
        p2_rounds = int(state[13])
        
        # Extract health
        p1_health = state[1]
        p2_health = state[4]
        
        print(f"  🎯 {label} STATE:")
        print(f"      🕹️  Arcade: Opponent {current_opponent}/3, Total wins: {total_wins}")
        if is_final:
            print(f"          🔥 FINAL OPPONENT!")
        print(f"      🥊 Match: Round {current_round}, Score {p1_rounds}-{p2_rounds}")
        print(f"      💚 Health: P1={p1_health:.1f}% P2={p2_health:.1f}%")
    
    def check_step_events(self, info: dict):
        """Check and report important events from step info"""
        
        # Round completion
        if info.get('round_completed', False):
            winner = info.get('round_winner', 'Unknown')
            print(f"    🏁 Round finished! Winner: {winner}")
        
        # Match completion  
        if info.get('match_completed', False):
            winner = info.get('match_winner', 'Unknown')
            current_opponent = info.get('current_opponent', 0)
            total_wins = info.get('arcade_wins', 0)
            print(f"    🎖️  MATCH COMPLETED! Winner: {winner}")
            print(f"        vs Opponent {current_opponent}, Total wins: {total_wins}")
            
            # Check for arcade events
            if info.get('arcade_completed', False):
                print(f"    🎉🎉 ARCADE COMPLETED! Beat all opponents!")
            elif winner != 'PLAYER 1':
                print(f"    🔄 AUTO-RESTART: Will face Opponent {current_opponent} again")
                
    def print_test_summary(self):
        """Print summary of all testing"""
        print("\n" + "="*60)
        print("📋 ARCADE LOGIC TEST SUMMARY")
        print("="*60)
        
        print(f"\nTest Episodes Run: {self.episode_count}")
        print(f"Arcade Attempts: {self.env.arcade_attempts}")
        
        if self.env.arcade_history:
            successes = sum(1 for h in self.env.arcade_history if h['success'])
            failures = len(self.env.arcade_history) - successes
            
            print(f"\nArcade Results:")
            print(f"  ✅ Successful completions: {successes}")
            print(f"  ❌ Failed attempts: {failures}")
            
            print(f"\nArcade History:")
            for i, result in enumerate(self.env.arcade_history, 1):
                status = "✅ SUCCESS" if result['success'] else "❌ FAILED"
                print(f"  Attempt {i}: {status} - {result['matches_won']} matches won")
                
            # Test coverage
            print(f"\n🧪 Test Coverage:")
            if successes > 0:
                print(f"  ✅ Arcade completion logic tested")
            else:
                print(f"  ⚠️  Arcade completion not tested")
                
            if failures > 0:
                print(f"  ✅ Arcade failure logic tested")  
            else:
                print(f"  ⚠️  Arcade failure not tested")
        else:
            print("\n⚠️ No arcade attempts completed")
            
        print("="*60)


def main():
    """Run the arcade logic test"""
    print("🕹️ ARCADE LOGIC TESTER")
    print("Tests arcade progression without RL training")
    print("Uses random actions to trigger various scenarios\n")
    
    tester = ArcadeLogicTester()
    tester.test_arcade_logic(max_episodes=30)
    
    print("\n✅ Arcade logic test complete!")


if __name__ == "__main__":
    main()