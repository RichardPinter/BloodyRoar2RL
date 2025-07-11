#!/usr/bin/env python3
"""
Test Arcade Training Script

Comprehensive testing of RL training in arcade mode where agent must
win 8 matches in a row to complete the arcade.
"""

import torch
import numpy as np
import time
import os
from datetime import datetime
from collections import defaultdict, deque
import json

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.arcade_rl_environment import ArcadeRLEnvironment
from core.rl_training_simple import PPOAgent


class ArcadeTrainingTester:
    """Test harness for arcade mode training"""
    
    def __init__(self, num_episodes: int = 50):  # More episodes for arcade mode
        self.num_episodes = num_episodes
        self.env = ArcadeRLEnvironment()
        self.agent = PPOAgent(
            state_dim=self.env.get_observation_space_size(),
            action_dim=self.env.get_action_space_size()
        )
        
        # Tracking metrics
        self.episode_rewards = []
        self.episode_lengths = []
        self.action_counts = defaultdict(int)
        
        # Arcade-specific tracking
        self.arcade_completions = 0
        self.best_arcade_run = 0  # Most matches won in one arcade
        self.current_arcade_wins = 0
        
        # Running statistics
        self.recent_rewards = deque(maxlen=20)
        
        # Create logs directory
        self.log_dir = "training_logs"
        os.makedirs(self.log_dir, exist_ok=True)
        self.log_file = os.path.join(
            self.log_dir, 
            f"arcade_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        
    def run_test(self):
        """Run comprehensive test of arcade training"""
        print("="*80)
        print("🕹️  ARCADE TRAINING TEST")
        print("="*80)
        print(f"Testing {self.num_episodes} episodes across multiple arcade attempts")
        print(f"State dimension: {self.env.get_observation_space_size()} (includes arcade context)")
        print(f"Action space: {self.env.get_action_space_size()} actions - {self.env.get_actions()}")
        print("="*80)
        
        start_time = time.time()
        
        for episode in range(self.num_episodes):
            self.run_episode(episode)
            self.print_episode_summary(episode)
            
            # Update policy every 5 episodes if we have enough data
            if len(self.agent.buffer.states) >= 50:
                print("\n📈 Updating policy...")
                policy_loss, value_loss = self.agent.update()
                print(f"   Policy Loss: {policy_loss:.4f}, Value Loss: {value_loss:.4f}")
        
        # Final summary
        self.print_final_summary()
        self.save_results()
        
        total_time = time.time() - start_time
        print(f"\nTotal test time: {total_time:.1f} seconds")
        
    def run_episode(self, episode_num: int):
        """Run a single episode (round) with arcade context"""
        print(f"\n{'='*60}")
        print(f"📺 EPISODE {episode_num + 1}/{self.num_episodes}")
        print(f"{'='*60}")
        
        # Reset environment
        state = self.env.reset()
        
        # Extract arcade context from state
        arcade_match_num = int(state[19])
        arcade_wins = int(state[20])
        is_final_match = bool(state[21])
        
        # Update tracking
        if arcade_wins == 0 and arcade_match_num == 1:
            # New arcade started
            if self.current_arcade_wins > self.best_arcade_run:
                self.best_arcade_run = self.current_arcade_wins
            self.current_arcade_wins = 0
        
        print(f"🕹️  Arcade Match {arcade_match_num} - Current wins: {arcade_wins}/8")
        if is_final_match:
            print(f"   🔥 FINAL MATCH OF ARCADE!")
        
        episode_reward = 0
        episode_steps = 0
        
        # Get initial health
        initial_p1_health = state[1]
        initial_p2_health = state[4]
        
        print(f"🎬 Round started - P1: {initial_p1_health:.1f}% vs P2: {initial_p2_health:.1f}%")
        
        done = False
        while not done:
            # Select action
            action, log_prob, value = self.agent.select_action(state)
            action_name = self.env.get_actions()[action]
            self.action_counts[action_name] += 1
            
            # Environment step
            next_state, reward, done, info = self.env.step(action)
            
            # Store experience
            self.agent.buffer.add(state, action, reward, next_state, done, log_prob, value)
            
            episode_reward += reward
            episode_steps += 1
            state = next_state
            
            # Check for significant events
            if info.get('match_completed', False):
                match_winner = info.get('match_winner')
                print(f"\n  🏁 Match completed! Winner: {match_winner}")
                print(f"  📊 Arcade progress: {info['arcade_wins']}-{info['arcade_losses']}")
                
                if match_winner == 'PLAYER 1':
                    self.current_arcade_wins += 1
                
                if info.get('arcade_completed', False):
                    self.arcade_completions += 1
                    print(f"\n  🎉🎉 ARCADE COMPLETED! Total completions: {self.arcade_completions}")
                elif not info.get('arcade_active', True):
                    print(f"\n  💀 Arcade ended. Best run: {self.current_arcade_wins} matches")
            
            if episode_steps % 10 == 0:
                print(f"  Step {episode_steps}: Action={action_name}, Reward={reward:.3f}")
        
        # Store episode data
        self.episode_rewards.append(episode_reward)
        self.episode_lengths.append(episode_steps)
        self.recent_rewards.append(episode_reward)
        
    def print_episode_summary(self, episode_num: int):
        """Print summary after each episode"""
        reward = self.episode_rewards[-1]
        length = self.episode_lengths[-1]
        
        print(f"\n  📊 Episode Summary:")
        print(f"     Total Reward: {reward:.3f}")
        print(f"     Episode Length: {length} steps")
        print(f"     Arcade Completions: {self.arcade_completions}")
        print(f"     Best Arcade Run: {self.best_arcade_run} matches")
        
        if len(self.recent_rewards) > 1:
            avg_recent = np.mean(list(self.recent_rewards))
            print(f"     Recent Avg Reward: {avg_recent:.3f}")
            
    def print_final_summary(self):
        """Print comprehensive summary after all episodes"""
        print("\n" + "="*80)
        print("📊 FINAL ARCADE TRAINING SUMMARY")
        print("="*80)
        
        # Overall statistics
        avg_reward = np.mean(self.episode_rewards)
        avg_length = np.mean(self.episode_lengths)
        
        print(f"\n🎯 Performance Metrics:")
        print(f"   Average Reward: {avg_reward:.3f}")
        print(f"   Average Episode Length: {avg_length:.1f} steps")
        
        print(f"\n🕹️  Arcade Statistics:")
        print(f"   Total Arcade Completions: {self.arcade_completions}")
        print(f"   Best Arcade Run: {self.best_arcade_run} matches won")
        print(f"   Arcade Success Rate: {self.env._get_success_rate():.1f}%")
        
        # Reward trend
        if len(self.episode_rewards) > 10:
            first_quarter = self.episode_rewards[:len(self.episode_rewards)//4]
            last_quarter = self.episode_rewards[3*len(self.episode_rewards)//4:]
            
            first_avg = np.mean(first_quarter)
            last_avg = np.mean(last_quarter)
            improvement = last_avg - first_avg
            
            print(f"\n📈 Learning Progress:")
            print(f"   First Quarter Avg: {first_avg:.3f}")
            print(f"   Last Quarter Avg: {last_avg:.3f}")
            print(f"   Improvement: {improvement:+.3f} {'✅' if improvement > 0 else '❌'}")
        
        # Action distribution
        print(f"\n🎮 Action Distribution:")
        total_actions = sum(self.action_counts.values())
        for action, count in sorted(self.action_counts.items()):
            percentage = count / total_actions * 100
            print(f"   {action}: {count} ({percentage:.1f}%)")
        
        # Print environment's arcade summary
        print("\n")
        self.env.print_arcade_summary()
        
    def save_results(self):
        """Save detailed results to file"""
        results = {
            'summary': {
                'num_episodes': self.num_episodes,
                'avg_reward': float(np.mean(self.episode_rewards)),
                'avg_length': float(np.mean(self.episode_lengths)),
                'arcade_completions': self.arcade_completions,
                'best_arcade_run': self.best_arcade_run,
                'arcade_attempts': self.env.arcade_attempts,
                'arcade_success_rate': self.env._get_success_rate(),
                'action_distribution': dict(self.action_counts)
            },
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'arcade_history': self.env.arcade_history
        }
        
        with open(self.log_file, 'w') as f:
            json.dump(results, f, indent=2)
            
        print(f"\n💾 Results saved to: {self.log_file}")


def main():
    """Run the arcade training test"""
    # Test with 50 episodes for more thorough arcade testing
    tester = ArcadeTrainingTester(num_episodes=50)
    
    try:
        tester.run_test()
    except KeyboardInterrupt:
        print("\n⏹️  Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Error during test: {e}")
        import traceback
        traceback.print_exc()
    finally:
        tester.env.close()
        print("\n✅ Test complete!")


if __name__ == "__main__":
    main()