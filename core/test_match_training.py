#!/usr/bin/env python3
"""
Test Match Training Script

Comprehensive testing of RL training for complete matches with detailed logging
to verify match-level coordination and round-level training.
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

from core.match_rl_environment import MatchRLEnvironment
from core.rl_training_simple import PPOAgent


class MatchTrainingTester:
    """Test harness for verifying RL training on complete matches"""
    
    def __init__(self, num_episodes: int = 6):  # 6 episodes should cover ~2 matches
        self.num_episodes = num_episodes
        self.env = MatchRLEnvironment()
        self.agent = PPOAgent(
            state_dim=self.env.get_observation_space_size(),
            action_dim=self.env.get_action_space_size()
        )
        
        # Tracking metrics
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_outcomes = []  # 'win', 'loss'
        self.action_counts = defaultdict(int)
        self.step_logs = []
        
        # Match-specific tracking
        self.matches_completed = 0
        self.match_winners = []
        self.rounds_per_match = []
        
        # Running statistics
        self.recent_rewards = deque(maxlen=10)
        self.win_count = 0
        self.loss_count = 0
        
        # Create logs directory
        self.log_dir = "training_logs"
        os.makedirs(self.log_dir, exist_ok=True)
        self.log_file = os.path.join(self.log_dir, f"match_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        
    def run_test(self):
        """Run comprehensive test of match training"""
        print("=" * 80)
        print("🧪 MATCH TRAINING TEST")
        print("==" * 80)
        print(f"Testing {self.num_episodes} episodes (rounds) across multiple matches")
        print(f"State dimension: {self.env.get_observation_space_size()} (includes match context)")
        print(f"Action space: {self.env.get_action_space_size()} actions - {self.env.get_actions()}")
        print("=" * 80)
        
        start_time = time.time()
        
        for episode in range(self.num_episodes):
            self.run_episode(episode)
            self.print_episode_summary(episode)
            
            # Update policy every 2 episodes if we have enough data
            if len(self.agent.buffer.states) >= 20:
                print("\n📈 Updating policy...")
                policy_loss, value_loss = self.agent.update()
                print(f"   Policy Loss: {policy_loss:.4f}, Value Loss: {value_loss:.4f}")
        
        # Final summary
        self.print_final_summary()
        self.save_results()
        
        total_time = time.time() - start_time
        print(f"\nTotal test time: {total_time:.1f} seconds")
        
    def run_episode(self, episode_num: int):
        """Run a single episode (round) with detailed logging"""
        print(f"\n{'='*60}")
        print(f"📺 EPISODE {episode_num + 1}/{self.num_episodes}")
        print(f"{'='*60}")
        
        # Reset environment (new round or new match)
        state = self.env.reset()
        
        # Extract match context from state
        round_num = int(state[11])
        p1_rounds = int(state[12])
        p2_rounds = int(state[13])
        rounds_to_win = int(state[14])
        is_match_point = bool(state[16]) or bool(state[17])
        
        print(f"🎬 Round {round_num} - Match Score: P1={p1_rounds} P2={p2_rounds} (first to {rounds_to_win})")
        if is_match_point:
            print(f"🔥 MATCH POINT! This round could decide the match!")
        
        episode_reward = 0
        episode_steps = 0
        step_data = []
        
        # Get initial health for tracking
        initial_p1_health = state[1]  # p1_health_end
        initial_p2_health = state[4]  # p2_health_end
        
        done = False
        while not done:
            step_start = time.time()
            
            # Select action
            action, log_prob, value = self.agent.select_action(state)
            action_name = self.env.get_actions()[action]
            self.action_counts[action_name] += 1
            
            # Log pre-action state
            pre_health_p1 = state[1]
            pre_health_p2 = state[4]
            
            print(f"\n  Step {episode_steps + 1}:")
            print(f"    🎮 Action: {action_name} (index {action})")
            
            # Environment step
            next_state, reward, done, info = self.env.step(action)
            
            # Log post-action state
            post_health_p1 = next_state[1]
            post_health_p2 = next_state[4]
            
            # Calculate health changes
            p1_health_change = post_health_p1 - pre_health_p1
            p2_health_change = post_health_p2 - pre_health_p2
            
            # Detailed step logging
            print(f"    💔 Health: P1: {pre_health_p1:.1f}% → {post_health_p1:.1f}% ({p1_health_change:+.1f})")
            print(f"              P2: {pre_health_p2:.1f}% → {post_health_p2:.1f}% ({p2_health_change:+.1f})")
            print(f"    🎯 Reward: {reward:.3f}")
            print(f"    ⏱️  Step time: {time.time() - step_start:.1f}s")
            
            # Log match context changes
            if info.get('round_completed', False):
                print(f"    🏁 Round finished! Winner: {info['round_winner']}")
                if info.get('match_completed', False):
                    print(f"    🎉 MATCH COMPLETED! Winner: {info['match_winner']}")
                    self.matches_completed += 1
                    self.match_winners.append(info['match_winner'])
                    self.rounds_per_match.append(info['total_rounds_played'])
            
            # Store experience
            self.agent.buffer.add(state, action, reward, next_state, done, log_prob, value)
            
            # Track step data
            step_info = {
                'step': episode_steps,
                'action': action_name,
                'reward': reward,
                'p1_health': post_health_p1,
                'p2_health': post_health_p2,
                'p1_health_change': p1_health_change,
                'p2_health_change': p2_health_change,
                'match_context': {
                    'round': int(next_state[11]),
                    'score': f"{int(next_state[12])}-{int(next_state[13])}",
                    'match_point': bool(next_state[16]) or bool(next_state[17])
                },
                'done': done
            }
            step_data.append(step_info)
            
            episode_reward += reward
            episode_steps += 1
            state = next_state
        
        # Determine episode outcome
        final_p1_health = state[1]
        final_p2_health = state[4]
        
        if final_p1_health <= 0:
            outcome = 'loss'
            self.loss_count += 1
            print(f"\n  ❌ ROUND LOST! P1 knocked out")
        elif final_p2_health <= 0:
            outcome = 'win'
            self.win_count += 1
            print(f"\n  ✅ ROUND WON! P2 knocked out")
        else:
            outcome = 'ongoing'
            print(f"\n  ⏱️  ROUND ONGOING! P1: {final_p1_health:.1f}% P2: {final_p2_health:.1f}%")
        
        # Store episode data
        self.episode_rewards.append(episode_reward)
        self.episode_lengths.append(episode_steps)
        self.episode_outcomes.append(outcome)
        self.recent_rewards.append(episode_reward)
        
        # Store detailed log
        episode_log = {
            'episode': episode_num,
            'reward': episode_reward,
            'length': episode_steps,
            'outcome': outcome,
            'initial_health': {'p1': initial_p1_health, 'p2': initial_p2_health},
            'final_health': {'p1': final_p1_health, 'p2': final_p2_health},
            'match_context': {
                'round': round_num,
                'initial_score': f"{p1_rounds}-{p2_rounds}",
                'was_match_point': is_match_point
            },
            'steps': step_data
        }
        self.step_logs.append(episode_log)
        
    def print_episode_summary(self, episode_num: int):
        """Print summary after each episode"""
        reward = self.episode_rewards[-1]
        length = self.episode_lengths[-1]
        outcome = self.episode_outcomes[-1]
        
        print(f"\n  📊 Episode Summary:")
        print(f"     Total Reward: {reward:.3f}")
        print(f"     Episode Length: {length} steps")
        print(f"     Outcome: {outcome.upper()}")
        
        if len(self.recent_rewards) > 1:
            avg_recent = np.mean(list(self.recent_rewards))
            print(f"     Recent Avg Reward: {avg_recent:.3f}")
            
    def print_final_summary(self):
        """Print comprehensive summary after all episodes"""
        print("\n" + "=" * 80)
        print("📊 FINAL MATCH TRAINING SUMMARY")
        print("=" * 80)
        
        # Overall statistics
        avg_reward = np.mean(self.episode_rewards)
        avg_length = np.mean(self.episode_lengths)
        
        print(f"\n🎯 Episode (Round) Performance:")
        print(f"   Average Reward: {avg_reward:.3f}")
        print(f"   Average Episode Length: {avg_length:.1f} steps")
        print(f"   Round Win Rate: {self.win_count}/{self.num_episodes} ({self.win_count/self.num_episodes*100:.1f}%)")
        print(f"   Round Loss Rate: {self.loss_count}/{self.num_episodes} ({self.loss_count/self.num_episodes*100:.1f}%)")
        
        # Match-level statistics
        print(f"\n🏆 Match Performance:")
        print(f"   Matches Completed: {self.matches_completed}")
        if self.match_winners:
            p1_match_wins = sum(1 for w in self.match_winners if w == 'PLAYER 1')
            print(f"   P1 Match Wins: {p1_match_wins}/{self.matches_completed} ({p1_match_wins/self.matches_completed*100:.1f}%)")
            avg_rounds = np.mean(self.rounds_per_match) if self.rounds_per_match else 0
            print(f"   Average Rounds per Match: {avg_rounds:.1f}")
        
        # Reward trend
        if len(self.episode_rewards) > 1:
            first_half_avg = np.mean(self.episode_rewards[:len(self.episode_rewards)//2])
            second_half_avg = np.mean(self.episode_rewards[len(self.episode_rewards)//2:])
            improvement = second_half_avg - first_half_avg
            
            print(f"\n📈 Learning Progress:")
            print(f"   First Half Avg: {first_half_avg:.3f}")
            print(f"   Second Half Avg: {second_half_avg:.3f}")
            print(f"   Improvement: {improvement:+.3f} {'✅' if improvement > 0 else '❌'}")
        
        # Action distribution
        print(f"\n🎮 Action Distribution:")
        total_actions = sum(self.action_counts.values())
        for action, count in sorted(self.action_counts.items()):
            percentage = count / total_actions * 100
            print(f"   {action}: {count} ({percentage:.1f}%)")
        
        # Print environment summary
        self.env.print_match_summary()
            
    def save_results(self):
        """Save detailed results to file"""
        
        # Helper function to convert numpy types to python types for JSON serialization
        def convert_numpy_types(obj):
            if isinstance(obj, np.float32):
                return float(obj)
            elif isinstance(obj, np.int32):
                return int(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            elif isinstance(obj, dict):
                return {k: convert_numpy_types(v) for k, v in obj.items()}
            return obj
        
        results = {
            'summary': {
                'num_episodes': self.num_episodes,
                'avg_reward': float(np.mean(self.episode_rewards)),
                'avg_length': float(np.mean(self.episode_lengths)),
                'round_win_rate': self.win_count / self.num_episodes,
                'round_loss_rate': self.loss_count / self.num_episodes,
                'matches_completed': self.matches_completed,
                'match_winners': self.match_winners,
                'action_distribution': dict(self.action_counts)
            },
            'episodes': convert_numpy_types(self.step_logs),
            'match_results': convert_numpy_types(self.env.match_results)
        }
        
        with open(self.log_file, 'w') as f:
            json.dump(results, f, indent=2)
            
        print(f"\n💾 Results saved to: {self.log_file}")


def main():
    """Run the match training test"""
    # Test with 6 episodes to cover multiple rounds and potentially 2+ matches
    tester = MatchTrainingTester(num_episodes=6)
    
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