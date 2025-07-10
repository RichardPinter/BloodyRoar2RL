#!/usr/bin/env python3
"""
Test Round Training Script

Comprehensive testing of RL training for single rounds with detailed logging
to verify all components are working correctly.
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

from core.slow_rl_environment import SlowRLEnvironment
from core.rl_training_simple import PPOAgent


class RoundTrainingTester:
    """Test harness for verifying RL training on single rounds"""
    
    def __init__(self, num_episodes: int = 5):
        self.num_episodes = num_episodes
        self.env = SlowRLEnvironment()
        self.agent = PPOAgent(
            state_dim=self.env.get_observation_space_size(),
            action_dim=self.env.get_action_space_size()
        )
        
        # Tracking metrics
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_outcomes = []  # 'win', 'loss', 'timeout'
        self.action_counts = defaultdict(int)
        self.step_logs = []
        self.diagnostic_issues = []
        
        # Running statistics
        self.recent_rewards = deque(maxlen=10)
        self.win_count = 0
        self.loss_count = 0
        self.timeout_count = 0
        
        # Create logs directory
        self.log_dir = "training_logs"
        os.makedirs(self.log_dir, exist_ok=True)
        self.log_file = os.path.join(self.log_dir, f"round_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        
    def run_test(self):
        """Run comprehensive test of round training"""
        print("=" * 80)
        print("🧪 ROUND TRAINING TEST")
        print("=" * 80)
        print(f"Testing {self.num_episodes} episodes")
        print(f"State dimension: {self.env.get_observation_space_size()}")
        print(f"Action space: {self.env.get_action_space_size()} actions - {self.env.actions}")
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
        self.run_diagnostics()
        
        total_time = time.time() - start_time
        print(f"\nTotal test time: {total_time:.1f} seconds")
        
    def run_episode(self, episode_num: int):
        """Run a single episode with detailed logging"""
        print(f"\n{'='*60}")
        print(f"📺 EPISODE {episode_num + 1}/{self.num_episodes}")
        print(f"{'='*60}")
        
        # Reset environment
        state = self.env.reset()
        self.verify_state("initial", state)
        
        episode_reward = 0
        episode_steps = 0
        step_data = []
        
        # Get initial health for tracking
        initial_p1_health = state[1]  # p1_health_end
        initial_p2_health = state[4]  # p2_health_end
        
        print(f"🎬 Round started - P1: {initial_p1_health:.1f}% vs P2: {initial_p2_health:.1f}%")
        
        done = False
        while not done:
            step_start = time.time()
            
            # Select action
            action, log_prob, value = self.agent.select_action(state)
            action_name = self.env.actions[action]
            self.action_counts[action_name] += 1
            
            # Log pre-action state
            pre_health_p1 = state[1]  # p1_health_end
            pre_health_p2 = state[4]  # p2_health_end
            
            print(f"\n  Step {episode_steps + 1}:")
            print(f"    🎮 Action: {action_name} (index {action})")
            
            # Environment step
            next_state, reward, done, info = self.env.step(action)
            
            # Log post-action state
            post_health_p1 = next_state[1]  # p1_health_end
            post_health_p2 = next_state[4]  # p2_health_end
            
            # Calculate health changes
            p1_health_change = post_health_p1 - pre_health_p1
            p2_health_change = post_health_p2 - pre_health_p2
            
            # Detailed step logging
            print(f"    💔 Health: P1: {pre_health_p1:.1f}% → {post_health_p1:.1f}% ({p1_health_change:+.1f})")
            print(f"              P2: {pre_health_p2:.1f}% → {post_health_p2:.1f}% ({p2_health_change:+.1f})")
            print(f"    🎯 Reward: {reward:.3f}")
            print(f"    ⏱️  Step time: {time.time() - step_start:.1f}s")
            
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
                'done': done
            }
            step_data.append(step_info)
            
            episode_reward += reward
            episode_steps += 1
            state = next_state
            
            # Check for issues
            if episode_steps > 30:
                self.diagnostic_issues.append(f"Episode {episode_num}: Very long episode ({episode_steps} steps)")
            
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
            outcome = 'timeout'
            self.timeout_count += 1
            print(f"\n  ⏱️  ROUND TIMEOUT! P1: {final_p1_health:.1f}% P2: {final_p2_health:.1f}%")
        
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
            'steps': step_data
        }
        self.step_logs.append(episode_log)
        
    def verify_state(self, state_name: str, state: np.ndarray):
        """Verify state vector is valid"""
        expected_size = self.env.get_observation_space_size()
        if state.shape[0] != expected_size:
            self.diagnostic_issues.append(f"{state_name} state has wrong shape: {state.shape} (expected {expected_size})")
        
        if np.any(np.isnan(state)):
            self.diagnostic_issues.append(f"{state_name} state contains NaN values")
            
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
        print("📊 FINAL TRAINING SUMMARY")
        print("=" * 80)
        
        # Overall statistics
        avg_reward = np.mean(self.episode_rewards)
        avg_length = np.mean(self.episode_lengths)
        
        print(f"\n🎯 Performance Metrics:")
        print(f"   Average Reward: {avg_reward:.3f}")
        print(f"   Average Episode Length: {avg_length:.1f} steps")
        print(f"   Win Rate: {self.win_count}/{self.num_episodes} ({self.win_count/self.num_episodes*100:.1f}%)")
        print(f"   Loss Rate: {self.loss_count}/{self.num_episodes} ({self.loss_count/self.num_episodes*100:.1f}%)")
        print(f"   Timeout Rate: {self.timeout_count}/{self.num_episodes} ({self.timeout_count/self.num_episodes*100:.1f}%)")
        
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
            
    def run_diagnostics(self):
        """Run diagnostic checks for common issues"""
        print(f"\n🔍 Diagnostic Check:")
        
        # Check for issues
        if not self.diagnostic_issues:
            print("   ✅ No issues detected")
        else:
            print("   ⚠️  Issues found:")
            for issue in self.diagnostic_issues:
                print(f"      - {issue}")
                
        # Check for common problems
        total_actions = sum(self.action_counts.values())
        for action, count in self.action_counts.items():
            if count / total_actions > 0.8:
                print(f"   ⚠️  Action '{action}' used {count/total_actions*100:.1f}% of time (possible stuck policy)")
                
        if all(r == self.episode_rewards[0] for r in self.episode_rewards):
            print("   ⚠️  All episodes have same reward (possible environment issue)")
            
        if self.win_count == 0 and self.loss_count == 0:
            print("   ⚠️  No wins or losses detected (possible win detection issue)")
            
    def save_results(self):
        """Save detailed results to file"""
        results = {
            'summary': {
                'num_episodes': self.num_episodes,
                'avg_reward': float(np.mean(self.episode_rewards)),
                'avg_length': float(np.mean(self.episode_lengths)),
                'win_rate': self.win_count / self.num_episodes,
                'loss_rate': self.loss_count / self.num_episodes,
                'timeout_rate': self.timeout_count / self.num_episodes,
                'action_distribution': dict(self.action_counts)
            },
            'episodes': self.step_logs,
            'diagnostic_issues': self.diagnostic_issues
        }
        
        with open(self.log_file, 'w') as f:
            json.dump(results, f, indent=2)
            
        print(f"\n💾 Results saved to: {self.log_file}")


def main():
    """Run the round training test"""
    # Test with 5 episodes for quick validation
    tester = RoundTrainingTester(num_episodes=5)
    
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