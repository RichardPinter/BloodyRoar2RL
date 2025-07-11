#!/usr/bin/env python3
"""
Interactive Arcade Training Script

Train PPO agent in arcade mode with manual stop control and model persistence.
User can stop training at any time and resume later from saved checkpoint.
"""

import torch
import numpy as np
import time
import os
import json
import threading
import sys
from datetime import datetime
from collections import deque
from typing import Optional, Dict, Any

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.arcade_rl_environment import ArcadeRLEnvironment
from core.rl_training_simple import PPOAgent


class InteractiveArcadeTrainer:
    """Interactive training system for arcade mode with manual controls"""
    
    def __init__(self, 
                 arcade_opponents: int = 3,
                 save_interval_episodes: int = 20,
                 models_dir: str = "models"):
        
        self.arcade_opponents = arcade_opponents
        self.save_interval = save_interval_episodes
        self.models_dir = models_dir
        
        # Create models directory
        os.makedirs(models_dir, exist_ok=True)
        
        # Initialize environment and agent
        print("🎮 Initializing Arcade Training Environment...")
        self.env = ArcadeRLEnvironment(matches_to_win=arcade_opponents)
        self.agent = PPOAgent(
            state_dim=self.env.get_observation_space_size(),
            action_dim=self.env.get_action_space_size()
        )
        
        # Training state
        self.episode_count = 0
        self.training_start_time = time.time()
        self.should_stop = False
        self.user_input = ""
        
        # Statistics tracking
        self.episode_rewards = deque(maxlen=100)
        self.arcade_completions = 0
        self.best_arcade_progress = 0  # Best opponent reached
        self.recent_arcade_attempts = deque(maxlen=10)
        
        # Performance tracking
        self.last_save_time = time.time()
        self.best_performance_score = 0.0
        
        print(f"✅ Trainer initialized:")
        print(f"   Arcade: {arcade_opponents} opponents")
        print(f"   State dim: {self.env.get_observation_space_size()}")
        print(f"   Action dim: {self.env.get_action_space_size()}")
        print(f"   Models dir: {models_dir}")
    
    def start_input_thread(self):
        """Start background thread to monitor user input"""
        def input_monitor():
            while not self.should_stop:
                try:
                    user_input = input().strip().lower()
                    if user_input in ['stop', 'quit', 'q', 'exit']:
                        print("\n🛑 Stop command received! Finishing current episode...")
                        self.should_stop = True
                        break
                    elif user_input in ['save', 's']:
                        print("💾 Saving checkpoint...")
                        self.save_checkpoint(reason="manual_save")
                    elif user_input in ['status', 'stat']:
                        self.print_status()
                    elif user_input == 'help':
                        self.print_help()
                except EOFError:
                    break
                except KeyboardInterrupt:
                    print("\n🛑 Keyboard interrupt! Stopping training...")
                    self.should_stop = True
                    break
        
        input_thread = threading.Thread(target=input_monitor, daemon=True)
        input_thread.start()
        return input_thread
    
    def print_help(self):
        """Print available commands"""
        print("\n📋 Available Commands:")
        print("   stop/quit/q  - Stop training and save model")
        print("   save/s       - Save checkpoint without stopping")
        print("   status       - Show current training status")
        print("   help         - Show this help message")
    
    def train(self, resume_from_latest: bool = True):
        """Main training loop with interactive controls"""
        print("\n" + "="*80)
        print("🕹️  INTERACTIVE ARCADE TRAINING")
        print("="*80)
        print("Commands: 'stop' to exit, 'save' to checkpoint, 'status' for info")
        print("Press Ctrl+C or type 'stop' to save and exit")
        print("="*80)
        
        # Try to resume from latest checkpoint
        if resume_from_latest:
            self.load_latest_checkpoint()
        
        # Start input monitoring thread
        input_thread = self.start_input_thread()
        
        try:
            # Main training loop
            while not self.should_stop:
                episode_success = self.run_episode()
                
                # Periodic saves and updates
                if self.episode_count % self.save_interval == 0:
                    self.save_checkpoint(reason="interval")
                
                # Update policy if we have enough experience
                if len(self.agent.buffer.states) >= 32:
                    policy_loss, value_loss = self.agent.update()
                    if self.episode_count % 10 == 0:
                        print(f"   📈 Policy Loss: {policy_loss:.4f}, Value Loss: {value_loss:.4f}")
                
                # Print progress every few episodes
                if self.episode_count % 5 == 0:
                    self.print_training_progress()
                
        except KeyboardInterrupt:
            print("\n🛑 Training interrupted!")
            self.should_stop = True
        
        finally:
            # Save final checkpoint
            print("\n💾 Saving final checkpoint...")
            self.save_checkpoint(reason="final")
            self.env.close()
            print("✅ Training session complete!")
    
    def run_episode(self) -> bool:
        """Run a single training episode"""
        self.episode_count += 1
        
        # Reset environment
        state = self.env.reset()
        episode_reward = 0
        episode_steps = 0
        episode_start_time = time.time()
        
        # Extract initial arcade info
        current_opponent = int(state[19])
        total_wins = int(state[20])
        
        done = False
        while not done and not self.should_stop:
            # Select action
            action, log_prob, value = self.agent.select_action(state)
            
            # Environment step
            next_state, reward, done, info = self.env.step(action)
            
            # Store experience
            self.agent.buffer.add(state, action, reward, next_state, done, log_prob, value)
            
            episode_reward += reward
            episode_steps += 1
            state = next_state
            
            # Check for arcade events
            if info.get('arcade_completed', False):
                self.arcade_completions += 1
                final_opponent = info.get('current_opponent', 0)
                self.best_arcade_progress = max(self.best_arcade_progress, final_opponent)
                print(f"\n🎉 ARCADE COMPLETED! Total completions: {self.arcade_completions}")
                
            elif info.get('match_completed', False) and info.get('match_winner') != 'PLAYER 1':
                # Track arcade attempt
                current_progress = info.get('arcade_wins', 0)
                self.recent_arcade_attempts.append(current_progress)
                self.best_arcade_progress = max(self.best_arcade_progress, current_progress)
        
        # Store episode statistics
        self.episode_rewards.append(episode_reward)
        
        episode_duration = time.time() - episode_start_time
        
        # Brief episode summary (not too verbose)
        if self.episode_count % 10 == 0 or episode_reward > 50:  # Only show notable episodes
            print(f"Episode {self.episode_count}: Reward {episode_reward:+.1f}, "
                  f"Opponent {current_opponent}, Duration {episode_duration:.1f}s")
        
        return True
    
    def print_training_progress(self):
        """Print current training progress"""
        duration = time.time() - self.training_start_time
        hours = duration / 3600
        
        avg_reward = np.mean(list(self.episode_rewards)) if self.episode_rewards else 0
        recent_progress = np.mean(list(self.recent_arcade_attempts)) if self.recent_arcade_attempts else 0
        
        print(f"\n📊 Training Progress (Episode {self.episode_count}):")
        print(f"   ⏱️  Duration: {hours:.1f}h")
        print(f"   🎯 Avg Reward (last 100): {avg_reward:+.2f}")
        print(f"   🏆 Arcade Completions: {self.arcade_completions}")
        print(f"   📈 Best Progress: {self.best_arcade_progress}/{self.arcade_opponents} opponents")
        print(f"   📊 Recent Avg Progress: {recent_progress:.1f} opponents")
    
    def print_status(self):
        """Print detailed status information"""
        print(f"\n🎮 TRAINING STATUS:")
        print(f"   Episodes: {self.episode_count}")
        print(f"   Training time: {(time.time() - self.training_start_time)/3600:.1f}h")
        print(f"   Best arcade progress: {self.best_arcade_progress}/{self.arcade_opponents}")
        print(f"   Total arcade completions: {self.arcade_completions}")
        if self.episode_rewards:
            print(f"   Recent avg reward: {np.mean(list(self.episode_rewards)):+.2f}")
        print(f"   Buffer size: {len(self.agent.buffer.states)}")
    
    def save_checkpoint(self, reason: str = "manual"):
        """Save training checkpoint"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save model
        model_path = os.path.join(self.models_dir, f"arcade_ppo_{timestamp}.pth")
        latest_path = os.path.join(self.models_dir, "arcade_ppo_latest.pth")
        
        # Determine if this is the best model so far
        current_score = self.calculate_performance_score()
        is_best = current_score > self.best_performance_score
        
        if is_best:
            self.best_performance_score = current_score
            best_path = os.path.join(self.models_dir, "arcade_ppo_best.pth")
        
        # Save training metadata
        metadata = {
            'episode_count': self.episode_count,
            'training_duration': time.time() - self.training_start_time,
            'arcade_completions': self.arcade_completions,
            'best_arcade_progress': self.best_arcade_progress,
            'avg_reward': np.mean(list(self.episode_rewards)) if self.episode_rewards else 0,
            'performance_score': current_score,
            'timestamp': timestamp,
            'save_reason': reason
        }
        
        # Save model with metadata
        self.agent.save(model_path, metadata)
        self.agent.save(latest_path, metadata)
        
        if is_best:
            self.agent.save(best_path, metadata)
            print(f"🏆 New best model saved! Score: {current_score:.3f}")
        
        # Save training log
        log_path = os.path.join(self.models_dir, "training_log.json")
        self.save_training_log(log_path, metadata)
        
        print(f"💾 Checkpoint saved: {model_path}")
        self.last_save_time = time.time()
    
    def calculate_performance_score(self) -> float:
        """Calculate a performance score for model ranking"""
        # Weighted combination of metrics
        completion_rate = self.arcade_completions / max(1, self.episode_count // 20)
        avg_reward = np.mean(list(self.episode_rewards)) if self.episode_rewards else 0
        progress_ratio = self.best_arcade_progress / self.arcade_opponents
        
        # Normalize and weight the metrics
        score = (completion_rate * 0.5 + 
                (avg_reward + 100) / 200 * 0.3 + 
                progress_ratio * 0.2)
        
        return max(0, score)
    
    def load_latest_checkpoint(self):
        """Load the latest checkpoint if available"""
        latest_path = os.path.join(self.models_dir, "arcade_ppo_latest.pth")
        
        if os.path.exists(latest_path):
            try:
                print(f"📂 Loading latest checkpoint...")
                checkpoint = torch.load(latest_path)
                self.agent.policy.load_state_dict(checkpoint['policy_state_dict'])
                self.agent.value.load_state_dict(checkpoint['value_state_dict'])
                self.agent.training_step = checkpoint['training_step']
                
                # Load metadata if available
                if 'metadata' in checkpoint:
                    metadata = checkpoint['metadata']
                    self.episode_count = metadata.get('episode_count', 0)
                    self.arcade_completions = metadata.get('arcade_completions', 0)
                    self.best_arcade_progress = metadata.get('best_arcade_progress', 0)
                    self.best_performance_score = metadata.get('performance_score', 0)
                    
                    print(f"✅ Resumed from Episode {self.episode_count}")
                    print(f"   Arcade completions: {self.arcade_completions}")
                    print(f"   Best progress: {self.best_arcade_progress}/{self.arcade_opponents}")
                else:
                    print("✅ Model loaded (no metadata available)")
                    
            except Exception as e:
                print(f"⚠️ Failed to load checkpoint: {e}")
                print("Starting fresh training...")
        else:
            print("🆕 No checkpoint found, starting fresh training...")
    
    def save_training_log(self, log_path: str, metadata: Dict[str, Any]):
        """Save training progress to log file"""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'episode': self.episode_count,
            'metadata': metadata
        }
        
        # Load existing log or create new
        if os.path.exists(log_path):
            with open(log_path, 'r') as f:
                log_data = json.load(f)
        else:
            log_data = {'training_sessions': []}
        
        # Add new entry
        log_data['training_sessions'].append(log_entry)
        
        # Save updated log
        with open(log_path, 'w') as f:
            json.dump(log_data, f, indent=2)


def main():
    """Main training function"""
    print("🎮 Starting Interactive Arcade Training")
    print("="*50)
    
    # Create trainer
    trainer = InteractiveArcadeTrainer(
        arcade_opponents=3,  # Start with 3 opponents for faster testing
        save_interval_episodes=20
    )
    
    # Print help
    trainer.print_help()
    
    # Start training
    trainer.train(resume_from_latest=True)


if __name__ == "__main__":
    main()