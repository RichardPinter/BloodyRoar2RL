#!/usr/bin/env python3
"""
Interactive DQN Arcade Training Script

Train DQN agent in arcade mode with manual stop control and model persistence.
User can stop training at any time and resume later from saved checkpoint.
Uses hybrid visual + health input for enhanced state representation.
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
from typing import Optional, Dict, Any, Tuple

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.arcade_rl_environment import ArcadeRLEnvironment
from dqn.dqn_agent import DQNAgent


class InteractiveDQNArcadeTrainer:
    """Interactive training system for DQN arcade mode with manual controls"""
    
    def __init__(self, 
                 arcade_opponents: int = 3,
                 save_interval_episodes: int = 20,
                 models_dir: str = "models",
                 frame_stack: int = 4,
                 img_size: Tuple[int, int] = (84, 84),
                 health_history_length: int = 4,
                 lr: float = 1e-4,
                 epsilon_start: float = 1.0,
                 epsilon_end: float = 0.1,
                 epsilon_decay: int = 100000,
                 replay_capacity: int = 200000,
                 batch_size: int = 32,
                 target_update_frequency: int = 2000):
        
        self.arcade_opponents = arcade_opponents
        self.save_interval = save_interval_episodes
        self.models_dir = models_dir
        self.batch_size = batch_size
        
        # Create models directory
        os.makedirs(models_dir, exist_ok=True)
        
        # Initialize environment and agent
        print("🎮 Initializing DQN Arcade Training Environment...")
        self.env = ArcadeRLEnvironment(
            matches_to_win=arcade_opponents, 
            env_type="dqn"  # Use DQN environment type
        )
        
        self.agent = DQNAgent(
            num_actions=self.env.get_action_space_size(),
            frame_stack=frame_stack,
            img_size=img_size,
            health_history_length=health_history_length,
            lr=lr,
            epsilon_start=epsilon_start,
            epsilon_end=epsilon_end,
            epsilon_decay=epsilon_decay,
            replay_capacity=replay_capacity,
            target_update_frequency=target_update_frequency
        )
        
        # Training state
        self.episode_count = 0
        self.total_steps = 0
        self.training_start_time = time.time()
        self.should_stop = False
        self.user_input = ""
        self.training_active = False
        
        # Statistics tracking
        self.episode_rewards = deque(maxlen=100)
        self.episode_lengths = deque(maxlen=100)
        self.arcade_completions = 0
        self.best_arcade_progress = 0  # Best opponent reached
        self.recent_arcade_attempts = deque(maxlen=20)
        self.training_losses = deque(maxlen=1000)
        
        # Performance tracking
        self.last_save_time = time.time()
        self.best_performance_score = 0.0
        self.best_reward = -float('inf')
        
        print(f"✅ DQN Arcade Trainer initialized:")
        print(f"   Arcade: {arcade_opponents} opponents")
        print(f"   Environment: DQN (hybrid visual + health)")
        print(f"   Screenshot input: {frame_stack} × {img_size}")
        print(f"   Health history: {health_history_length} frames")
        print(f"   State dim: {self.env.get_observation_space_size()}")
        print(f"   Action dim: {self.env.get_action_space_size()}")
        print(f"   Models dir: {models_dir}")
        print(f"   Exploration: ε {epsilon_start} → {epsilon_end} over {epsilon_decay:,} steps")
        print(f"   Replay buffer: {replay_capacity:,} transitions")
        
    def start_input_thread(self):
        """Start background thread to monitor user input"""
        def input_monitor():
            while not self.should_stop:
                try:
                    user_input = input().strip().lower()
                    if user_input in ['stop', 'quit', 'q', 'exit']:
                        print("\\n🛑 Stop command received! Finishing current episode...")
                        self.should_stop = True
                        break
                    elif user_input in ['save', 's']:
                        print("💾 Saving checkpoint...")
                        self.save_checkpoint(reason="manual_save")
                    elif user_input in ['status', 'stat']:
                        self.print_status()
                    elif user_input == 'help':
                        self.print_help()
                    elif user_input in ['epsilon', 'eps', 'e']:
                        self.print_exploration_status()
                    elif user_input in ['replay', 'buffer', 'rb']:
                        self.print_replay_buffer_status()
                except EOFError:
                    break
                except KeyboardInterrupt:
                    print("\\n🛑 Keyboard interrupt! Stopping training...")
                    self.should_stop = True
                    break
        
        input_thread = threading.Thread(target=input_monitor, daemon=True)
        input_thread.start()
        return input_thread
    
    def print_help(self):
        """Print available commands"""
        print("\\n📋 AVAILABLE COMMANDS:")
        print("  stop/quit/q     - Stop training after current episode")
        print("  save/s          - Save current model checkpoint")
        print("  status/stat     - Show detailed training status")
        print("  epsilon/eps/e   - Show exploration status")
        print("  replay/buffer   - Show replay buffer status")
        print("  help            - Show this help message")
        print()
    
    def print_exploration_status(self):
        """Print detailed exploration status"""
        current_epsilon = self.agent.get_current_epsilon()
        print(f"\\n🎯 EXPLORATION STATUS:")
        print(f"  Current epsilon: {current_epsilon:.4f}")
        print(f"  Steps done: {self.agent.steps_done:,}")
        print(f"  Target steps for min epsilon: {self.agent.epsilon_decay:,}")
        progress = min(100, (self.agent.steps_done / self.agent.epsilon_decay) * 100)
        print(f"  Exploration decay progress: {progress:.1f}%")
        print()
    
    def print_replay_buffer_status(self):
        """Print replay buffer status"""
        buffer_size = self.agent.replay_buffer.size()
        capacity = self.agent.replay_buffer.capacity
        fill_percentage = (buffer_size / capacity) * 100
        
        print(f"\\n🧠 REPLAY BUFFER STATUS:")
        print(f"  Size: {buffer_size:,} / {capacity:,} ({fill_percentage:.1f}%)")
        print(f"  Memory usage: ~{(buffer_size * 84 * 84 * 4) / (1024**3):.2f} GB")
        print(f"  Ready for training: {'Yes' if buffer_size >= 1000 else 'No'}")
        print()
    
    def print_status(self):
        """Print comprehensive training status"""
        current_time = time.time()
        training_duration = current_time - self.training_start_time
        
        print(f"\\n📊 TRAINING STATUS")
        print("=" * 60)
        
        # Basic info
        print(f"Episode: {self.episode_count}")
        print(f"Total steps: {self.total_steps:,}")
        print(f"Training time: {training_duration/3600:.1f} hours")
        print(f"Training active: {'Yes' if self.training_active else 'No'}")
        
        # Performance
        if len(self.episode_rewards) > 0:
            recent_rewards = list(self.episode_rewards)[-10:]
            print(f"\\nPerformance:")
            print(f"  Recent reward (last 10): {np.mean(recent_rewards):.3f}")
            print(f"  Average reward (last 100): {np.mean(self.episode_rewards):.3f}")
            print(f"  Best reward: {self.best_reward:.3f}")
            print(f"  Episode length avg: {np.mean(self.episode_lengths):.1f}")
        
        # Arcade progress
        if len(self.recent_arcade_attempts) > 0:
            recent_progress = list(self.recent_arcade_attempts)[-5:]
            print(f"\\nArcade Progress:")
            print(f"  Completions: {self.arcade_completions}")
            print(f"  Best progress: {self.best_arcade_progress}/{self.arcade_opponents} opponents")
            print(f"  Recent attempts: {recent_progress}")
        
        # Network training
        current_epsilon = self.agent.get_current_epsilon()
        buffer_size = self.agent.replay_buffer.size()
        print(f"\\nDQN Training:")
        print(f"  Epsilon: {current_epsilon:.4f}")
        print(f"  Replay buffer: {buffer_size:,}")
        print(f"  Training updates: {self.agent.training_step:,}")
        
        if len(self.training_losses) > 0:
            recent_losses = list(self.training_losses)[-100:]
            print(f"  Recent loss: {np.mean(recent_losses):.4f}")
        
        print("=" * 60)
        print()
    
    def save_checkpoint(self, reason: str = "periodic"):
        """Save training checkpoint"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"dqn_arcade_checkpoint_{timestamp}.pth"
        filepath = os.path.join(self.models_dir, filename)
        
        # Prepare metadata
        metadata = {
            'episode': self.episode_count,
            'total_steps': self.total_steps,
            'arcade_completions': self.arcade_completions,
            'best_arcade_progress': self.best_arcade_progress,
            'best_reward': self.best_reward,
            'training_time': time.time() - self.training_start_time,
            'reason': reason,
            'timestamp': datetime.now().isoformat(),
            'replay_buffer_size': self.agent.replay_buffer.size(),
            'current_epsilon': self.agent.get_current_epsilon()
        }
        
        if len(self.episode_rewards) > 0:
            metadata['avg_reward_last_100'] = float(np.mean(self.episode_rewards))
            
        # Save model
        self.agent.save_model(filepath, metadata)
        self.last_save_time = time.time()
        
        print(f"✅ Checkpoint saved: {filename}")
        return filepath
    
    def load_checkpoint(self, filepath: str) -> bool:
        """Load training checkpoint"""
        try:
            if os.path.exists(filepath):
                checkpoint = torch.load(filepath, map_location='cpu')
                
                # Load agent state
                self.agent.load_model(filepath)
                
                # Restore training state from metadata
                if 'metadata' in checkpoint:
                    meta = checkpoint['metadata']
                    self.episode_count = meta.get('episode', 0)
                    self.total_steps = meta.get('total_steps', 0)
                    self.arcade_completions = meta.get('arcade_completions', 0)
                    self.best_arcade_progress = meta.get('best_arcade_progress', 0)
                    self.best_reward = meta.get('best_reward', -float('inf'))
                    
                    print(f"📂 Checkpoint loaded: {os.path.basename(filepath)}")
                    print(f"   Episode: {self.episode_count}")
                    print(f"   Total steps: {self.total_steps:,}")
                    print(f"   Best reward: {self.best_reward:.3f}")
                    print(f"   Arcade completions: {self.arcade_completions}")
                    
                return True
            else:
                print(f"❌ Checkpoint file not found: {filepath}")
                return False
                
        except Exception as e:
            print(f"❌ Error loading checkpoint: {e}")
            return False
    
    def train(self, 
              max_episodes: int = 1000,
              min_replay_size: int = 2000,
              training_start_episode: int = 20,
              update_interval: int = 4):
        """
        Main interactive training loop.
        
        Args:
            max_episodes: Maximum episodes to train
            min_replay_size: Minimum replay buffer size before training
            training_start_episode: Episode to start training
            update_interval: Steps between network updates
        """
        
        print(f"\\n🚀 Starting Interactive DQN Arcade Training")
        print(f"Max episodes: {max_episodes}")
        print(f"Training starts: Episode {training_start_episode}")
        print(f"Min replay size: {min_replay_size:,}")
        print("\\n💡 Type 'help' for available commands during training")
        print("=" * 80)
        
        # Start input monitoring thread
        input_thread = self.start_input_thread()
        
        # Training loop
        steps_since_update = 0
        
        try:
            while self.episode_count < max_episodes and not self.should_stop:
                episode_start_time = time.time()
                
                # Reset environment
                state = self.env.reset()
                if not isinstance(state, tuple) or len(state) != 2:
                    print(f"⚠️ Warning: Expected tuple state, got {type(state)}")
                    continue
                    
                screenshots, health_history = state
                episode_reward = 0
                episode_length = 0
                episode_loss = 0
                loss_count = 0
                
                self.episode_count += 1
                print(f"\\n🎮 Episode {self.episode_count}/{max_episodes}")
                
                # Episode loop
                done = False
                while not done and not self.should_stop:
                    # Select action
                    action = self.agent.select_action(screenshots, health_history)
                    
                    # Environment step
                    next_state, reward, done, info = self.env.step(action)
                    
                    if not isinstance(next_state, tuple) or len(next_state) != 2:
                        print(f"⚠️ Warning: Invalid next_state format")
                        break
                        
                    next_screenshots, next_health_history = next_state
                    
                    # Store experience
                    self.agent.store_transition(
                        screenshots, health_history, action, reward,
                        next_screenshots, next_health_history, done
                    )
                    
                    # Update counters
                    episode_reward += reward
                    episode_length += 1
                    self.total_steps += 1
                    steps_since_update += 1
                    
                    # Train the agent
                    if (self.episode_count >= training_start_episode and 
                        self.agent.replay_buffer.size() >= min_replay_size):
                        
                        if not self.training_active:
                            print(f"  🧠 DQN training activated at episode {self.episode_count}")
                            self.training_active = True
                        
                        if steps_since_update >= update_interval:
                            loss = self.agent.update(self.batch_size)
                            if loss is not None:
                                episode_loss += loss
                                loss_count += 1
                                self.training_losses.append(loss)
                            steps_since_update = 0
                    
                    # Move to next state
                    screenshots, health_history = next_screenshots, next_health_history
                
                # Episode completed
                episode_duration = time.time() - episode_start_time
                self.episode_rewards.append(episode_reward)
                self.episode_lengths.append(episode_length)
                
                # Track arcade progress
                arcade_info = info.get('arcade_info', {})
                current_progress = arcade_info.get('current_opponent', 0)
                arcade_completed = arcade_info.get('arcade_completed', False)
                
                if arcade_completed:
                    self.arcade_completions += 1
                    print(f"  🏆 ARCADE COMPLETED! Total completions: {self.arcade_completions}")
                
                if current_progress > self.best_arcade_progress:
                    self.best_arcade_progress = current_progress
                    print(f"  📈 New best progress: {self.best_arcade_progress}/{self.arcade_opponents}")
                
                self.recent_arcade_attempts.append(current_progress)
                
                # Calculate statistics
                avg_reward = np.mean(self.episode_rewards)
                current_epsilon = self.agent.get_current_epsilon()
                avg_loss = episode_loss / max(loss_count, 1)
                
                # Print episode summary
                print(f"  ✅ Episode {self.episode_count} Complete:")
                print(f"     Reward: {episode_reward:.3f} (Avg: {avg_reward:.3f})")
                print(f"     Length: {episode_length} steps ({episode_duration:.1f}s)")
                print(f"     Arcade progress: {current_progress}/{self.arcade_opponents}")
                print(f"     Epsilon: {current_epsilon:.3f}")
                print(f"     Total steps: {self.total_steps:,}")
                
                if self.training_active and loss_count > 0:
                    print(f"     Avg loss: {avg_loss:.4f}")
                    
                # Save best model
                if episode_reward > self.best_reward:
                    self.best_reward = episode_reward
                    best_path = os.path.join(self.models_dir, 'best_dqn_arcade_model.pth')
                    metadata = {
                        'episode': self.episode_count,
                        'reward': episode_reward,
                        'total_steps': self.total_steps,
                        'arcade_progress': current_progress,
                        'timestamp': datetime.now().isoformat()
                    }
                    self.agent.save_model(best_path, metadata)
                    print(f"  🏆 New best reward: {self.best_reward:.3f}")
                
                # Periodic checkpoint
                if self.episode_count % self.save_interval == 0:
                    self.save_checkpoint("periodic")
                
                # Handle stop request
                if self.should_stop:
                    print(f"\\n🛑 Training stopped by user after episode {self.episode_count}")
                    break
                    
        except KeyboardInterrupt:
            print(f"\\n⏹️  Training interrupted by keyboard")
        except Exception as e:
            print(f"\\n❌ Training error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # Final save
            if not self.should_stop:
                final_path = self.save_checkpoint("final")
                print(f"\\n💾 Final checkpoint saved: {os.path.basename(final_path)}")
            
            # Print final statistics
            self.print_final_statistics()
            
            # Cleanup
            self.env.close()
    
    def print_final_statistics(self):
        """Print final training statistics"""
        training_duration = time.time() - self.training_start_time
        
        print(f"\\n🎉 TRAINING COMPLETE")
        print("=" * 60)
        print(f"Episodes completed: {self.episode_count}")
        print(f"Total steps: {self.total_steps:,}")
        print(f"Training time: {training_duration/3600:.1f} hours")
        print(f"Steps per hour: {self.total_steps/(training_duration/3600):,.0f}")
        
        if len(self.episode_rewards) > 0:
            print(f"\\nPerformance:")
            print(f"  Best reward: {self.best_reward:.3f}")
            print(f"  Average reward (last 100): {np.mean(self.episode_rewards):.3f}")
            print(f"  Average episode length: {np.mean(self.episode_lengths):.1f}")
        
        print(f"\\nArcade Progress:")
        print(f"  Completions: {self.arcade_completions}")
        print(f"  Best progress: {self.best_arcade_progress}/{self.arcade_opponents}")
        
        print(f"\\nDQN Training:")
        print(f"  Training updates: {self.agent.training_step:,}")
        print(f"  Replay buffer size: {self.agent.replay_buffer.size():,}")
        print(f"  Final epsilon: {self.agent.get_current_epsilon():.4f}")
        
        if len(self.training_losses) > 0:
            print(f"  Final average loss: {np.mean(list(self.training_losses)[-100:]):.4f}")
        
        print("=" * 60)


def main():
    """Main function"""
    print("🎮 Interactive DQN Arcade Training")
    print("=" * 50)
    
    # Create trainer
    trainer = InteractiveDQNArcadeTrainer(
        arcade_opponents=3,      # Fight 3 opponents per arcade
        frame_stack=4,           # 4 screenshot frames
        img_size=(84, 84),       # 84x84 screenshots for speed
        health_history_length=4, # 4 health samples
        lr=1e-4,                 # Conservative learning rate
        epsilon_decay=100000,    # Long exploration phase
        replay_capacity=200000,  # Large replay buffer for arcade
        target_update_frequency=2000  # Less frequent target updates
    )
    
    # Check for existing checkpoints
    checkpoints = [f for f in os.listdir(trainer.models_dir) 
                  if f.startswith('dqn_arcade_checkpoint_') and f.endswith('.pth')]
    
    if checkpoints:
        print(f"\\n🔍 Found {len(checkpoints)} existing checkpoint(s):")
        for i, checkpoint in enumerate(sorted(checkpoints)[-5:], 1):  # Show last 5
            print(f"  {i}. {checkpoint}")
        
        response = input("\\nLoad checkpoint? (y/n): ").strip().lower()
        if response in ['y', 'yes']:
            if len(checkpoints) == 1:
                latest_checkpoint = checkpoints[0]
            else:
                latest_checkpoint = sorted(checkpoints)[-1]  # Most recent
            
            checkpoint_path = os.path.join(trainer.models_dir, latest_checkpoint)
            trainer.load_checkpoint(checkpoint_path)
    
    try:
        # Start training
        trainer.train(
            max_episodes=2000,       # Long training session
            min_replay_size=3000,    # Larger minimum for stability
            training_start_episode=30, # Longer warmup for arcade
            update_interval=4        # Update every 4 steps
        )
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()