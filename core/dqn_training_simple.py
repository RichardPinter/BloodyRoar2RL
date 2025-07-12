#!/usr/bin/env python3
"""
Simple DQN Training Script for Bloody Roar 2

Uses Deep Q-Network (DQN) with experience replay for training an agent
in the DQN slow RL environment (hybrid visual + health input).
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from collections import deque
import time
import os
from datetime import datetime
from typing import List, Tuple, Dict, Any
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.dqn_slow_rl_environment import DQNSlowRLEnvironment
from dqn.dqn_agent import DQNAgent


class DQNTrainer:
    """Main DQN training class that coordinates everything"""
    
    def __init__(self, 
                 frame_stack: int = 8,
                 img_size: Tuple[int, int] = (84, 84),
                 health_history_length: int = 8,
                 observation_window_seconds: int = 8,
                 lr: float = 1e-4,
                 epsilon_start: float = 1.0,
                 epsilon_end: float = 0.1,
                 epsilon_decay: int = 50000,
                 replay_capacity: int = 100000,
                 batch_size: int = 32,
                 target_update_frequency: int = 1000):
        """
        Initialize DQN trainer.
        
        Args:
            frame_stack: Number of screenshot frames to stack
            img_size: Target size for screenshots (height, width)
            health_history_length: Number of health frames to track
            observation_window_seconds: How many seconds to observe (1 screenshot per second)
            lr: Learning rate for network optimization
            epsilon_start: Initial exploration rate
            epsilon_end: Final exploration rate
            epsilon_decay: Steps to decay exploration over
            replay_capacity: Size of experience replay buffer
            batch_size: Mini-batch size for training
            target_update_freq: Frequency to update target network
        """
        
        print("🎮 Initializing DQN Training Environment...")
        
        # Environment setup
        self.env = DQNSlowRLEnvironment(
            frame_stack_size=frame_stack,
            img_size=img_size,
            health_history_length=health_history_length,
            observation_window_seconds=observation_window_seconds
        )
        
        # Agent setup
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
        
        # Training parameters
        self.batch_size = batch_size
        self.target_update_freq = target_update_frequency
        
        # Training stats
        self.episode_rewards = deque(maxlen=100)
        self.episode_lengths = deque(maxlen=100)
        self.episode_epsilons = deque(maxlen=100)
        self.training_losses = deque(maxlen=1000)
        
        # Create model directory
        self.model_dir = "models"
        os.makedirs(self.model_dir, exist_ok=True)
        
        print(f"✅ DQN Trainer initialized:")
        print(f"   Environment: DQN Slow RL (hybrid visual + health)")
        print(f"   Screenshot input: {frame_stack} × {img_size}")
        print(f"   Health history: {health_history_length} frames")
        print(f"   Observation window: {observation_window_seconds} seconds")
        print(f"   Action space: {self.env.get_action_space_size()}")
        print(f"   Observation space: {self.env.get_observation_space_size()}")
        print(f"   Exploration: ε {epsilon_start} → {epsilon_end} over {epsilon_decay:,} steps")
        print(f"   Replay buffer: {replay_capacity:,} transitions")
        print(f"   Target update: every {self.target_update_freq} steps")
        
    def train(self, 
              num_episodes: int = 200, 
              min_replay_size: int = 1000,
              training_start_episode: int = 10,
              log_interval: int = 10):
        """
        Main training loop.
        
        Args:
            num_episodes: Number of episodes to train
            min_replay_size: Minimum replay buffer size before training
            training_start_episode: Episode to start training (collect experience first)
            log_interval: Episodes between detailed logging
        """
        print("\\n🎯 Starting DQN Training")
        print(f"Episodes: {num_episodes}")
        print(f"Training starts: Episode {training_start_episode}")
        print(f"Min replay size: {min_replay_size:,}")
        print("=" * 80)
        
        total_steps = 0
        best_reward = -float('inf')
        training_started = False
        steps_since_training = 0  # Counter for training every 4 steps
        
        for episode in range(num_episodes):
            # Reset environment and get initial state
            state = self.env.reset()
            if isinstance(state, tuple):
                screenshots, health_history = state
            else:
                print(f"⚠️ Warning: Expected tuple state, got {type(state)}")
                continue
                
            episode_reward = 0
            episode_length = 0
            episode_loss = 0
            loss_count = 0
            
            print(f"\\n📺 Episode {episode + 1}/{num_episodes}")
            
            done = False
            while not done:
                # Select action using DQN agent
                action = self.agent.select_action(screenshots, health_history)
                
                # Environment step
                next_state, reward, done, info = self.env.step(action)
                
                if isinstance(next_state, tuple):
                    next_screenshots, next_health_history = next_state
                else:
                    print(f"⚠️ Warning: Expected tuple next_state, got {type(next_state)}")
                    break
                
                # Store experience in replay buffer
                self.agent.store_transition(
                    screenshots, health_history, action, reward, 
                    next_screenshots, next_health_history, done
                )
                
                # Update counters
                episode_reward += reward
                episode_length += 1
                total_steps += 1
                steps_since_training += 1
                
                # Debug: Show training conditions every 4 steps
                if steps_since_training >= 4:
                    print(f"    Debug: episode={episode+1}, buffer_size={self.agent.replay_buffer.size}, steps_since_training={steps_since_training}")
                
                # Train the agent every 4 steps (if enough experience and past warmup period)
                if (episode >= training_start_episode and 
                    self.agent.replay_buffer.size >= min_replay_size and
                    steps_since_training >= 4):
                    
                    if not training_started:
                        print(f"  🧠 Starting DQN training at episode {episode + 1}")
                        training_started = True
                    
                    loss = self.agent.update(self.batch_size)
                    if loss is not None:
                        episode_loss += loss
                        loss_count += 1
                        self.training_losses.append(loss)
                        print(f"    🧠 Training update at step {episode_length} (loss: {loss:.4f})")
                    
                    steps_since_training = 0  # Reset counter after training
                
                # Move to next state
                screenshots, health_history = next_screenshots, next_health_history
                    
            # Episode complete - record statistics
            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(episode_length)
            current_epsilon = self.agent.get_current_epsilon()
            self.episode_epsilons.append(current_epsilon)
            
            # Calculate averages
            avg_reward = np.mean(self.episode_rewards)
            avg_length = np.mean(self.episode_lengths)
            avg_loss = episode_loss / max(loss_count, 1)
            
            # Print episode summary
            print(f"  ✅ Episode Complete:")
            print(f"     Reward: {episode_reward:.3f} (Avg: {avg_reward:.3f})")
            print(f"     Length: {episode_length} steps (Avg: {avg_length:.1f})")
            print(f"     Epsilon: {current_epsilon:.3f}")
            print(f"     Total Steps: {total_steps:,}")
            print(f"     Replay Size: {self.agent.replay_buffer.size:,}")
            
            if training_started and loss_count > 0:
                print(f"     Avg Loss: {avg_loss:.4f}")
            
            # Save best model
            if episode_reward > best_reward:
                best_reward = episode_reward
                model_path = os.path.join(self.model_dir, 'best_dqn_model.pth')
                metadata = {
                    'episode': episode + 1,
                    'reward': episode_reward,
                    'total_steps': total_steps,
                    'timestamp': datetime.now().isoformat()
                }
                self.agent.save(model_path, metadata)
                print(f"  🏆 New best reward: {best_reward:.3f}")
                
            # Periodic detailed logging
            if (episode + 1) % log_interval == 0:
                self.print_detailed_stats(episode + 1, total_steps, training_started)
                
            # Periodic checkpoint save
            if (episode + 1) % 25 == 0:
                checkpoint_path = os.path.join(self.model_dir, f'dqn_checkpoint_ep{episode+1}.pth')
                metadata = {
                    'episode': episode + 1,
                    'total_steps': total_steps,
                    'avg_reward': avg_reward,
                    'training_started': training_started,
                    'timestamp': datetime.now().isoformat()
                }
                self.agent.save(checkpoint_path, metadata)
                print(f"  💾 Checkpoint saved: episode {episode + 1}")
                
        print("\\n🎉 DQN Training Complete!")
        print(f"Best reward: {best_reward:.3f}")
        print(f"Average reward (last 100): {np.mean(self.episode_rewards):.3f}")
        print(f"Total steps: {total_steps:,}")
        print(f"Final epsilon: {self.agent.get_current_epsilon():.3f}")
        
    def print_detailed_stats(self, episode: int, total_steps: int, training_started: bool):
        """Print detailed training statistics"""
        print(f"\\n📊 TRAINING STATS (Episode {episode})")
        print("=" * 50)
        
        # Reward statistics
        if len(self.episode_rewards) > 0:
            print(f"Rewards:")
            print(f"  Recent (last 10): {np.mean(list(self.episode_rewards)[-10:]):.3f}")
            print(f"  Average (last 100): {np.mean(self.episode_rewards):.3f}")
            print(f"  Best: {max(self.episode_rewards):.3f}")
            print(f"  Worst: {min(self.episode_rewards):.3f}")
        
        # Training progress
        print(f"Progress:")
        print(f"  Total steps: {total_steps:,}")
        print(f"  Replay buffer: {self.agent.replay_buffer.size:,}")
        print(f"  Training active: {'Yes' if training_started else 'No'}")
        
        # Exploration
        current_epsilon = self.agent.get_current_epsilon()
        print(f"Exploration:")
        print(f"  Current epsilon: {current_epsilon:.3f}")
        print(f"  Steps done: {self.agent.steps_done:,}")
        
        # Loss statistics (if training)
        if training_started and len(self.training_losses) > 0:
            recent_losses = list(self.training_losses)[-100:]
            print(f"Training:")
            print(f"  Recent loss (last 100): {np.mean(recent_losses):.4f}")
            print(f"  Training updates: {self.agent.training_step:,}")
        
        print("=" * 50)
        
    def evaluate(self, num_episodes: int = 10, epsilon: float = 0.05):
        """
        Evaluate the trained agent.
        
        Args:
            num_episodes: Number of episodes to evaluate
            epsilon: Small exploration rate for evaluation
        """
        print(f"\\n🧪 Evaluating DQN Agent ({num_episodes} episodes)")
        print("=" * 50)
        
        eval_rewards = []
        eval_lengths = []
        
        # Temporarily set low epsilon for evaluation
        original_epsilon = self.agent.get_current_epsilon()
        
        for episode in range(num_episodes):
            state = self.env.reset()
            if isinstance(state, tuple):
                screenshots, health_history = state
            else:
                print(f"⚠️ Warning: Expected tuple state during evaluation")
                continue
                
            episode_reward = 0
            episode_length = 0
            done = False
            
            while not done:
                # Use mostly greedy action selection (small epsilon)
                if np.random.random() < epsilon:
                    action = np.random.randint(0, self.env.get_action_space_size())
                else:
                    action = self.agent.select_greedy_action(screenshots, health_history)
                
                next_state, reward, done, info = self.env.step(action)
                
                if isinstance(next_state, tuple):
                    screenshots, health_history = next_state
                else:
                    break
                    
                episode_reward += reward
                episode_length += 1
                
            eval_rewards.append(episode_reward)
            eval_lengths.append(episode_length)
            print(f"Episode {episode + 1}: Reward = {episode_reward:.3f}, Length = {episode_length}")
            
        # Print evaluation summary
        print(f"\\nEvaluation Results:")
        print(f"  Average Reward: {np.mean(eval_rewards):.3f} ± {np.std(eval_rewards):.3f}")
        print(f"  Average Length: {np.mean(eval_lengths):.1f} ± {np.std(eval_lengths):.1f}")
        print(f"  Best Episode: {max(eval_rewards):.3f}")
        print(f"  Worst Episode: {min(eval_rewards):.3f}")
        
    def load_model(self, model_path: str):
        """Load a saved model"""
        if os.path.exists(model_path):
            self.agent.load(model_path)
            print(f"📂 Loaded model from {model_path}")
        else:
            print(f"❌ Model file not found: {model_path}")


def main():
    """Main function"""
    # Create trainer with sensible defaults for fighting game
    trainer = DQNTrainer(
        frame_stack=8,           # 8 frames for full temporal information
        img_size=(84, 84),       # Smaller images for faster training
        health_history_length=8,  # 8 health samples
        observation_window_seconds=8,  # 8 seconds of observation (configurable)
        lr=1e-4,                 # Conservative learning rate
        epsilon_decay=50000,     # Explore for first ~250 episodes (200 steps/episode avg)
        replay_capacity=100000,  # Large replay buffer
        batch_size=32,           # Standard batch size
        target_update_frequency=1000  # Update target network every 1000 steps
    )
    
    try:
        # Check for existing model to resume from
        best_model_path = os.path.join(trainer.model_dir, 'best_dqn_model.pth')
        if os.path.exists(best_model_path):
            print(f"\\n🔍 Found existing model: {best_model_path}")
            response = input("Load existing model? (y/n): ").strip().lower()
            if response in ['y', 'yes']:
                trainer.load_model(best_model_path)
        
        # Train the agent
        print(f"\\n🚀 Starting training...")
        trainer.train(
            num_episodes=300,        # Extended training for DQN
            min_replay_size=100,     # Start training after 100 transitions (was 2000)
            training_start_episode=1, # Start training from episode 1 (was 20)
            log_interval=20          # Log every 20 episodes
        )
        
        # Evaluate performance
        trainer.evaluate(num_episodes=10)
        
    except KeyboardInterrupt:
        print("\\n⏹️  Training interrupted by user")
    except Exception as e:
        print(f"❌ Error during training: {e}")
        import traceback
        traceback.print_exc()
    finally:
        trainer.env.close()


if __name__ == "__main__":
    main()