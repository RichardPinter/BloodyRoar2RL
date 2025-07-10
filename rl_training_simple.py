#!/usr/bin/env python3
"""
Simple RL Training Script for Bloody Roar 2

Uses a basic PPO implementation with PyTorch for training an agent
in the slow RL environment (1-second sampling, variable observation windows).
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

from slow_rl_environment import SlowRLEnvironment


class PolicyNetwork(nn.Module):
    """Actor network - outputs action probabilities"""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_size: int = 128):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_dim)
        
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        action_probs = F.softmax(self.fc3(x), dim=-1)
        return action_probs


class ValueNetwork(nn.Module):
    """Critic network - estimates state value"""
    
    def __init__(self, state_dim: int, hidden_size: int = 128):
        super(ValueNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)
        
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        value = self.fc3(x)
        return value


class ExperienceBuffer:
    """Stores experience tuples for training"""
    
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []
        self.log_probs = []
        self.values = []
        
    def add(self, state, action, reward, next_state, done, log_prob, value):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.dones.append(done)
        self.log_probs.append(log_prob)
        self.values.append(value)
        
    def get_batch(self):
        """Convert lists to tensors and calculate advantages"""
        states = torch.FloatTensor(np.array(self.states))
        actions = torch.LongTensor(self.actions)
        rewards = torch.FloatTensor(self.rewards)
        next_states = torch.FloatTensor(np.array(self.next_states))
        dones = torch.FloatTensor(self.dones)
        log_probs = torch.FloatTensor(self.log_probs)
        values = torch.FloatTensor(self.values)
        
        # Calculate advantages using GAE
        advantages = self._calculate_advantages(rewards, values, dones)
        returns = advantages + values
        
        return states, actions, log_probs, returns, advantages
        
    def _calculate_advantages(self, rewards, values, dones, gamma=0.99, lam=0.95):
        """Generalized Advantage Estimation"""
        advantages = torch.zeros_like(rewards)
        last_advantage = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[t + 1]
                
            delta = rewards[t] + gamma * next_value * (1 - dones[t]) - values[t]
            last_advantage = delta + gamma * lam * (1 - dones[t]) * last_advantage
            advantages[t] = last_advantage
            
        return advantages
        
    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.next_states.clear()
        self.dones.clear()
        self.log_probs.clear()
        self.values.clear()


class PPOAgent:
    """PPO Agent for training"""
    
    def __init__(self, state_dim: int, action_dim: int, lr: float = 3e-4):
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Networks
        self.policy = PolicyNetwork(state_dim, action_dim)
        self.value = ValueNetwork(state_dim)
        
        # Optimizers
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.value_optimizer = optim.Adam(self.value.parameters(), lr=lr)
        
        # Experience buffer
        self.buffer = ExperienceBuffer()
        
        # Training stats
        self.training_step = 0
        
    def select_action(self, state: np.ndarray) -> Tuple[int, float, float]:
        """Select action using current policy"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        
        with torch.no_grad():
            action_probs = self.policy(state_tensor)
            value = self.value(state_tensor)
            
        # Sample action from probability distribution
        dist = torch.distributions.Categorical(action_probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        
        return action.item(), log_prob.item(), value.item()
        
    def update(self, ppo_epochs: int = 4, clip_ratio: float = 0.2):
        """Update policy and value networks using PPO"""
        states, actions, old_log_probs, returns, advantages = self.buffer.get_batch()
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        for _ in range(ppo_epochs):
            # Policy update
            action_probs = self.policy(states)
            dist = torch.distributions.Categorical(action_probs)
            new_log_probs = dist.log_prob(actions)
            
            # PPO clipped objective
            ratio = torch.exp(new_log_probs - old_log_probs)
            clipped_ratio = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio)
            policy_loss = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()
            
            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            self.policy_optimizer.step()
            
            # Value update
            values = self.value(states).squeeze()
            value_loss = F.mse_loss(values, returns)
            
            self.value_optimizer.zero_grad()
            value_loss.backward()
            self.value_optimizer.step()
            
        self.training_step += 1
        self.buffer.clear()
        
        return policy_loss.item(), value_loss.item()
        
    def save(self, path: str):
        """Save model weights"""
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'value_state_dict': self.value.state_dict(),
            'training_step': self.training_step
        }, path)
        print(f"💾 Model saved to {path}")
        
    def load(self, path: str):
        """Load model weights"""
        checkpoint = torch.load(path)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.value.load_state_dict(checkpoint['value_state_dict'])
        self.training_step = checkpoint['training_step']
        print(f"📂 Model loaded from {path}")


class RLTrainer:
    """Main training class that coordinates everything"""
    
    def __init__(self):
        self.env = SlowRLEnvironment()
        self.agent = PPOAgent(
            state_dim=self.env.get_observation_space_size(),
            action_dim=self.env.get_action_space_size()
        )
        
        # Training stats
        self.episode_rewards = deque(maxlen=100)
        self.episode_lengths = deque(maxlen=100)
        
        # Create model directory
        self.model_dir = "models"
        os.makedirs(self.model_dir, exist_ok=True)
        
    def train(self, num_episodes: int = 100, update_interval: int = 32):
        """Main training loop"""
        print("🎮 Starting RL Training")
        print(f"Episodes: {num_episodes}")
        print(f"State dimension: {self.env.get_observation_space_size()}")
        print(f"Action dimension: {self.env.get_action_space_size()}")
        print("=" * 60)
        
        steps_collected = 0
        best_reward = -float('inf')
        
        for episode in range(num_episodes):
            state = self.env.reset()
            episode_reward = 0
            episode_length = 0
            
            print(f"\n📺 Episode {episode + 1}/{num_episodes}")
            
            done = False
            while not done:
                # Select action
                action, log_prob, value = self.agent.select_action(state)
                
                # Environment step
                next_state, reward, done, info = self.env.step(action)
                
                # Store experience
                self.agent.buffer.add(
                    state, action, reward, next_state, 
                    done, log_prob, value
                )
                
                episode_reward += reward
                episode_length += 1
                steps_collected += 1
                state = next_state
                
                # Update policy when enough steps collected
                if steps_collected >= update_interval:
                    policy_loss, value_loss = self.agent.update()
                    print(f"  📈 Update: Policy Loss={policy_loss:.4f}, Value Loss={value_loss:.4f}")
                    steps_collected = 0
                    
            # Episode complete
            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(episode_length)
            
            # Print episode summary
            avg_reward = np.mean(self.episode_rewards)
            print(f"  ✅ Episode Complete:")
            print(f"     Reward: {episode_reward:.3f} (Avg: {avg_reward:.3f})")
            print(f"     Length: {episode_length} steps")
            
            # Save best model
            if episode_reward > best_reward:
                best_reward = episode_reward
                model_path = os.path.join(self.model_dir, 'best_model.pth')
                self.agent.save(model_path)
                print(f"  🏆 New best reward: {best_reward:.3f}")
                
            # Periodic save
            if (episode + 1) % 10 == 0:
                model_path = os.path.join(self.model_dir, f'checkpoint_ep{episode+1}.pth')
                self.agent.save(model_path)
                
        print("\n🎉 Training Complete!")
        print(f"Best reward: {best_reward:.3f}")
        print(f"Average reward (last 100): {np.mean(self.episode_rewards):.3f}")
        
    def evaluate(self, num_episodes: int = 10):
        """Evaluate the trained agent"""
        print("\n🧪 Evaluating Agent")
        print("=" * 40)
        
        eval_rewards = []
        
        for episode in range(num_episodes):
            state = self.env.reset()
            episode_reward = 0
            done = False
            
            while not done:
                # Use greedy action selection (no exploration)
                with torch.no_grad():
                    state_tensor = torch.FloatTensor(state).unsqueeze(0)
                    action_probs = self.agent.policy(state_tensor)
                    action = torch.argmax(action_probs).item()
                
                state, reward, done, info = self.env.step(action)
                episode_reward += reward
                
            eval_rewards.append(episode_reward)
            print(f"Episode {episode + 1}: Reward = {episode_reward:.3f}")
            
        print(f"\nAverage Evaluation Reward: {np.mean(eval_rewards):.3f}")
        print(f"Std Dev: {np.std(eval_rewards):.3f}")


def main():
    """Main function"""
    trainer = RLTrainer()
    
    try:
        # Train the agent
        trainer.train(num_episodes=50)
        
        # Evaluate performance
        trainer.evaluate(num_episodes=5)
        
    except KeyboardInterrupt:
        print("\n⏹️  Training interrupted by user")
    except Exception as e:
        print(f"❌ Error during training: {e}")
        import traceback
        traceback.print_exc()
    finally:
        trainer.env.close()


if __name__ == "__main__":
    main()