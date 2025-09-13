#!/usr/bin/env python3
"""
Neural network trainer module.
Handles DQN training, model updates, and checkpointing.
"""
import time
import torch
import torch.nn as nn
import re
import os
from config import *
from logging_utils import log_learner, log_state
from models import DQNNet
from game_vision import legal_mask_from_extras
from torch.utils.tensorboard import SummaryWriter

class Trainer:
    """Handles neural network training and model management"""
    
    def __init__(self, shared_state):
        self.shared = shared_state
        self.writer = SummaryWriter(log_dir=f"{LOG_DIR}/tensorboard_trainer")
        
        # Training components
        self.optimizer = torch.optim.Adam(
            self.shared.policy_net.parameters(), 
            lr=LEARNING_RATE
        )
        self.criterion = nn.SmoothL1Loss()  # Huber loss
        
        # Training state
        self.train_steps = 0
        self.learn_tick = 0
        
        # Load checkpoint if available
        self.load_checkpoint()
        
        log_state(f"Trainer initialized - {'TEST MODE' if TEST_MODE else 'TRAINING MODE'}")
    
    def load_checkpoint(self):
        """Load model checkpoint if specified"""
        if LOAD_CHECKPOINT and os.path.exists(LOAD_CHECKPOINT):
            checkpoint = torch.load(LOAD_CHECKPOINT, map_location=DEVICE)
            self.shared.policy_net.load_state_dict(checkpoint)
            self.shared.target_net.load_state_dict(checkpoint)
            log_state(f"✅ Loaded checkpoint from {LOAD_CHECKPOINT}")
            
            # Extract match number from filename
            match = re.search(r'model_match_(\d+)', LOAD_CHECKPOINT)
            if match:
                start_match = int(match.group(1)) + 1
                self.shared.match_number = start_match
                log_state(f"   Continuing from match {start_match}")
        else:
            self.shared.target_net.load_state_dict(self.shared.policy_net.state_dict())
            if LOAD_CHECKPOINT:
                log_state(f"⚠️  Checkpoint {LOAD_CHECKPOINT} not found, training from scratch")
    
    @torch.no_grad()
    def soft_update(self, tau=TAU):
        """Soft update of target network parameters"""
        for tp, sp in zip(self.shared.target_net.parameters(), 
                         self.shared.policy_net.parameters()):
            tp.data.mul_(1.0 - tau).add_(sp.data, alpha=tau)
    
    def train_step(self):
        """Perform one training step"""
        # Sample batch from replay buffer
        (states, extras, actions, rewards, 
         next_states, next_extras, dones) = self.shared.replay_buffer.sample(BATCH_SIZE)
        
        # Log buffer statistics periodically
        if self.train_steps % 1000 == 0:
            self.log_buffer_stats()
        
        # Compute TD targets with Double DQN
        with torch.no_grad():
            # Mask illegal actions in next state
            mask_next = legal_mask_from_extras(next_extras)  # [B, A]
            
            # Select actions with online network
            q_online_next = self.shared.policy_net(next_states, next_extras)
            q_online_next = q_online_next.masked_fill(~mask_next, float("-inf"))
            next_actions = q_online_next.argmax(1, keepdim=True)  # [B,1]
            
            # Evaluate with target network
            q_target_next = self.shared.target_net(next_states, next_extras)
            q_target_next = q_target_next.masked_fill(~mask_next, float("-inf"))
            next_q = q_target_next.gather(1, next_actions).squeeze(1)
            
            # Compute targets
            target = rewards + GAMMA * next_q * (1 - dones.float())
            target = target.clamp(-3.0, 3.0)
            
            # Log TD error statistics
            current_q_detached = self.shared.policy_net(states, extras).gather(
                1, actions.unsqueeze(1)).squeeze(1)
            td_error = (target - current_q_detached).abs()
            self.writer.add_scalar("training/td_error_mean", 
                                  td_error.mean().item(), self.train_steps)
            self.writer.add_scalar("training/td_error_max", 
                                  td_error.max().item(), self.train_steps)
        
        # Forward pass
        q_vals = self.shared.policy_net(states, extras).gather(
            1, actions.unsqueeze(1)).squeeze(1)
        loss = self.criterion(q_vals, target)
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        
        # Log gradient statistics
        total_grad_norm = self.log_gradients()
        
        # Clip gradients
        clipped_norm = torch.nn.utils.clip_grad_norm_(
            self.shared.policy_net.parameters(), max_norm=1.0)
        self.writer.add_scalar("gradients/norm_after_clip", 
                              clipped_norm.item(), self.train_steps)
        
        # Optimizer step
        self.optimizer.step()
        
        # Soft update target network
        self.soft_update()
        
        # Log training metrics
        self.writer.add_scalar("loss/train", loss.item(), self.train_steps)
        self.writer.add_scalar("training/learning_rate", LEARNING_RATE, self.train_steps)
        self.writer.add_scalar("buffer/size", self.shared.replay_buffer.len, self.train_steps)
        self.writer.add_scalar("buffer/utilization", 
                              self.shared.replay_buffer.len / self.shared.replay_buffer.size, 
                              self.train_steps)
        
        # Log layer statistics periodically
        if self.train_steps % 500 == 0:
            self.log_layer_stats()
        
        self.train_steps += 1
        
        # Log progress
        if self.train_steps % 100 == 0:
            log_learner(f"[Learner] step={self.train_steps:6d} "
                       f"loss={loss.item():.4f} "
                       f"buffer_len={self.shared.replay_buffer.len} "
                       f"grad_norm={total_grad_norm:.4f}")
        
        return loss.item()
    
    def log_buffer_stats(self):
        """Log replay buffer statistics"""
        buffer = self.shared.replay_buffer
        if buffer.len > 0:
            rewards_in_buffer = buffer.rewards[:buffer.len]
            self.writer.add_scalar("buffer/reward_mean", 
                                  rewards_in_buffer.mean(), self.train_steps)
            self.writer.add_scalar("buffer/reward_std", 
                                  rewards_in_buffer.std(), self.train_steps)
            self.writer.add_scalar("buffer/reward_min", 
                                  rewards_in_buffer.min(), self.train_steps)
            self.writer.add_scalar("buffer/reward_max", 
                                  rewards_in_buffer.max(), self.train_steps)
            
            # Log action distribution in buffer
            actions_in_buffer = buffer.actions[:buffer.len]
            for i, action_name in enumerate(ACTIONS):
                action_pct = (actions_in_buffer == i).sum() / len(actions_in_buffer)
                self.writer.add_scalar(f"buffer/action_distribution/{action_name}",
                                      action_pct, self.train_steps)
            
            # Log done percentage
            dones_in_buffer = buffer.dones[:buffer.len]
            self.writer.add_scalar("buffer/done_percentage",
                                  dones_in_buffer.sum() / len(dones_in_buffer), 
                                  self.train_steps)
    
    def log_gradients(self):
        """Log gradient statistics before clipping"""
        total_grad_norm = 0
        for p in self.shared.policy_net.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2).item()
                total_grad_norm += param_norm ** 2
        total_grad_norm = total_grad_norm ** 0.5
        self.writer.add_scalar("gradients/norm_before_clip", 
                              total_grad_norm, self.train_steps)
        return total_grad_norm
    
    def log_layer_stats(self):
        """Log layer-wise statistics"""
        for name, param in self.shared.policy_net.named_parameters():
            if param.requires_grad and param.grad is not None:
                self.writer.add_histogram(f"weights/{name}", 
                                         param.data, self.train_steps)
                self.writer.add_histogram(f"gradients/{name}", 
                                         param.grad.data, self.train_steps)
                self.writer.add_scalar(f"weights/{name}/mean", 
                                      param.data.mean().item(), self.train_steps)
                self.writer.add_scalar(f"weights/{name}/std", 
                                      param.data.std().item(), self.train_steps)
    
    def save_checkpoint(self):
        """Save model checkpoint"""
        match_num = self.shared.match_number
        model_path = f"{MODEL_DIR}/model_match_{match_num}.pth"
        torch.save(self.shared.policy_net.state_dict(), model_path)
        log_learner(f"[Learner] Saved model to {model_path}")
        
        # Log match completion
        self.writer.add_scalar("training/match_completed", match_num, self.train_steps)
        
        # Don't clear buffer - preserve experience
        log_learner(f"[Learner] Buffer preserved with {self.shared.replay_buffer.len} samples")
    
    def run(self):
        """Main training loop"""
        if TEST_MODE:
            log_learner("[Learner] TEST MODE - Training disabled")
            while not self.shared.stop_event.is_set():
                # Still check for match end to save current model state
                if self.shared.match_end_event.wait(timeout=1.0):
                    self.shared.match_end_event.clear()
                    log_learner(f"[Learner] Match ended in test mode")
                    self.shared.match_number += 1
            return
        
        while not self.shared.stop_event.is_set():
            self.learn_tick += 1
            
            # Train when replay is warm and throttle by TRAIN_FREQ
            if (self.shared.replay_buffer.len >= LEARNING_STARTS and 
                self.learn_tick % TRAIN_FREQ == 0):
                self.train_step()
            
            # Check for match end to save model
            if self.shared.match_end_event.wait(timeout=0.01):
                self.shared.match_end_event.clear()
                self.save_checkpoint()
                self.shared.match_number += 1
            
            # Small sleep to prevent CPU spinning
            time.sleep(0.001)
        
        self.writer.close()
        log_learner("[Learner] Training stopped")