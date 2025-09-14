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

from src.config import *
from src.logging_utils import log_learner, log_state
from src.models import DQNNet  # kept for clarity; networks come from shared_state
from src.game_vision import legal_mask_from_extras
from torch.utils.tensorboard import SummaryWriter


# How often to print Q stats + loss to console
_CONSOLE_LOG_EVERY_STEPS = 100
# How often to log buffer stats
_BUFFER_LOG_EVERY_STEPS = 1000
# Gradient clipping (you can also move this to config if preferred)
_MAX_GRAD_NORM = 1.0
# Optional clamp on TD targets for stability (keep your previous behavior)
_TARGET_CLAMP = (-3.0, 3.0)


class Trainer:
    """Handles neural network training and model management"""

    def __init__(self, shared_state):
        self.shared = shared_state
        self.writer = SummaryWriter(
            log_dir=f"{LOG_DIR}/tensorboard_trainer",
            flush_secs=2,
            max_queue=1000
        )

        # Optimizer & loss
        self.optimizer = torch.optim.Adam(
            self.shared.policy_net.parameters(),
            lr=LEARNING_RATE
        )
        self.criterion = nn.SmoothL1Loss()  # Huber loss

        # Training state
        self.train_steps = 0
        self.learn_tick = 0
        self.best_loss = float("inf")

        # Load checkpoint if available
        self.load_checkpoint()

        log_state(f"Trainer initialized - {'TEST MODE' if TEST_MODE else 'TRAINING MODE'}")

    def load_checkpoint(self):
        """Load model checkpoint if specified (supports raw or full ckpt dict)."""
        if LOAD_CHECKPOINT and os.path.exists(LOAD_CHECKPOINT):
            ckpt = torch.load(LOAD_CHECKPOINT, map_location=DEVICE)
            if isinstance(ckpt, dict) and "policy" in ckpt:
                # full checkpoint
                self.shared.policy_net.load_state_dict(ckpt["policy"])
                self.shared.target_net.load_state_dict(ckpt.get("target", ckpt["policy"]))
                if "optimizer" in ckpt:
                    try:
                        self.optimizer.load_state_dict(ckpt["optimizer"])
                    except Exception:
                        log_state("⚠️  Optimizer state in checkpoint incompatible; skipping.")
                self.train_steps = ckpt.get("train_steps", self.train_steps)
                self.shared.match_number = ckpt.get("match_number", self.shared.match_number)
            else:
                # raw state_dict
                self.shared.policy_net.load_state_dict(ckpt)
                self.shared.target_net.load_state_dict(ckpt)
            log_state(f"✅ Loaded checkpoint from {LOAD_CHECKPOINT}")

            # Extract match number from filename (legacy behavior)
            match = re.search(r'model_match_(\d+)', LOAD_CHECKPOINT)
            if match:
                start_match = int(match.group(1)) + 1
                self.shared.match_number = max(self.shared.match_number, start_match)
                log_state(f"   Continuing from match {self.shared.match_number}")
        else:
            # Start target as a copy of policy
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
        """Perform one training step (Double-DQN / masked)"""
        # Sample batch from replay buffer
        (states, extras, actions, rewards,
         next_states, next_extras, dones) = self.shared.replay_buffer.sample(BATCH_SIZE)

        # Periodic buffer stats
        if self.train_steps % _BUFFER_LOG_EVERY_STEPS == 0:
            self.log_buffer_stats()

        # -----------------------
        # Compute Double-DQN target with legality masking
        # -----------------------
        with torch.no_grad():
            # Mask illegal actions in next state
            mask_next = legal_mask_from_extras(next_extras)  # [B, A] boolean

            # Online net selects argmax action
            q_online_next = self.shared.policy_net(next_states, next_extras)  # [B, A]
            q_online_next = q_online_next.masked_fill(~mask_next, float("-inf"))
            next_actions = q_online_next.argmax(1, keepdim=True)  # [B, 1]

            # Target net evaluates that action
            q_target_next = self.shared.target_net(next_states, next_extras)  # [B, A]
            q_target_next = q_target_next.masked_fill(~mask_next, float("-inf"))
            next_q = q_target_next.gather(1, next_actions).squeeze(1)  # [B]

            # Target = r + γ (1-d) Q_tgt(s', a*)
            target = rewards + GAMMA * next_q * (1.0 - dones.float())
            if _TARGET_CLAMP is not None:
                lo, hi = _TARGET_CLAMP
                target = target.clamp(lo, hi)

        # -----------------------
        # Forward pass (single forward; reuse for loss & Q stats)
        # -----------------------
        q_batch = self.shared.policy_net(states, extras)                      # [B, A]
        q_vals = q_batch.gather(1, actions.unsqueeze(1)).squeeze(1)           # [B]

        loss = self.criterion(q_vals, target)

        # -----------------------
        # Backward + step
        # -----------------------
        self.optimizer.zero_grad()
        loss.backward()

        # Gradient norms (before clipping)
        total_grad_norm = self.log_gradients()

        # Clip gradients
        if _MAX_GRAD_NORM is not None and _MAX_GRAD_NORM > 0:
            clipped_norm = torch.nn.utils.clip_grad_norm_(
                self.shared.policy_net.parameters(), max_norm=_MAX_GRAD_NORM
            )
            self.writer.add_scalar("gradients/norm_after_clip",
                                   float(clipped_norm), self.train_steps)

        self.optimizer.step()

        # Soft update target network
        self.soft_update()

        # -----------------------
        # Metrics: loss, buffer, Q stats, TD-error
        # -----------------------
        self.writer.add_scalar("loss/train", float(loss.item()), self.train_steps)
        self.writer.add_scalar("training/learning_rate", float(LEARNING_RATE), self.train_steps)
        self.writer.add_scalar("buffer/size", int(self.shared.replay_buffer.len), self.train_steps)
        self.writer.add_scalar(
            "buffer/utilization",
            float(self.shared.replay_buffer.len) / float(self.shared.replay_buffer.size),
            self.train_steps
        )

        # Q statistics (detach to avoid graph retention)
        with torch.no_grad():
            q_mean = q_batch.mean().item()
            q_min = q_batch.min().item()
            q_max = q_batch.max().item()
            # Per-action mean Q across batch
            q_by_action = q_batch.mean(dim=0).detach().cpu().tolist()  # len = NUM_ACTIONS

            # TD error stats (|target - Q(s,a)|)
            current_q_detached = q_batch.detach().gather(1, actions.unsqueeze(1)).squeeze(1)
            td_error = (target - current_q_detached).abs()
            self.writer.add_scalar("training/td_error_mean", td_error.mean().item(), self.train_steps)
            self.writer.add_scalar("training/td_error_max",  td_error.max().item(),  self.train_steps)

            # Write Q stats to TensorBoard
            self.writer.add_scalar("q/mean", q_mean, self.train_steps)
            self.writer.add_scalar("q/min",  q_min,  self.train_steps)
            self.writer.add_scalar("q/max",  q_max,  self.train_steps)
            for i, action_name in enumerate(ACTIONS):
                self.writer.add_scalar(f"q/mean_per_action/{action_name}", q_by_action[i], self.train_steps)

        # Layer stats occasionally (heavy)
        if self.train_steps % 500 == 0:
            self.log_layer_stats()

        self.train_steps += 1

        # -----------------------
        # Console logging (loss + Qs) + best checkpoint
        # -----------------------
        if self.train_steps % _CONSOLE_LOG_EVERY_STEPS == 0:
            per_act_str = ", ".join(
                f"{ACTIONS[i]}={q_by_action[i]:.2f}" for i in range(len(q_by_action))
            )
            log_learner(
                f"[Learner] step={self.train_steps:6d} "
                f"loss={loss.item():.4f} "
                f"q_mean={q_mean:.3f} q_min={q_min:.3f} q_max={q_max:.3f} "
                f"buf={self.shared.replay_buffer.len} grad_norm={total_grad_norm:.3f}"
            )
            q0 = q_batch[0].detach().cpu().tolist()
            q0_str = ", ".join(f"{ACTIONS[i]}={q0[i]:.2f}" for i in range(len(q0)))
            log_learner(f"[Learner] Q[0]: {q0_str}")

        # Save "best" snapshot when loss improves
        if loss.item() < self.best_loss - 1e-4:
            self.best_loss = loss.item()
            self.save_checkpoint(tag="best")

        return float(loss.item())

    def log_buffer_stats(self):
        """Log replay buffer statistics"""
        buffer = self.shared.replay_buffer
        if buffer.len > 0:
            rewards_in_buffer = buffer.rewards[:buffer.len]
            self.writer.add_scalar("buffer/reward_mean",
                                   float(rewards_in_buffer.mean()), self.train_steps)
            self.writer.add_scalar("buffer/reward_std",
                                   float(rewards_in_buffer.std()), self.train_steps)
            self.writer.add_scalar("buffer/reward_min",
                                   float(rewards_in_buffer.min()), self.train_steps)
            self.writer.add_scalar("buffer/reward_max",
                                   float(rewards_in_buffer.max()), self.train_steps)

            # Log action distribution in buffer
            actions_in_buffer = buffer.actions[:buffer.len]
            for i, action_name in enumerate(ACTIONS):
                action_pct = (actions_in_buffer == i).sum() / len(actions_in_buffer)
                self.writer.add_scalar(f"buffer/action_distribution/{action_name}",
                                       float(action_pct), self.train_steps)

            # Log done percentage
            dones_in_buffer = buffer.dones[:buffer.len]
            self.writer.add_scalar("buffer/done_percentage",
                                   float(dones_in_buffer.sum()) / float(len(dones_in_buffer)),
                                   self.train_steps)

    def log_gradients(self):
        """Log gradient statistics before clipping"""
        total_grad_sq = 0.0
        for p in self.shared.policy_net.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2).item()
                total_grad_sq += param_norm ** 2
        total_grad_norm = total_grad_sq ** 0.5
        self.writer.add_scalar("gradients/norm_before_clip",
                               float(total_grad_norm), self.train_steps)
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
                                       float(param.data.mean().item()), self.train_steps)
                self.writer.add_scalar(f"weights/{name}/std",
                                       float(param.data.std().item()), self.train_steps)

    def save_checkpoint(self, tag=None):
        """Save model checkpoint (policy weights only) and print to console."""
        match_num = self.shared.match_number
        suffix = f"_{tag}" if tag else ""
        model_path = f"{MODEL_DIR}/model_match_{match_num}{suffix}.pth"

        # atomic-ish write
        tmp_path = model_path + ".tmp"
        torch.save(self.shared.policy_net.state_dict(), tmp_path)
        os.replace(tmp_path, model_path)

        msg = f"[Learner] Saved model to {model_path}"
        log_learner(msg)
        log_state(msg)  # mirror into the main log you’re tailing

        # TensorBoard breadcrumbs
        self.writer.add_scalar("training/checkpoint_step", self.train_steps, self.train_steps)
        self.writer.add_text("training/checkpoint_path", model_path, self.train_steps)
        self.writer.add_scalar("training/match_completed", int(match_num), self.train_steps)

        # Do not clear the buffer; preserve experience
        log_learner(f"[Learner] Buffer preserved with {self.shared.replay_buffer.len} samples")

    def run(self):
        """Main training loop"""
        if TEST_MODE:
            log_learner("[Learner] TEST MODE - Training disabled")
            while not self.shared.stop_event.is_set():
                # Still check for match end to save current model state
                if self.shared.match_end_event.wait(timeout=1.0):
                    self.shared.match_end_event.clear()
                    self.save_checkpoint(tag=f"match_{self.shared.match_number}")
                    log_learner(f"[Learner] Match ended in test mode")
                    self.shared.match_number += 1
            # final snapshot
            self.save_checkpoint(tag="shutdown")
            self.writer.close()
            return

        # periodic checkpoint cadence
        CKPT_EVERY_STEPS = 1000     # save every 1k train steps
        CKPT_EVERY_SEC   = 600      # and every 10 minutes
        last_time_ckpt = time.time()

        while not self.shared.stop_event.is_set():
            self.learn_tick += 1

            # Train when replay is warm and throttle by TRAIN_FREQ
            if (self.shared.replay_buffer.len >= max(LEARNING_STARTS, BATCH_SIZE) and
                self.learn_tick % TRAIN_FREQ == 0):
                try:
                    self.train_step()
                except Exception as e:
                    import traceback
                    log_state(f"[Learner] CRASH in train_step: {e}\n{traceback.format_exc()}")
                    time.sleep(1)

                # step-based checkpoint
                if self.train_steps > 0 and self.train_steps % CKPT_EVERY_STEPS == 0:
                    self.save_checkpoint(tag=f"step_{self.train_steps}")

            # Match-end based checkpoint
            if self.shared.match_end_event.wait(timeout=0.01):
                self.shared.match_end_event.clear()
                self.save_checkpoint(tag=f"match_{self.shared.match_number}")
                self.shared.match_number += 1

            # time-based checkpoint
            now = time.time()
            if now - last_time_ckpt >= CKPT_EVERY_SEC:
                self.save_checkpoint(tag=f"time_{int(now)}")
                last_time_ckpt = now

            # Small sleep to prevent CPU spinning
            time.sleep(0.001)

        # final snapshot on exit
        self.save_checkpoint(tag="shutdown")
        self.writer.close()
        log_learner("[Learner] Training stopped")
