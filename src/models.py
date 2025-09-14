"""
Neural network models and replay buffer for the RL agent.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.config import DEVICE, FRAME_STACK, CNN_SIZE, NUM_ACTIONS

class DQNNet(nn.Module):
    """Deep Q-Network architecture"""
    def __init__(self, in_ch, n_actions, extra_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, 32, 8, 4)
        self.conv2 = nn.Conv2d(32, 64, 4, 2)
        self.conv3 = nn.Conv2d(64, 64, 3, 1)
        conv_out = 64 * 7 * 7
        self.fc1 = nn.Linear(conv_out + extra_dim, 512)
        self.out = nn.Linear(512, n_actions)

    def forward(self, x, extra):
        # x: (B, in_ch, H, W), extra: (B, extra_dim)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.flatten(1)                   # (B, conv_out)
        x = torch.cat([x, extra], dim=1)   # (B, conv_out+extra_dim)
        x = F.relu(self.fc1(x))
        return self.out(x)

class ReplayBuffer:
    """Experience replay buffer for DQN training"""
    def __init__(self, size, extra_dim):
        self.size = size
        self.extra_dim = extra_dim
        self.clear()

    def clear(self):
        # Store frames as uint8 for 4x memory savings
        self.states = np.zeros((self.size, FRAME_STACK, *CNN_SIZE), dtype=np.uint8)
        self.next_states = np.zeros((self.size, FRAME_STACK, *CNN_SIZE), dtype=np.uint8)
        self.extras = np.zeros((self.size, self.extra_dim), dtype=np.float32)
        self.next_extras = np.zeros((self.size, self.extra_dim), dtype=np.float32)
        self.actions = np.zeros(self.size, dtype=np.int64)
        self.rewards = np.zeros(self.size, dtype=np.float32)
        self.dones = np.zeros(self.size, dtype=bool)
        self.ptr = 0
        self.len = 0

    def add(self, s, extra, a, r, s2, next_extra, done):
        # s, s2 expected as float32 in [0,1]; store as uint8
        self.states[self.ptr] = (s * 255.0).astype(np.uint8)
        self.next_states[self.ptr] = (s2 * 255.0).astype(np.uint8)
        self.extras[self.ptr] = extra
        self.next_extras[self.ptr] = next_extra
        self.actions[self.ptr] = a
        self.rewards[self.ptr] = r
        self.dones[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.size
        self.len = min(self.len + 1, self.size)

    def sample(self, batch_size):
        idx = np.random.randint(0, self.len, size=batch_size)
        # Convert back to float in [0,1] on GPU/CPU
        states = torch.from_numpy(self.states[idx]).to(DEVICE).float().div_(255.0)
        next_states = torch.from_numpy(self.next_states[idx]).to(DEVICE).float().div_(255.0)
        extras = torch.from_numpy(self.extras[idx]).to(DEVICE)
        next_extras = torch.from_numpy(self.next_extras[idx]).to(DEVICE)
        actions = torch.from_numpy(self.actions[idx]).to(DEVICE)
        rewards = torch.from_numpy(self.rewards[idx]).to(DEVICE)
        dones = torch.from_numpy(self.dones[idx].astype(np.uint8)).to(DEVICE)
        return states, extras, actions, rewards, next_states, next_extras, dones