import numpy as np
import torch
from config import FRAME_STACK, CNN_SIZE, REPLAY_SIZE, DEVICE, BATCH_SIZE

class ReplayBuffer:
    def __init__(self, size: int = REPLAY_SIZE):
        """
        Circular buffer for storing and sampling experience tuples.
        """
        # Preallocate memory
        self.states      = np.zeros((size, FRAME_STACK, *CNN_SIZE), dtype=np.float32)
        self.actions     = np.zeros(size, dtype=np.int64)
        self.rewards     = np.zeros(size, dtype=np.float32)
        self.next_states = np.zeros((size, FRAME_STACK, *CNN_SIZE), dtype=np.float32)
        self.dones       = np.zeros(size, dtype=np.bool_)

        self.max_size    = size
        self.ptr         = 0
        self.len         = 0

    def add(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ) -> None:
        """
        Add a new experience to the buffer.
        """
        self.states[self.ptr]      = state
        self.actions[self.ptr]     = action
        self.rewards[self.ptr]     = reward
        self.next_states[self.ptr] = next_state
        self.dones[self.ptr]       = done

        # Advance pointer and update length
        self.ptr = (self.ptr + 1) % self.max_size
        self.len = min(self.len + 1, self.max_size)

    def sample(
        self,
        batch_size: int = BATCH_SIZE
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Uniformly sample a batch of experiences.
        Returns:
            states, actions, rewards, next_states, dones
        """
        idx = np.random.randint(0, self.len, size=batch_size)

        # Convert numpy arrays to torch tensors on DEVICE
        states      = torch.from_numpy(self.states[idx]).to(DEVICE)
        actions     = torch.from_numpy(self.actions[idx]).to(DEVICE)
        rewards     = torch.from_numpy(self.rewards[idx]).to(DEVICE)
        next_states = torch.from_numpy(self.next_states[idx]).to(DEVICE)
        dones       = torch.from_numpy(self.dones[idx].astype(np.uint8)).to(DEVICE)

        return states, actions, rewards, next_states, dones
