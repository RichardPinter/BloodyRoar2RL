#!/usr/bin/env python3
"""
Shared state management for inter-thread communication.
Centralizes all shared resources and synchronization.
"""
import threading
from queue import Queue
from collections import deque
from src.config import FRAME_STACK, REPLAY_SIZE, EXTRA_DIM, DEVICE
from src.models import DQNNet, ReplayBuffer

class SharedState:
    """Manages all shared state between threads"""
    
    def __init__(self):
        # Queues
        self.frame_queue = Queue(maxsize=16)
        
        # Events for synchronization
        self.stop_event = threading.Event()
        self.round_end_event = threading.Event()
        self.match_end_event = threading.Event()
        
        # Shared counters with locks
        self._lock = threading.Lock()
        self._match_number = 1
        self._global_step = 0
        self._episode_number = 0
        
        # Neural networks
        self.policy_net = DQNNet(FRAME_STACK, ACTIONS_LEN, EXTRA_DIM).to(DEVICE)
        self.target_net = DQNNet(FRAME_STACK, ACTIONS_LEN, EXTRA_DIM).to(DEVICE)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer(REPLAY_SIZE, EXTRA_DIM)
        
        # Frame stack for current episode
        self.frame_stack = deque(maxlen=FRAME_STACK)
        
        # Results storage
        self.results = []
        
    # Thread-safe property accessors
    @property
    def match_number(self):
        with self._lock:
            return self._match_number
    
    @match_number.setter
    def match_number(self, value):
        with self._lock:
            self._match_number = value
    
    @property
    def global_step(self):
        with self._lock:
            return self._global_step
    
    def increment_global_step(self):
        with self._lock:
            self._global_step += 1
            return self._global_step
    
    @property
    def episode_number(self):
        with self._lock:
            return self._episode_number
    
    def increment_episode_number(self):
        with self._lock:
            self._episode_number += 1
            return self._episode_number
    
    def signal_round_end(self):
        """Signal that a round has ended"""
        self.round_end_event.set()
    
    def signal_match_end(self):
        """Signal that a match has ended"""
        self.match_end_event.set()
    
    def shutdown(self):
        """Signal all threads to stop"""
        self.stop_event.set()

# Import after SharedState definition to avoid circular import
from src.config import ACTIONS
ACTIONS_LEN = len(ACTIONS)