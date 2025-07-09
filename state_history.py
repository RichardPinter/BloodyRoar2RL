# state_history.py
from collections import deque
from typing import List, Optional
import numpy as np
from game_state import GameState, PlayerState

class StateHistory:
    """Manages game state history for temporal features"""
    
    def __init__(self, history_size: int = 5):
        self.history_size = history_size
        self.states: deque[GameState] = deque(maxlen=history_size)
        self.timestamps: deque[float] = deque(maxlen=history_size)
        
    def add_state(self, state: GameState, timestamp: float):
        """Add a new state to history"""
        self.states.append(state)
        self.timestamps.append(timestamp)
        
    def get_latest(self) -> Optional[GameState]:
        """Get the most recent state"""
        return self.states[-1] if self.states else None
    
    def calculate_velocities(self) -> Optional[GameState]:
        """
        Calculate velocities based on position history
        Returns updated latest state with velocities
        """
        if len(self.states) < 2:
            return self.get_latest()
            
        # Get last two states
        prev_state = self.states[-2]
        curr_state = self.states[-1]
        
        # Calculate time delta
        dt = self.timestamps[-1] - self.timestamps[-2]
        if dt <= 0:
            return curr_state
            
        # Calculate velocities for player 1
        curr_state.player1.velocity_x = (curr_state.player1.x - prev_state.player1.x) / dt
        curr_state.player1.velocity_y = (curr_state.player1.y - prev_state.player1.y) / dt
        curr_state.player1.health_delta = curr_state.player1.health - prev_state.player1.health
        
        # Calculate velocities for player 2
        curr_state.player2.velocity_x = (curr_state.player2.x - prev_state.player2.x) / dt
        curr_state.player2.velocity_y = (curr_state.player2.y - prev_state.player2.y) / dt
        curr_state.player2.health_delta = curr_state.player2.health - prev_state.player2.health
        
        return curr_state
    
    def get_temporal_features(self) -> np.ndarray:
        """
        Extract temporal features from history
        Returns array of temporal information
        """
        if len(self.states) < 2:
            return np.zeros(4)  # Return zeros if not enough history
            
        features = []
        
        # Average velocity over history
        avg_p1_vx = np.mean([s.player1.velocity_x for s in self.states])
        avg_p2_vx = np.mean([s.player2.velocity_x for s in self.states])
        
        # Distance trend (closing/separating)
        distances = [s.distance for s in self.states]
        distance_trend = (distances[-1] - distances[0]) / len(distances)
        
        # Health trends
        p1_health_trend = (self.states[-1].player1.health - self.states[0].player1.health) / len(self.states)
        p2_health_trend = (self.states[-1].player2.health - self.states[0].player2.health) / len(self.states)
        
        return np.array([
            avg_p1_vx / 50.0,  # Normalized
            avg_p2_vx / 50.0,
            distance_trend / 100.0,
            (p1_health_trend - p2_health_trend) / 10.0  # Health advantage trend
        ])