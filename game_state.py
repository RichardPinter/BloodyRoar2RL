# game_state.py
from dataclasses import dataclass
from typing import Optional, List
import numpy as np
import time

@dataclass
class PlayerState:
    """State information for a single player"""
    # Position
    x: float = 0.0
    y: float = 0.0
    
    # Health
    health: float = 100.0
    
    # Derived info (calculated)
    velocity_x: float = 0.0
    velocity_y: float = 0.0
    health_delta: float = 0.0  # Health change since last frame
    
    def to_array(self) -> np.ndarray:
        """Convert to numpy array for RL"""
        return np.array([
            self.x, self.y, 
            self.health,
            self.velocity_x, self.velocity_y,
            self.health_delta
        ])

@dataclass
class GameState:
    """Complete game state at a single point in time"""
    # Players
    player1: PlayerState
    player2: PlayerState
    
    # Global info
    distance: float = 0.0
    frame_time: float = 0.0
    
    # Screen dimensions (for normalization)
    screen_width: int = 1920
    screen_height: int = 1080
    
    def __post_init__(self):
        """Calculate derived values"""
        if self.player1 and self.player2:
            # Calculate distance if not provided
            if self.distance == 0.0:
                dx = self.player2.x - self.player1.x
                dy = self.player2.y - self.player1.y
                self.distance = np.sqrt(dx**2 + dy**2)
    
    def to_relative_state(self, player_perspective: int = 1) -> dict:
        """
        Convert to relative state from a player's perspective
        
        Args:
            player_perspective: 1 or 2 (which player's POV)
            
        Returns:
            Dictionary with relative measurements
        """
        if player_perspective == 1:
            agent = self.player1
            opponent = self.player2
        else:
            agent = self.player2
            opponent = self.player1
            
        return {
            # Agent absolute info
            'agent_health': agent.health,
            'agent_x': agent.x,
            'agent_y': agent.y,
            
            # Opponent relative to agent
            'opponent_relative_x': opponent.x - agent.x,
            'opponent_relative_y': opponent.y - agent.y,
            'opponent_health': opponent.health,
            
            # Interaction
            'distance': self.distance,
            
            # Velocities
            'agent_velocity_x': agent.velocity_x,
            'agent_velocity_y': agent.velocity_y,
            'opponent_velocity_x': opponent.velocity_x,
            'opponent_velocity_y': opponent.velocity_y,
        }