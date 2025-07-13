# state_normalizer.py
import numpy as np
from dataclasses import dataclass
from typing import Optional
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from detection.game_state import GameState

@dataclass
class NormalizationConfig:
    """Configuration for state normalization"""
    screen_width: int = 1920
    screen_height: int = 1080
    max_distance: float = 1920  # Maximum relevant distance
    max_velocity: float = 50.0   # Maximum velocity in pixels/frame

class StateNormalizer:
    """Normalizes game state for neural network input"""
    
    def __init__(self, config: Optional[NormalizationConfig] = None):
        self.config = config or NormalizationConfig()
        
    def normalize_state(self, game_state: GameState, 
                       player_perspective: int = 1) -> np.ndarray:
        """
        Convert game state to normalized RL observation
        
        Returns:
            Normalized state vector ready for neural network
        """
        # Get relative state
        rel_state = game_state.to_relative_state(player_perspective)
        
        # Create normalized observation vector
        observation = np.array([
            # Agent info (5 values)
            rel_state['agent_health'] / 100.0,  # [0, 1]
            rel_state['agent_x'] / self.config.screen_width,  # [0, 1]
            rel_state['agent_y'] / self.config.screen_height,  # [0, 1]
            np.clip(rel_state['agent_velocity_x'] / self.config.max_velocity, -1, 1),
            np.clip(rel_state['agent_velocity_y'] / self.config.max_velocity, -1, 1),
            
            # Opponent info (5 values)
            rel_state['opponent_health'] / 100.0,  # [0, 1]
            np.clip(rel_state['opponent_relative_x'] / self.config.max_distance, -1, 1),
            np.clip(rel_state['opponent_relative_y'] / 300, -1, 1),  # Smaller Y range
            np.clip(rel_state['opponent_velocity_x'] / self.config.max_velocity, -1, 1),
            np.clip(rel_state['opponent_velocity_y'] / self.config.max_velocity, -1, 1),
            
            # Interaction (2 values)
            min(rel_state['distance'] / self.config.max_distance, 1.0) if rel_state['distance'] is not None else 0.5,  # [0, 1]
            1.0 if rel_state['opponent_relative_x'] > 0 else -1.0,  # Facing direction
        ], dtype=np.float32)
        
        return observation
    
    def get_observation_space_size(self) -> int:
        """Return the size of the observation vector"""
        return 12  # Based on the observation vector above