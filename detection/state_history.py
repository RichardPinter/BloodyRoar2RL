# state_history.py
from collections import deque
from typing import List, Optional
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from detection.game_state import GameState, PlayerState

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

class FrameGroupAnalyzer:
    """Analyzes groups of 20 frames and calculates state metrics"""
    
    def __init__(self):
        self.frame_group_size = 20
        
    def analyze_frame_group(self, states: List[GameState]) -> dict:
        """
        Analyze a group of 20 frames and calculate various state values
        
        Args:
            states: List of GameState objects (should be 20 frames)
            
        Returns:
            Dictionary with calculated metrics
        """
        if len(states) != self.frame_group_size:
            raise ValueError(f"Expected {self.frame_group_size} states, got {len(states)}")
            
        # Health analysis
        health_metrics = self._analyze_health(states)
        
        # Position/movement analysis
        position_metrics = self._analyze_positions(states)
        
        # Combat metrics
        combat_metrics = self._analyze_combat(states)
        
        # Overall frame group summary
        summary = {
            'frame_count': len(states),
            'time_span': states[-1].frame_time - states[0].frame_time if states[0].frame_time and states[-1].frame_time else 0,
            **health_metrics,
            **position_metrics,
            **combat_metrics
        }
        
        return summary
    
    def _analyze_health(self, states: List[GameState]) -> dict:
        """Analyze health changes across frame group"""
        p1_healths = [s.player1.health for s in states]
        p2_healths = [s.player2.health for s in states]
        
        return {
            'p1_health_start': p1_healths[0],
            'p1_health_end': p1_healths[-1],
            'p1_health_change': p1_healths[-1] - p1_healths[0],
            'p1_health_min': min(p1_healths),
            'p1_health_max': max(p1_healths),
            'p1_health_avg': np.mean(p1_healths),
            
            'p2_health_start': p2_healths[0],
            'p2_health_end': p2_healths[-1],
            'p2_health_change': p2_healths[-1] - p2_healths[0],
            'p2_health_min': min(p2_healths),
            'p2_health_max': max(p2_healths),
            'p2_health_avg': np.mean(p2_healths),
            
            'health_advantage_start': p1_healths[0] - p2_healths[0],
            'health_advantage_end': p1_healths[-1] - p2_healths[-1],
            'health_advantage_change': (p1_healths[-1] - p2_healths[-1]) - (p1_healths[0] - p2_healths[0])
        }
    
    def _analyze_positions(self, states: List[GameState]) -> dict:
        """Analyze position and movement across frame group"""
        p1_x_positions = [s.player1.x for s in states]
        p1_y_positions = [s.player1.y for s in states]
        p2_x_positions = [s.player2.x for s in states]
        p2_y_positions = [s.player2.y for s in states]
        distances = [s.distance for s in states]
        
        # Calculate movement distances
        p1_total_movement = sum(
            np.sqrt((p1_x_positions[i] - p1_x_positions[i-1])**2 + 
                   (p1_y_positions[i] - p1_y_positions[i-1])**2)
            for i in range(1, len(states))
        )
        
        p2_total_movement = sum(
            np.sqrt((p2_x_positions[i] - p2_x_positions[i-1])**2 + 
                   (p2_y_positions[i] - p2_y_positions[i-1])**2)
            for i in range(1, len(states))
        )
        
        return {
            'p1_x_start': p1_x_positions[0],
            'p1_x_end': p1_x_positions[-1],
            'p1_x_change': p1_x_positions[-1] - p1_x_positions[0],
            'p1_y_start': p1_y_positions[0],
            'p1_y_end': p1_y_positions[-1],
            'p1_y_change': p1_y_positions[-1] - p1_y_positions[0],
            'p1_total_movement': p1_total_movement,
            
            'p2_x_start': p2_x_positions[0],
            'p2_x_end': p2_x_positions[-1],
            'p2_x_change': p2_x_positions[-1] - p2_x_positions[0],
            'p2_y_start': p2_y_positions[0],
            'p2_y_end': p2_y_positions[-1],
            'p2_y_change': p2_y_positions[-1] - p2_y_positions[0],
            'p2_total_movement': p2_total_movement,
            
            'distance_start': distances[0],
            'distance_end': distances[-1],
            'distance_change': distances[-1] - distances[0],
            'distance_min': min(distances),
            'distance_max': max(distances),
            'distance_avg': np.mean(distances),
            
            'players_approaching': distances[-1] < distances[0],
            'players_separating': distances[-1] > distances[0]
        }
    
    def _analyze_combat(self, states: List[GameState]) -> dict:
        """Analyze combat-related metrics"""
        # Count frames where health changed (potential hits)
        p1_hit_frames = sum(1 for i in range(1, len(states)) 
                           if states[i].player1.health < states[i-1].player1.health)
        p2_hit_frames = sum(1 for i in range(1, len(states)) 
                           if states[i].player2.health < states[i-1].player2.health)
        
        # Calculate average velocities
        p1_velocities_x = [s.player1.velocity_x for s in states if hasattr(s.player1, 'velocity_x')]
        p1_velocities_y = [s.player1.velocity_y for s in states if hasattr(s.player1, 'velocity_y')]
        p2_velocities_x = [s.player2.velocity_x for s in states if hasattr(s.player2, 'velocity_x')]
        p2_velocities_y = [s.player2.velocity_y for s in states if hasattr(s.player2, 'velocity_y')]
        
        return {
            'p1_hit_frames': p1_hit_frames,
            'p2_hit_frames': p2_hit_frames,
            'total_hit_frames': p1_hit_frames + p2_hit_frames,
            
            'p1_avg_velocity_x': np.mean(p1_velocities_x) if p1_velocities_x else 0,
            'p1_avg_velocity_y': np.mean(p1_velocities_y) if p1_velocities_y else 0,
            'p2_avg_velocity_x': np.mean(p2_velocities_x) if p2_velocities_x else 0,
            'p2_avg_velocity_y': np.mean(p2_velocities_y) if p2_velocities_y else 0,
            
            'p1_max_speed': max([np.sqrt(vx**2 + vy**2) for vx, vy in zip(p1_velocities_x, p1_velocities_y)]) if p1_velocities_x else 0,
            'p2_max_speed': max([np.sqrt(vx**2 + vy**2) for vx, vy in zip(p2_velocities_x, p2_velocities_y)]) if p2_velocities_x else 0,
            
            'combat_intensity': (p1_hit_frames + p2_hit_frames) / len(states),
            'p1_aggression': p1_hit_frames / len(states),
            'p2_aggression': p2_hit_frames / len(states)
        }