# game_state_monitor.py
import time
import numpy as np
from typing import Optional, Tuple
from dataclasses import dataclass

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from detection.window_capture import WindowCapture
from detection.health_detector import HealthDetector
from detection.fighter_detector import FighterDetector
from detection.game_state import GameState, PlayerState
from state_normalizer import StateNormalizer
from detection.state_history import StateHistory

class GameStateMonitor:
    """Main orchestrator that combines all detection systems"""
    
    def __init__(self, window_title: str):
        # Initialize capture
        self.capture = WindowCapture(window_title)
        if not self.capture.is_valid:
            raise RuntimeError(f"Window '{window_title}' not found!")
        
        # Initialize detectors
        self.health_detector = HealthDetector()
        self.fighter_detector = FighterDetector()
        
        # State management
        self.state_normalizer = StateNormalizer()
        self.state_history = StateHistory(history_size=5)
        
        # Timing
        self.last_capture_time = 0
        
    def capture_state(self) -> Optional[GameState]:
        """Capture current game state"""
        # Single frame capture
        frame = self.capture.capture()
        if frame is None:
            return None
            
        current_time = time.time()
        
        # Run detections in parallel (they use the same frame)
        health_state = self.health_detector.detect(self.capture)
        fighter_detection = self.fighter_detector.detect(frame)
        
        # Check if we have valid detections
        if not health_state or not fighter_detection.player1 or not fighter_detection.player2:
            return None
        
        # Create game state
        game_state = GameState(
            player1=PlayerState(
                x=fighter_detection.player1.center[0],
                y=fighter_detection.player1.center[1],
                health=health_state.p1_health
            ),
            player2=PlayerState(
                x=fighter_detection.player2.center[0],
                y=fighter_detection.player2.center[1],
                health=health_state.p2_health
            ),
            distance=fighter_detection.distance if fighter_detection.distance is not None else 0.0,
            frame_time=current_time
        )
        
        # Add to history and calculate velocities
        self.state_history.add_state(game_state, current_time)
        game_state = self.state_history.calculate_velocities()
        
        self.last_capture_time = current_time
        return game_state
    
    def get_normalized_observation(self, player_perspective: int = 1) -> Optional[np.ndarray]:
        """Get normalized observation for RL agent"""
        state = self.state_history.get_latest()
        if state is None:
            return None
            
        return self.state_normalizer.normalize_state(state, player_perspective)
    
    def get_temporal_features(self) -> np.ndarray:
        """Get temporal features from history"""
        return self.state_history.get_temporal_features()
    
    def get_raw_frame(self) -> Optional[np.ndarray]:
        """Get the last captured frame for visualization"""
        return self.capture.capture()
    
    def visualize_state(self, frame: np.ndarray, game_state: GameState) -> np.ndarray:
        """Combine visualizations from both detectors"""
        import cv2
        
        # Let health detector add its visualization
        display = self.health_detector.visualize_health_bars(frame, 
            self.health_detector.detect(self.capture))
        
        # Let fighter detector add its visualization
        fighter_detection = self.fighter_detector.detect(frame)
        display = self.fighter_detector.visualize_fighters(display, fighter_detection)
        
        # Add state info at the top
        info_text = (f"Distance: {game_state.distance:.0f} | "
                    f"P1 Vel: ({game_state.player1.velocity_x:.1f}, {game_state.player1.velocity_y:.1f}) | "
                    f"P2 Vel: ({game_state.player2.velocity_x:.1f}, {game_state.player2.velocity_y:.1f})")
        cv2.putText(display, info_text, (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return display