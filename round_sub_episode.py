import time
import numpy as np
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from br2_env import BR2Environment
from game_state import GameState
from window_capture import WindowCapture
from health_detector import HealthDetector, HealthBarConfig

class RoundOutcome(Enum):
    """Possible outcomes of a round"""
    ONGOING = "ongoing"
    PLAYER_WIN = "player_win"
    PLAYER_LOSS = "player_loss"
    DRAW = "draw"
    TIMEOUT = "timeout"
    ERROR = "error"

@dataclass
class RoundStats:
    """Statistics for a single round"""
    start_time: float
    end_time: Optional[float] = None
    steps_taken: int = 0
    total_reward: float = 0.0
    final_p1_health: float = 0.0
    final_p2_health: float = 0.0
    outcome: RoundOutcome = RoundOutcome.ONGOING
    
    @property
    def duration(self) -> float:
        """Round duration in seconds"""
        if self.end_time is None:
            return time.time() - self.start_time
        return self.end_time - self.start_time
    
    @property
    def is_finished(self) -> bool:
        """Check if round is finished"""
        return self.outcome != RoundOutcome.ONGOING

class RoundSubEpisode:
    """
    Manages a single round of fighting within an arcade episode.
    Wraps the BR2Environment and handles round-specific logic.
    """
    
    def __init__(self, window_title: str = "Bloody Roar II (USA) [PlayStation] - BizHawk"):
        # Initialize the underlying environment
        self.env = BR2Environment(window_title)
        self.window_title = window_title
        
        # Initialize health detection
        try:
            self.capture = WindowCapture(window_title)
            self.health_detector = HealthDetector()
            self.health_detection_available = True
        except Exception as e:
            print(f"Warning: Health detection not available: {e}")
            self.health_detection_available = False
        
        # Round state
        self.stats = None
        self.is_active = False
        self.max_round_time = 120.0  # 2 minutes max per round
        
        # Win detection state
        self.p1_zero_frames = 0
        self.p2_zero_frames = 0
        self.zero_threshold = 10  # Consecutive zero frames needed for death
        self.p1_health_pct = 0.0
        self.p2_health_pct = 0.0
        
        print("RoundSubEpisode initialized with health detection")
    
    def reset(self) -> np.ndarray:
        """
        Reset the round and start a new one
        
        Returns:
            Initial observation
        """
        print("Starting new round...")
        
        # Reset the underlying environment
        observation = self.env.reset()
        
        # Initialize round stats
        self.stats = RoundStats(
            start_time=time.time(),
            steps_taken=0,
            total_reward=0.0
        )
        
        # Reset win detection state
        self.p1_zero_frames = 0
        self.p2_zero_frames = 0
        self.p1_health_pct = 0.0
        self.p2_health_pct = 0.0
        
        # Mark as active
        self.is_active = True
        
        print(f"Round started - P1: {self.env.current_state.player1.health:.1f}%, P2: {self.env.current_state.player2.health:.1f}%")
        
        return observation
    
    def _detect_health_percentages(self) -> Tuple[float, float]:
        """
        Detect health percentages using pixel-based detection
        
        Returns:
            (p1_health_pct, p2_health_pct)
        """
        if not self.health_detection_available:
            # Fallback to BR2Environment health if available
            if self.env.current_state:
                return self.env.current_state.player1.health, self.env.current_state.player2.health
            return 0.0, 0.0
        
        try:
            # Capture health bar regions
            config = self.health_detector.config
            
            p1_strip = self.capture.capture_region(
                x=config.p1_x, y=config.bar_y,
                width=config.bar_length, height=config.bar_height
            )
            p2_strip = self.capture.capture_region(
                x=config.p2_x - config.bar_length, y=config.bar_y,
                width=config.bar_length, height=config.bar_height
            )
            
            if p1_strip is None or p2_strip is None:
                return 0.0, 0.0
            
            # Calculate yellow pixel percentages (same logic as test script)
            p1_pct, _ = self._calculate_yellow_percentage(p1_strip, config)
            p2_pct, _ = self._calculate_yellow_percentage(p2_strip, config)
            
            return p1_pct, p2_pct
            
        except Exception as e:
            print(f"Health detection error: {e}")
            return 0.0, 0.0
    
    def _calculate_yellow_percentage(self, pixel_strip, config):
        """Calculate percentage of yellow pixels in health bar strip"""
        if pixel_strip is None:
            return 0.0, 0
        
        # Handle different array shapes
        if len(pixel_strip.shape) == 3:
            if pixel_strip.shape[0] == 1:
                pixel_strip = pixel_strip[0]
            elif pixel_strip.shape[2] == 3:
                pixel_strip = pixel_strip.reshape(-1, 3)
        
        # Extract BGR channels
        b = pixel_strip[:, 0]
        g = pixel_strip[:, 1]
        r = pixel_strip[:, 2]
        
        # Create mask for yellow pixels
        yellow_mask = (
            (r >= config.lower_bgr[2]) & (r <= config.upper_bgr[2]) &
            (g >= config.lower_bgr[1]) & (g <= config.upper_bgr[1]) &
            (b >= config.lower_bgr[0]) & (b <= config.upper_bgr[0])
        )
        
        # Count yellow pixels
        yellow_count = np.sum(yellow_mask)
        total_pixels = len(pixel_strip)
        
        # Calculate percentage
        if total_pixels > 0:
            percentage = (yellow_count / total_pixels) * 100.0
        else:
            percentage = 0.0
        
        return percentage, yellow_count
    
    def _update_win_detection(self):
        """Update consecutive zero-frame counters for win detection"""
        # Check P1 health
        if self.p1_health_pct <= 0.0:
            self.p1_zero_frames += 1
        else:
            self.p1_zero_frames = 0
        
        # Check P2 health
        if self.p2_health_pct <= 0.0:
            self.p2_zero_frames += 1
        else:
            self.p2_zero_frames = 0
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        Take a step in the round
        
        Args:
            action: Action to take
            
        Returns:
            observation, reward, done, info
        """
        if not self.is_active:
            raise RuntimeError("Round is not active. Call reset() first.")
        
        # Take step in underlying environment
        observation, reward, env_done, info = self.env.step(action)
        
        # Detect health percentages using pixel detection
        self.p1_health_pct, self.p2_health_pct = self._detect_health_percentages()
        
        # Update win detection counters
        self._update_win_detection()
        
        # Update round stats
        self.stats.steps_taken += 1
        self.stats.total_reward += reward
        
        # Check if round should end (using our win detection instead of env)
        round_done, outcome = self._check_round_end_with_win_detection(env_done, info)
        
        # Update stats if round is finished
        if round_done:
            self._finish_round(outcome, info)
        
        # Update info with round-specific data
        round_info = self._get_round_info(info)
        
        return observation, reward, round_done, round_info
    
    def _check_round_end_with_win_detection(self, env_done: bool, info: Dict[str, Any]) -> Tuple[bool, RoundOutcome]:
        """
        Check if round should end using win detection logic
        
        Args:
            env_done: Whether the underlying environment says it's done
            info: Info from the environment step
            
        Returns:
            (is_done, outcome)
        """
        # Check timeout
        if self.stats.duration > self.max_round_time:
            return True, RoundOutcome.TIMEOUT
        
        # Check win conditions using consecutive zero frames
        p1_dead = self.p1_zero_frames >= self.zero_threshold
        p2_dead = self.p2_zero_frames >= self.zero_threshold
        
        if p1_dead and p2_dead:
            return True, RoundOutcome.DRAW
        elif p1_dead:
            return True, RoundOutcome.PLAYER_LOSS
        elif p2_dead:
            return True, RoundOutcome.PLAYER_WIN
        
        # Round continues
        return False, RoundOutcome.ONGOING
    
    def _check_round_end(self, env_done: bool, info: Dict[str, Any]) -> Tuple[bool, RoundOutcome]:
        """
        Check if the round should end and determine outcome
        
        Args:
            env_done: Whether the underlying environment says it's done
            info: Info from the environment step
            
        Returns:
            (is_done, outcome)
        """
        # Check timeout
        if self.stats.duration > self.max_round_time:
            return True, RoundOutcome.TIMEOUT
        
        # Check if environment says it's done
        if env_done:
            # Determine outcome based on health
            if self.env.current_state is not None:
                p1_health = self.env.current_state.player1.health
                p2_health = self.env.current_state.player2.health
                
                if p1_health <= self.health_threshold and p2_health <= self.health_threshold:
                    return True, RoundOutcome.DRAW
                elif p1_health <= self.health_threshold:
                    return True, RoundOutcome.PLAYER_LOSS
                elif p2_health <= self.health_threshold:
                    return True, RoundOutcome.PLAYER_WIN
                else:
                    # Environment ended but unclear why - check who has more health
                    if p1_health > p2_health:
                        return True, RoundOutcome.PLAYER_WIN
                    elif p2_health > p1_health:
                        return True, RoundOutcome.PLAYER_LOSS
                    else:
                        return True, RoundOutcome.DRAW
            else:
                # No valid state - assume error
                return True, RoundOutcome.ERROR
        
        # Round continues
        return False, RoundOutcome.ONGOING
    
    def _finish_round(self, outcome: RoundOutcome, info: Dict[str, Any]):
        """
        Finish the round and update final stats
        
        Args:
            outcome: How the round ended
            info: Final step info
        """
        self.stats.end_time = time.time()
        self.stats.outcome = outcome
        
        # Store final health values
        if self.env.current_state is not None:
            self.stats.final_p1_health = self.env.current_state.player1.health
            self.stats.final_p2_health = self.env.current_state.player2.health
        
        # Mark as inactive
        self.is_active = False
        
        # Print round summary
        print(f"Round finished: {outcome.value}")
        print(f"  Duration: {self.stats.duration:.1f}s")
        print(f"  Steps: {self.stats.steps_taken}")
        print(f"  Total reward: {self.stats.total_reward:.2f}")
        print(f"  Final health - P1: {self.stats.final_p1_health:.1f}%, P2: {self.stats.final_p2_health:.1f}%")
    
    def _get_round_info(self, env_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create round-specific info dictionary
        
        Args:
            env_info: Info from environment step
            
        Returns:
            Enhanced info dictionary
        """
        round_info = env_info.copy()
        
        # Add round-specific information
        round_info.update({
            'round_active': self.is_active,
            'round_steps': self.stats.steps_taken if self.stats else 0,
            'round_duration': self.stats.duration if self.stats else 0.0,
            'round_total_reward': self.stats.total_reward if self.stats else 0.0,
            'round_outcome': self.stats.outcome.value if self.stats else RoundOutcome.ONGOING.value,
            # Win detection info
            'p1_health_percentage': self.p1_health_pct,
            'p2_health_percentage': self.p2_health_pct,
            'p1_zero_frames': self.p1_zero_frames,
            'p2_zero_frames': self.p2_zero_frames,
            'zero_threshold': self.zero_threshold,
            'win_detection_active': self.health_detection_available,
        })
        
        return round_info
    
    def get_stats(self) -> Optional[RoundStats]:
        """Get current round statistics"""
        return self.stats
    
    def is_round_active(self) -> bool:
        """Check if round is currently active"""
        return self.is_active
    
    def close(self):
        """Clean up resources"""
        if hasattr(self, 'env'):
            self.env.close()
        print("RoundSubEpisode closed")
    
    def __del__(self):
        """Cleanup on deletion"""
        self.close()

# Test function
if __name__ == "__main__":
    print("Testing RoundSubEpisode...")
    
    round_episode = RoundSubEpisode()
    
    try:
        # Test a single round
        obs = round_episode.reset()
        print(f"Initial observation shape: {obs.shape}")
        
        # Take a few random actions
        for i in range(10):
            action = np.random.randint(0, round_episode.env.action_space.n)
            obs, reward, done, info = round_episode.step(action)
            
            print(f"Step {i+1}: Action {action}, Reward {reward:.2f}, Done {done}")
            
            if done:
                print(f"Round ended after {i+1} steps")
                break
        
        # Print final stats
        stats = round_episode.get_stats()
        if stats:
            print(f"Final stats: {stats}")
    
    except Exception as e:
        print(f"Error during test: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        round_episode.close()