import time
import numpy as np
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from br2_env import BR2Environment
from game_state import GameState

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
        
        # Round state
        self.stats = None
        self.is_active = False
        self.max_round_time = 120.0  # 2 minutes max per round
        
        # Health thresholds for determining round end
        self.health_threshold = 5.0  # Health below this = round over
        
        print("RoundSubEpisode initialized")
    
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
        
        # Mark as active
        self.is_active = True
        
        print(f"Round started - P1: {self.env.current_state.player1.health:.1f}%, P2: {self.env.current_state.player2.health:.1f}%")
        
        return observation
    
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
        
        # Update round stats
        self.stats.steps_taken += 1
        self.stats.total_reward += reward
        
        # Check if round should end
        round_done, outcome = self._check_round_end(env_done, info)
        
        # Update stats if round is finished
        if round_done:
            self._finish_round(outcome, info)
        
        # Update info with round-specific data
        round_info = self._get_round_info(info)
        
        return observation, reward, round_done, round_info
    
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