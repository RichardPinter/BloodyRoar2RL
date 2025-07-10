import time
import numpy as np
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from enum import Enum

from round_sub_episode import RoundSubEpisode, RoundOutcome, RoundStats

class ArcadeOutcome(Enum):
    """Possible outcomes of an arcade episode"""
    ONGOING = "ongoing"
    GAME_OVER = "game_over"  # Lost 2 rounds to same opponent
    ARCADE_COMPLETE = "arcade_complete"  # Beat all opponents
    ERROR = "error"

@dataclass
class OpponentRecord:
    """Record against a specific opponent"""
    opponent_number: int
    rounds_won: int = 0
    rounds_lost: int = 0
    rounds_drawn: int = 0
    
    @property
    def total_rounds(self) -> int:
        return self.rounds_won + self.rounds_lost + self.rounds_drawn
    
    @property
    def is_defeated(self) -> bool:
        """Check if we've lost 2 rounds to this opponent (game over)"""
        return self.rounds_lost >= 2
    
    @property
    def is_beaten(self) -> bool:
        """Check if we've beaten this opponent (won 2 rounds)"""
        return self.rounds_won >= 2

@dataclass
class ArcadeStats:
    """Statistics for an entire arcade episode"""
    start_time: float
    end_time: Optional[float] = None
    total_rounds: int = 0
    total_opponents: int = 0
    current_opponent: int = 1
    opponents_beaten: int = 0
    outcome: ArcadeOutcome = ArcadeOutcome.ONGOING
    
    # Round history
    round_history: List[RoundStats] = None
    
    def __post_init__(self):
        if self.round_history is None:
            self.round_history = []
    
    @property
    def duration(self) -> float:
        """Arcade episode duration in seconds"""
        if self.end_time is None:
            return time.time() - self.start_time
        return self.end_time - self.start_time
    
    @property
    def is_finished(self) -> bool:
        """Check if arcade episode is finished"""
        return self.outcome != ArcadeOutcome.ONGOING

class ArcadeEpisode:
    """
    Manages a complete arcade run through multiple opponents.
    Handles round progression, opponent tracking, and game over conditions.
    """
    
    def __init__(self, max_opponents: int = 8, window_title: str = "Bloody Roar II (USA) [PlayStation] - BizHawk"):
        self.max_opponents = max_opponents
        self.window_title = window_title
        
        # Episode state
        self.stats = None
        self.is_active = False
        
        # Opponent tracking
        self.opponent_records: Dict[int, OpponentRecord] = {}
        self.current_opponent = 1
        
        # Current round
        self.current_round: Optional[RoundSubEpisode] = None
        
        print(f"ArcadeEpisode initialized (max opponents: {max_opponents})")
    
    def reset(self) -> Dict[str, Any]:
        """
        Reset and start a new arcade episode
        
        Returns:
            Initial arcade state info
        """
        print("Starting new arcade episode...")
        
        # Reset all state
        self.stats = ArcadeStats(
            start_time=time.time(),
            total_opponents=self.max_opponents
        )
        
        # Reset opponent tracking
        self.opponent_records.clear()
        self.current_opponent = 1
        
        # Clean up any existing round
        if self.current_round:
            self.current_round.close()
            self.current_round = None
        
        # Mark as active
        self.is_active = True
        
        print(f"Arcade episode started - facing opponent {self.current_opponent}")
        
        return self._get_arcade_info()
    
    def get_current_round(self) -> RoundSubEpisode:
        """
        Get the current round sub-episode for training
        
        Returns:
            RoundSubEpisode instance for the current round
        """
        if not self.is_active:
            raise RuntimeError("Arcade episode is not active. Call reset() first.")
        
        # Create new round if needed
        if self.current_round is None or not self.current_round.is_round_active():
            if self.current_round:
                self.current_round.close()
            
            self.current_round = RoundSubEpisode(self.window_title)
            print(f"Created new round vs opponent {self.current_opponent}")
        
        return self.current_round
    
    def complete_round(self, round_outcome: RoundOutcome) -> Dict[str, Any]:
        """
        Complete the current round and update arcade state
        
        Args:
            round_outcome: How the round ended
            
        Returns:
            Updated arcade state info
        """
        if not self.is_active:
            raise RuntimeError("Arcade episode is not active.")
        
        if self.current_round is None:
            raise RuntimeError("No active round to complete.")
        
        # Get round stats
        round_stats = self.current_round.get_stats()
        if round_stats:
            self.stats.round_history.append(round_stats)
            self.stats.total_rounds += 1
        
        # Update opponent record
        self._update_opponent_record(round_outcome)
        
        # Check if arcade episode should end
        arcade_done = self._check_arcade_end()
        
        # Progress to next opponent if current one is beaten
        if self._is_opponent_beaten():
            self._progress_to_next_opponent()
        
        # Close current round
        self.current_round.close()
        self.current_round = None
        
        print(f"Round completed: {round_outcome.value}")
        print(f"Opponent {self.current_opponent} record: {self._get_opponent_record()}")
        
        return self._get_arcade_info()
    
    def _update_opponent_record(self, outcome: RoundOutcome):
        """Update the record against current opponent"""
        if self.current_opponent not in self.opponent_records:
            self.opponent_records[self.current_opponent] = OpponentRecord(self.current_opponent)
        
        record = self.opponent_records[self.current_opponent]
        
        if outcome == RoundOutcome.PLAYER_WIN:
            record.rounds_won += 1
        elif outcome == RoundOutcome.PLAYER_LOSS:
            record.rounds_lost += 1
        elif outcome == RoundOutcome.DRAW:
            record.rounds_drawn += 1
        # Timeout and error count as losses for simplicity
        elif outcome in [RoundOutcome.TIMEOUT, RoundOutcome.ERROR]:
            record.rounds_lost += 1
    
    def _check_arcade_end(self) -> bool:
        """Check if arcade episode should end"""
        current_record = self._get_opponent_record()
        
        # Game over if lost 2 rounds to current opponent
        if current_record.is_defeated:
            self.stats.outcome = ArcadeOutcome.GAME_OVER
            self.stats.end_time = time.time()
            self.is_active = False
            print(f"GAME OVER: Lost 2 rounds to opponent {self.current_opponent}")
            return True
        
        # Check if we've beaten all opponents
        if self.current_opponent > self.max_opponents:
            self.stats.outcome = ArcadeOutcome.ARCADE_COMPLETE
            self.stats.end_time = time.time()
            self.is_active = False
            print("ARCADE COMPLETE: Beat all opponents!")
            return True
        
        return False
    
    def _is_opponent_beaten(self) -> bool:
        """Check if current opponent is beaten"""
        current_record = self._get_opponent_record()
        return current_record.is_beaten
    
    def _progress_to_next_opponent(self):
        """Progress to the next opponent"""
        if self._is_opponent_beaten():
            self.stats.opponents_beaten += 1
            self.current_opponent += 1
            print(f"Opponent {self.current_opponent - 1} defeated! Moving to opponent {self.current_opponent}")
    
    def _get_opponent_record(self) -> OpponentRecord:
        """Get record against current opponent"""
        if self.current_opponent not in self.opponent_records:
            self.opponent_records[self.current_opponent] = OpponentRecord(self.current_opponent)
        return self.opponent_records[self.current_opponent]
    
    def _get_arcade_info(self) -> Dict[str, Any]:
        """Get current arcade state information"""
        current_record = self._get_opponent_record()
        
        return {
            'arcade_active': self.is_active,
            'current_opponent': self.current_opponent,
            'opponent_wins': current_record.rounds_won,
            'opponent_losses': current_record.rounds_lost,
            'opponent_draws': current_record.rounds_drawn,
            'opponents_beaten': self.stats.opponents_beaten,
            'total_rounds': self.stats.total_rounds,
            'arcade_duration': self.stats.duration,
            'arcade_outcome': self.stats.outcome.value,
            'max_opponents': self.max_opponents,
        }
    
    def is_done(self) -> bool:
        """Check if arcade episode is finished"""
        return not self.is_active
    
    def get_stats(self) -> Optional[ArcadeStats]:
        """Get current arcade statistics"""
        return self.stats
    
    def trigger_tas_restart(self):
        """Trigger TAS restart (placeholder for now)"""
        print("TAS RESTART: Restarting arcade from beginning...")
        # TODO: Implement actual TAS restart mechanism
        # This could involve:
        # - Sending commands to BizHawk to load savestate
        # - Resetting emulator to start of arcade mode
        # - Clearing all game state
        time.sleep(1)  # Simulate restart time
        print("TAS restart completed")
    
    def close(self):
        """Clean up resources"""
        if self.current_round:
            self.current_round.close()
            self.current_round = None
        
        if self.is_active:
            self.is_active = False
            if self.stats:
                self.stats.end_time = time.time()
                self.stats.outcome = ArcadeOutcome.ERROR
        
        print("ArcadeEpisode closed")
    
    def __del__(self):
        """Cleanup on deletion"""
        self.close()

# Test function
if __name__ == "__main__":
    print("Testing ArcadeEpisode...")
    
    arcade = ArcadeEpisode(max_opponents=3)  # Small test
    
    try:
        # Start arcade
        arcade_info = arcade.reset()
        print(f"Arcade started: {arcade_info}")
        
        # Simulate a few rounds
        for round_num in range(8):  # Enough to test opponent progression
            print(f"\n--- Round {round_num + 1} ---")
            
            # Get current round
            current_round = arcade.get_current_round()
            print(f"Got round vs opponent {arcade.current_opponent}")
            
            # Simulate round (without actually playing)
            # For testing, we'll just create fake outcomes
            if round_num % 3 == 0:
                outcome = RoundOutcome.PLAYER_WIN
            elif round_num % 3 == 1:
                outcome = RoundOutcome.PLAYER_LOSS
            else:
                outcome = RoundOutcome.PLAYER_WIN
            
            # Complete round
            arcade_info = arcade.complete_round(outcome)
            print(f"Round completed: {outcome.value}")
            print(f"Arcade state: {arcade_info}")
            
            # Check if arcade is done
            if arcade.is_done():
                print("Arcade episode finished!")
                break
        
        # Show final stats
        final_stats = arcade.get_stats()
        if final_stats:
            print(f"\nFinal arcade stats:")
            print(f"  Duration: {final_stats.duration:.1f}s")
            print(f"  Total rounds: {final_stats.total_rounds}")
            print(f"  Opponents beaten: {final_stats.opponents_beaten}")
            print(f"  Outcome: {final_stats.outcome.value}")
    
    except Exception as e:
        print(f"Error during test: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        arcade.close()