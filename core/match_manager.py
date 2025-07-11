#!/usr/bin/env python3
"""
Match Manager

Manages a best-of-3 match against one opponent using RoundStateMonitor.
Tracks round results and determines match winner.
Updated to remove draw logic since draws are not possible.
"""

import time
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
from enum import Enum
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.round_sub_episode import RoundStateMonitor, RoundOutcome, GameState
from detection.health_detector import HealthDetector
from detection.window_capture import WindowCapture

class MatchOutcome(Enum):
    """Possible outcomes of a match"""
    ONGOING = "ongoing"
    PLAYER_WIN = "player_win"  # P1 wins match
    PLAYER_LOSS = "player_loss"  # P2 wins match
    ERROR = "error"

@dataclass
class RoundResult:
    """Result of a single round"""
    round_number: int
    winner: str  # "PLAYER 1" or "PLAYER 2" (no draws possible)
    outcome: RoundOutcome
    duration: float
    final_p1_health: float
    final_p2_health: float

@dataclass
class MatchStats:
    """Statistics for a complete match"""
    start_time: float
    end_time: Optional[float] = None
    p1_rounds_won: int = 0
    p2_rounds_won: int = 0
    total_rounds: int = 0
    round_results: List[RoundResult] = None
    match_outcome: MatchOutcome = MatchOutcome.ONGOING
    
    def __post_init__(self):
        if self.round_results is None:
            self.round_results = []
    
    @property
    def duration(self) -> float:
        """Match duration in seconds"""
        if self.end_time is None:
            return time.time() - self.start_time
        return self.end_time - self.start_time
    
    @property
    def is_finished(self) -> bool:
        """Check if match is finished"""
        return self.match_outcome != MatchOutcome.ONGOING

class MatchManager:
    """
    Manages a best-of-3 match against one opponent.
    Uses RoundStateMonitor for individual round monitoring.
    No draws possible - one player always wins.
    """
    
    def __init__(self, max_rounds: int = 3, rounds_to_win: int = 2, 
                 window_title: str = "Bloody Roar II (USA) [PlayStation] - BizHawk"):
        self.max_rounds = max_rounds
        self.rounds_to_win = rounds_to_win
        self.window_title = window_title
        
        # Initialize health detection for round transitions
        try:
            self.capture = WindowCapture(window_title)
            self.health_detector = HealthDetector()
            self.health_detection_available = True
            print("✅ Health detection for round transitions initialized")
        except Exception as e:
            print(f"❌ Health detection failed: {e}")
            self.health_detection_available = False
        
        # Match state
        self.stats = None
        self.current_round_monitor: Optional[RoundStateMonitor] = None
        self.current_round_number = 0
        self.is_active = False
        
        print(f"MatchManager initialized (best-of-{max_rounds}, first to {rounds_to_win})")
    
    def start_match(self) -> Dict[str, Any]:
        """
        Start a new match
        
        Returns:
            Initial match info
        """
        print("🥊 Starting new match...")
        
        # Initialize match stats
        self.stats = MatchStats(start_time=time.time())
        self.current_round_number = 0
        self.is_active = True
        
        print(f"Match started - first to {self.rounds_to_win} rounds wins!")
        
        return self._get_match_info()
    
    def start_next_round(self, wait_for_health_bars: bool = True, zero_threshold: int = 7) -> bool:
        """
        Start the next round in the match
        
        Args:
            wait_for_health_bars: Whether to wait for health bars before starting
            zero_threshold: Death detection threshold for the round
            
        Returns:
            True if round started, False if match is already finished
        """
        if not self.is_active:
            print("❌ Match is not active")
            return False
        
        if self.is_match_finished():
            print("❌ Match is already finished")
            return False
        
        # Clean up previous round
        if self.current_round_monitor:
            self.current_round_monitor.close()
        
        # Increment round number
        self.current_round_number += 1
        self.stats.total_rounds = self.current_round_number
        
        print(f"\n🔄 Starting Round {self.current_round_number}")
        print("-" * 40)
        
        # Wait for health bars to appear (round transition)
        if wait_for_health_bars and self.health_detection_available:
            if not self.wait_for_round_ready():
                print("❌ Failed to detect health bars - starting round anyway")
        
        try:
            self.current_round_monitor = RoundStateMonitor(self.window_title, zero_threshold=zero_threshold)
            self.current_round_monitor.reset()
            print(f"✅ Round {self.current_round_number} ready")
            return True
        except Exception as e:
            print(f"❌ Failed to start round: {e}")
            return False
    
    def wait_for_round_ready(self, timeout: float = 30.0) -> bool:
        """
        Wait until the next round is ready (health bars visible)
        
        Args:
            timeout: Maximum time to wait in seconds
            
        Returns:
            True if round is ready, False if timeout or detection unavailable
        """
        if not self.health_detection_available:
            print("⚠️  Health detection not available - skipping wait")
            return True
        
        return self.health_detector.wait_for_health_bars(self.capture, timeout)
    
    def monitor_current_round(self) -> Optional[RoundResult]:
        """
        Monitor the current round until it finishes
        
        Returns:
            RoundResult if round finished, None if still ongoing
        """
        if not self.current_round_monitor:
            print("❌ No active round to monitor")
            return None
        
        # Get current state
        state = self.current_round_monitor.get_current_state()
        
        # Print state (every 10th frame to reduce spam)
        if state.frame_count % 10 == 0:
            self.current_round_monitor.print_state(state)
        
        # Check if round is finished
        if self.current_round_monitor.is_round_finished():
            return self._complete_current_round(state)
        
        return None
    
    def _complete_current_round(self, final_state: GameState) -> RoundResult:
        """Complete the current round and update match stats"""
        winner = self.current_round_monitor.get_winner()
        round_duration = time.time() - self.current_round_monitor.start_time
        
        # Create round result
        round_result = RoundResult(
            round_number=self.current_round_number,
            winner=winner or "UNKNOWN",
            outcome=final_state.round_outcome,
            duration=round_duration,
            final_p1_health=final_state.p1_health,
            final_p2_health=final_state.p2_health
        )
        
        # Update match stats
        self.stats.round_results.append(round_result)
        
        if winner == "PLAYER 1":
            self.stats.p1_rounds_won += 1
        elif winner == "PLAYER 2":
            self.stats.p2_rounds_won += 1
        
        print(f"\n🏁 Round {self.current_round_number} finished!")
        print(f"   Winner: {winner}")
        print(f"   Duration: {round_duration:.1f}s")
        print(f"   Final health: P1={final_state.p1_health:.1f}% P2={final_state.p2_health:.1f}%")
        print(f"   Match score: P1={self.stats.p1_rounds_won} P2={self.stats.p2_rounds_won}")
        
        # Check if match is won
        self._check_match_completion()
        
        return round_result
    
    def _check_match_completion(self):
        """Check if match is complete and update outcome"""
        if self.stats.p1_rounds_won >= self.rounds_to_win:
            self.stats.match_outcome = MatchOutcome.PLAYER_WIN
            self.stats.end_time = time.time()
            self.is_active = False
            print(f"\n🎉 MATCH WON BY PLAYER 1! ({self.stats.p1_rounds_won}-{self.stats.p2_rounds_won})")
            
        elif self.stats.p2_rounds_won >= self.rounds_to_win:
            self.stats.match_outcome = MatchOutcome.PLAYER_LOSS
            self.stats.end_time = time.time()
            self.is_active = False
            print(f"\n💀 MATCH WON BY PLAYER 2! ({self.stats.p1_rounds_won}-{self.stats.p2_rounds_won})")
            
        elif self.current_round_number >= self.max_rounds:
            # All rounds played, determine winner by score (no draws possible in best-of-odd)
            if self.stats.p1_rounds_won > self.stats.p2_rounds_won:
                self.stats.match_outcome = MatchOutcome.PLAYER_WIN
                print(f"\n🎉 MATCH WON BY PLAYER 1 (max rounds)! ({self.stats.p1_rounds_won}-{self.stats.p2_rounds_won})")
            else:
                # P2 must have won more rounds (no ties possible in best-of-odd)
                self.stats.match_outcome = MatchOutcome.PLAYER_LOSS
                print(f"\n💀 MATCH WON BY PLAYER 2 (max rounds)! ({self.stats.p1_rounds_won}-{self.stats.p2_rounds_won})")
            
            self.stats.end_time = time.time()
            self.is_active = False
    
    def is_match_finished(self) -> bool:
        """Check if match is finished"""
        return not self.is_active or (self.stats and self.stats.is_finished)
    
    def get_match_winner(self) -> Optional[str]:
        """Get the match winner if match is finished"""
        if not self.stats or not self.stats.is_finished:
            return None
        
        if self.stats.match_outcome == MatchOutcome.PLAYER_WIN:
            return "PLAYER 1"
        elif self.stats.match_outcome == MatchOutcome.PLAYER_LOSS:
            return "PLAYER 2"
        return None  # No draws possible
    
    def _get_match_info(self) -> Dict[str, Any]:
        """Get current match information"""
        return {
            'match_active': self.is_active,
            'current_round': self.current_round_number,
            'max_rounds': self.max_rounds,
            'p1_rounds_won': self.stats.p1_rounds_won if self.stats else 0,
            'p2_rounds_won': self.stats.p2_rounds_won if self.stats else 0,
            'rounds_to_win': self.rounds_to_win,
            'match_duration': self.stats.duration if self.stats else 0,
            'match_outcome': self.stats.match_outcome.value if self.stats else MatchOutcome.ONGOING.value,
            'is_finished': self.is_match_finished(),
        }
    
    def print_match_summary(self):
        """Print final match summary"""
        if not self.stats:
            print("No match data available")
            return
        
        print("\n" + "=" * 60)
        print("🏆 MATCH SUMMARY")
        print("=" * 60)
        print(f"Final score: P1={self.stats.p1_rounds_won} P2={self.stats.p2_rounds_won}")
        print(f"Total rounds: {self.stats.total_rounds}")
        print(f"Match duration: {self.stats.duration:.1f}s")
        print(f"Winner: {self.get_match_winner()}")
        
        print("\nRound details:")
        for result in self.stats.round_results:
            print(f"  Round {result.round_number}: {result.winner} ({result.duration:.1f}s)")
        
        print("=" * 60)
    
    def close(self):
        """Clean up resources"""
        if self.current_round_monitor:
            self.current_round_monitor.close()
            self.current_round_monitor = None
        
        if self.is_active:
            self.is_active = False
            if self.stats:
                self.stats.end_time = time.time()
                self.stats.match_outcome = MatchOutcome.ERROR
        
        print("MatchManager closed")


def test_match_management():
    """Test the match management functionality"""
    print("🧪 Testing Match Manager")
    print("=" * 60)
    print("This will manage a best-of-3 match with round monitoring")
    print("Press Ctrl+C to stop")
    print("-" * 60)
    
    match_manager = MatchManager()
    
    try:
        # Start match
        match_info = match_manager.start_match()
        print(f"Match started: {match_info}")
        
        # Play rounds until match is finished
        while not match_manager.is_match_finished():
            # Start next round
            if match_manager.start_next_round():
                
                # Monitor round until it finishes
                round_result = None
                while round_result is None and not match_manager.is_match_finished():
                    round_result = match_manager.monitor_current_round()
                    time.sleep(0.1)  # 10 FPS monitoring
                
                if round_result:
                    print(f"Round {round_result.round_number} completed")
            else:
                print("Failed to start round")
                break
        
        # Print final summary
        match_manager.print_match_summary()
        
    except KeyboardInterrupt:
        print(f"\n\n⏹️  Match monitoring stopped by user")
        match_manager.print_match_summary()
    except Exception as e:
        print(f"\n❌ Error during match: {e}")
        import traceback
        traceback.print_exc()
    finally:
        match_manager.close()


if __name__ == "__main__":
    test_match_management()