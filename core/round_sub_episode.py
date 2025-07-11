#!/usr/bin/env python3
"""
Round State Monitor

This class is responsible for getting the states in the environment:
- Health values (from health_detector.py)
- Fighter positions and rotations (from fighter_detector.py) 
- Win detection (health = 0 for 10+ frames)

No actions/rewards/environment complexity - just pure state monitoring.
"""

import time
import numpy as np
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from detection.window_capture import WindowCapture
from detection.health_detector import HealthDetector, HealthState
from detection.fighter_detector import FighterDetector, FighterDetection

class RoundOutcome(Enum):
    """Possible outcomes of a round"""
    ONGOING = "ongoing"
    PLAYER_WIN = "player_win"  # P1 wins
    PLAYER_LOSS = "player_loss"  # P2 wins
    DRAW = "draw"
    TIMEOUT = "timeout"
    ERROR = "error"

@dataclass
class GameState:
    """Complete game state for one frame"""
    # Health information
    p1_health: float = 0.0  # 0-100%
    p2_health: float = 0.0  # 0-100%
    
    # Fighter positions
    p1_position: Optional[Tuple[int, int]] = None  # (x, y)
    p2_position: Optional[Tuple[int, int]] = None  # (x, y)
    p1_bbox: Optional[Tuple[int, int, int, int]] = None  # (x1, y1, x2, y2)
    p2_bbox: Optional[Tuple[int, int, int, int]] = None  # (x1, y1, x2, y2)
    
    # Distance between fighters
    fighter_distance: Optional[float] = None
    
    # Win detection state
    p1_zero_frames: int = 0
    p2_zero_frames: int = 0
    round_outcome: RoundOutcome = RoundOutcome.ONGOING
    
    # Detection status
    health_detection_working: bool = False
    fighter_detection_working: bool = False
    
    # Timing
    timestamp: float = 0.0
    frame_count: int = 0

class RoundStateMonitor:
    """
    Monitors game state during a round - health, positions, and win conditions.
    This class focuses purely on state observation, no actions or rewards.
    """
    
    def __init__(self, window_title: str = "Bloody Roar II (USA) [PlayStation] - BizHawk", 
                 zero_threshold: int = 10):
        self.window_title = window_title
        
        # Initialize detection systems
        try:
            self.capture = WindowCapture(window_title)
            self.health_detector = HealthDetector()
            self.health_available = True
            print("✅ Health detection initialized")
        except Exception as e:
            print(f"❌ Health detection failed: {e}")
            self.health_available = False
        
        try:
            self.fighter_detector = FighterDetector()
            self.fighter_available = True
            print("✅ Fighter detection initialized")
        except Exception as e:
            print(f"❌ Fighter detection failed: {e}")
            self.fighter_available = False
        
        # State tracking
        self.current_state = GameState()
        self.frame_count = 0
        self.start_time = time.time()
        
        # Health history tracking for robust winner detection
        self.health_history = []  # List of (timestamp, p1_health, p2_health, frame_count)
        
        # Win detection parameters
        self.zero_threshold = zero_threshold  # Configurable frames of 0% health needed for death
        self.max_round_time = 120.0  # 2 minutes max
        
        print(f"RoundStateMonitor initialized")
        print(f"  Health detection: {'✅' if self.health_available else '❌'}")
        print(f"  Fighter detection: {'✅' if self.fighter_available else '❌'}")
        print(f"  Death detection threshold: {self.zero_threshold} frames")
    
    def get_current_state(self) -> GameState:
        """
        Get the current complete game state
        
        Returns:
            GameState with all current information
        """
        self.frame_count += 1
        current_time = time.time()
        
        # Create new state object
        state = GameState(
            timestamp=current_time,
            frame_count=self.frame_count
        )
        
        # Get health information
        if self.health_available:
            health_state = self._detect_health()
            if health_state:
                state.p1_health = health_state.p1_health
                state.p2_health = health_state.p2_health
                state.health_detection_working = True
                
                # Record health history for winner analysis
                self.health_history.append((
                    current_time,
                    state.p1_health,
                    state.p2_health,
                    self.frame_count
                ))
        
        # Get fighter positions
        if self.fighter_available:
            fighter_detection = self._detect_fighters()
            if fighter_detection:
                if fighter_detection.player1:
                    state.p1_position = fighter_detection.player1.center
                    state.p1_bbox = fighter_detection.player1.bbox
                
                if fighter_detection.player2:
                    state.p2_position = fighter_detection.player2.center
                    state.p2_bbox = fighter_detection.player2.bbox
                
                if fighter_detection.distance:
                    state.fighter_distance = fighter_detection.distance
                
                state.fighter_detection_working = True
        
        # Update win detection
        self._update_win_detection(state)
        
        # Check round end conditions
        state.round_outcome = self._check_round_outcome(state)
        
        # Store as current state
        self.current_state = state
        
        return state
    
    def _detect_health(self) -> Optional[HealthState]:
        """Detect health using pixel-based detection"""
        try:
            return self.health_detector.detect(self.capture)
        except Exception as e:
            print(f"Health detection error: {e}")
            return None
    
    def _detect_fighters(self) -> Optional[FighterDetection]:
        """Detect fighter positions using YOLO"""
        try:
            # Capture current frame
            frame = self.capture.capture()
            if frame is None:
                return None
            
            return self.fighter_detector.detect(frame)
        except Exception as e:
            print(f"Fighter detection error: {e}")
            return None
    
    def _update_win_detection(self, state: GameState):
        """Update win detection counters based on health"""
        # Update zero frame counters
        if state.p1_health <= 0.0:
            state.p1_zero_frames = self.current_state.p1_zero_frames + 1
        else:
            state.p1_zero_frames = 0
        
        if state.p2_health <= 0.0:
            state.p2_zero_frames = self.current_state.p2_zero_frames + 1
        else:
            state.p2_zero_frames = 0
    
    def _check_round_outcome(self, state: GameState) -> RoundOutcome:
        """Check if round should end based on actual player deaths only"""
        # Only check for actual deaths - no artificial time limits
        p1_dead = state.p1_zero_frames >= self.zero_threshold
        p2_dead = state.p2_zero_frames >= self.zero_threshold
        
        # One player always dies first - no simultaneous deaths
        if p1_dead:
            return RoundOutcome.PLAYER_LOSS  # P1 dead, P2 wins
        elif p2_dead:
            return RoundOutcome.PLAYER_WIN   # P2 dead, P1 wins
        
        # Check if we should use winner analysis for complex death scenarios
        # (when health has gone to zero multiple times but current streak < threshold)
        if len(self.health_history) >= 20:  # Only after sufficient data
            total_p1_zeros = sum(1 for h in self.health_history if h['p1_health'] <= 0.0)
            total_p2_zeros = sum(1 for h in self.health_history if h['p2_health'] <= 0.0)
            
            # If either player has significant zero frames, run winner analysis
            if total_p1_zeros >= self.zero_threshold or total_p2_zeros >= self.zero_threshold:
                winner, analysis = self.analyze_winner_from_history()
                if winner == "PLAYER 1":
                    print(f"🏆 Round ended via winner analysis: {winner}")
                    return RoundOutcome.PLAYER_WIN
                elif winner == "PLAYER 2":
                    print(f"🏆 Round ended via winner analysis: {winner}")
                    return RoundOutcome.PLAYER_LOSS
        
        return RoundOutcome.ONGOING  # Round continues until actual death
    
    def print_state(self, state: GameState, show_positions: bool = True):
        """Print current state in a readable format"""
        # Health status
        health_str = f"P1: {state.p1_health:5.1f}% | P2: {state.p2_health:5.1f}%"
        
        # Zero frame counters (show if > 0)
        zero_str = ""
        if state.p1_zero_frames > 0:
            zero_str += f" P1-zero:{state.p1_zero_frames}"
        if state.p2_zero_frames > 0:
            zero_str += f" P2-zero:{state.p2_zero_frames}"
        
        # Positions
        pos_str = ""
        if show_positions:
            if state.p1_position and state.p2_position:
                dist_str = f" dist:{state.fighter_distance:.0f}" if state.fighter_distance else ""
                pos_str = f" | P1@{state.p1_position} P2@{state.p2_position}{dist_str}"
            elif state.p1_position:
                pos_str = f" | P1@{state.p1_position} P2:---"
            elif state.p2_position:
                pos_str = f" | P1:--- P2@{state.p2_position}"
        
        # Outcome
        outcome_str = f" | {state.round_outcome.value}"
        if state.round_outcome != RoundOutcome.ONGOING:
            outcome_str = f" | 🏆 {state.round_outcome.value.upper()}"
        
        # Detection status
        status_str = ""
        if not state.health_detection_working:
            status_str += " [HEALTH-FAIL]"
        if not state.fighter_detection_working:
            status_str += " [FIGHTER-FAIL]"
        
        print(f"Frame {state.frame_count:4d} | {health_str}{zero_str}{pos_str}{outcome_str}{status_str}")
    
    def reset(self):
        """Reset for a new round"""
        self.frame_count = 0
        self.start_time = time.time()
        self.current_state = GameState()
        self.health_history = []  # Clear health history for new round
        print("🔄 Round state monitor reset")
    
    def is_round_finished(self) -> bool:
        """Check if round is finished"""
        return self.current_state.round_outcome != RoundOutcome.ONGOING
    
    def get_winner(self) -> Optional[str]:
        """Get the winner if round is finished"""
        if self.current_state.round_outcome == RoundOutcome.PLAYER_WIN:
            return "PLAYER 1"
        elif self.current_state.round_outcome == RoundOutcome.PLAYER_LOSS:
            return "PLAYER 2"
        # No draws possible - one player always dies first
        return None
    
    def analyze_winner_from_history(self) -> tuple[Optional[str], dict]:
        """
        Analyze health history to find winner:
        1. First check for persistent health bar disappearance (UI timeout)
        2. If found, use health values from before disappearance  
        3. Otherwise use zero streak analysis for actual deaths
        
        Returns:
            (winner, analysis_details)
        """
        if not self.health_history:
            return None, {"error": "No health history available"}
        
        # Step 1: Find persistent health bar disappearance (3+ consecutive both=0 frames)
        disappearance_start = None
        consecutive_zeros = 0
        
        # Look backwards through history
        for i in range(len(self.health_history) - 1, -1, -1):
            timestamp, p1_health, p2_health, frame_count = self.health_history[i]
            
            if p1_health <= 0.0 and p2_health <= 0.0:
                consecutive_zeros += 1
                if consecutive_zeros >= 3:  # Persistent disappearance found
                    disappearance_start = i + 2  # Index of start of 3-frame sequence
                    break
            else:
                consecutive_zeros = 0  # Reset counter
        
        # Step 2: If we found persistent disappearance, use pre-disappearance health
        if disappearance_start is not None and disappearance_start > 0:
            # Get health from just before disappearance
            pre_disappearance_idx = disappearance_start - 1
            _, p1_health_before, p2_health_before, frame_before = self.health_history[pre_disappearance_idx]
            
            # Determine winner based on health before UI disappeared
            if p1_health_before > p2_health_before:
                winner = "PLAYER 1"
                reason = f"Higher health before UI timeout: P1={p1_health_before:.1f}% > P2={p2_health_before:.1f}%"
            elif p2_health_before > p1_health_before:
                winner = "PLAYER 2"
                reason = f"Higher health before UI timeout: P2={p2_health_before:.1f}% > P1={p1_health_before:.1f}%"
            else:
                # If exactly equal health (very rare), P1 wins by default
                winner = "PLAYER 1"
                reason = f"Equal health before UI timeout: P1={p1_health_before:.1f}% = P2={p2_health_before:.1f}% (P1 wins by default)"
            
            analysis = {
                "winner": winner,
                "reason": reason,
                "method": "pre_timeout_health",
                "health_before_timeout": {
                    "p1": p1_health_before,
                    "p2": p2_health_before,
                    "frame": frame_before
                },
                "ui_disappeared_at_frame": self.health_history[disappearance_start][3],
                "consecutive_zero_frames": consecutive_zeros,
                "total_frames_analyzed": len(self.health_history)
            }
            
            return winner, analysis
        
        # Step 3: No persistent disappearance found - use zero streak analysis for actual deaths
        return self._analyze_winner_by_zero_streaks()
    
    def _analyze_winner_by_zero_streaks(self) -> tuple[Optional[str], dict]:
        """Analyze winner based on zero health streaks (for actual player deaths)"""
        
        # Calculate zero streaks for both players
        p1_zero_streaks = []
        p2_zero_streaks = []
        
        current_p1_streak = 0
        current_p2_streak = 0
        
        for timestamp, p1_health, p2_health, frame_count in self.health_history:
            # Track P1 zero streaks
            if p1_health <= 0.0:
                current_p1_streak += 1
            else:
                if current_p1_streak > 0:
                    p1_zero_streaks.append(current_p1_streak)
                current_p1_streak = 0
            
            # Track P2 zero streaks
            if p2_health <= 0.0:
                current_p2_streak += 1
            else:
                if current_p2_streak > 0:
                    p2_zero_streaks.append(current_p2_streak)
                current_p2_streak = 0
        
        # Add final streaks if they're ongoing
        if current_p1_streak > 0:
            p1_zero_streaks.append(current_p1_streak)
        if current_p2_streak > 0:
            p2_zero_streaks.append(current_p2_streak)
        
        # Calculate totals
        p1_total_zero_frames = sum(p1_zero_streaks)
        p2_total_zero_frames = sum(p2_zero_streaks)
        p1_max_streak = max(p1_zero_streaks) if p1_zero_streaks else 0
        p2_max_streak = max(p2_zero_streaks) if p2_zero_streaks else 0
        
        # Determine winner based on total zero time (more zero time = loser)
        winner = None
        reason = ""
        
        if p1_total_zero_frames > p2_total_zero_frames:
            winner = "PLAYER 2"  # P1 had more zero time, so P2 wins
            reason = f"P1 had {p1_total_zero_frames} total zero frames vs P2's {p2_total_zero_frames}"
        elif p2_total_zero_frames > p1_total_zero_frames:
            winner = "PLAYER 1"  # P2 had more zero time, so P1 wins  
            reason = f"P2 had {p2_total_zero_frames} total zero frames vs P1's {p1_total_zero_frames}"
        elif p1_max_streak > p2_max_streak:
            winner = "PLAYER 2"  # P1 had longer single streak
            reason = f"P1 had longer max streak ({p1_max_streak} vs {p2_max_streak})"
        elif p2_max_streak > p1_max_streak:
            winner = "PLAYER 1"  # P2 had longer single streak
            reason = f"P2 had longer max streak ({p2_max_streak} vs {p1_max_streak})"
        else:
            # If exactly equal zero time (very rare), P1 wins by default
            winner = "PLAYER 1"
            reason = f"Equal zero time for both players ({p1_total_zero_frames} frames each, P1 wins by default)"
        
        # Get final health values
        final_p1_health = self.health_history[-1][1] if self.health_history else 0
        final_p2_health = self.health_history[-1][2] if self.health_history else 0
        
        analysis = {
            "winner": winner,
            "reason": reason,
            "method": "zero_streak_analysis",
            "p1_total_zero_frames": p1_total_zero_frames,
            "p2_total_zero_frames": p2_total_zero_frames,
            "p1_max_zero_streak": p1_max_streak,
            "p2_max_zero_streak": p2_max_streak,
            "p1_zero_streaks": p1_zero_streaks,
            "p2_zero_streaks": p2_zero_streaks,
            "final_p1_health": final_p1_health,
            "final_p2_health": final_p2_health,
            "total_frames_analyzed": len(self.health_history)
        }
        
        return winner, analysis
    
    def print_winner_analysis(self):
        """Print detailed winner analysis based on health history"""
        # Only analyze if round actually ended (not used anymore since no timeouts)
        if self.current_state.round_outcome == RoundOutcome.ONGOING:
            print("Round is still ongoing")
            return
        
        winner, analysis = self.analyze_winner_from_history()
        
        print(f"\n🔍 WINNER ANALYSIS (Health History):")
        print(f"   Winner: {winner}")
        print(f"   Method: {analysis['method']}")
        print(f"   Reason: {analysis['reason']}")
        
        if analysis['method'] == 'pre_timeout_health':
            # UI timeout scenario - show health before bars disappeared
            health_before = analysis['health_before_timeout']
            print(f"   Health before timeout: P1={health_before['p1']:.1f}% P2={health_before['p2']:.1f}% (frame {health_before['frame']})")
            print(f"   UI disappeared at frame: {analysis['ui_disappeared_at_frame']}")
            print(f"   Consecutive zero frames: {analysis['consecutive_zero_frames']}")
        else:
            # Zero streak analysis - show detailed death analysis
            print(f"   P1 zero streaks: {analysis['p1_zero_streaks']} (total: {analysis['p1_total_zero_frames']} frames)")
            print(f"   P2 zero streaks: {analysis['p2_zero_streaks']} (total: {analysis['p2_total_zero_frames']} frames)")
            print(f"   Final health: P1={analysis['final_p1_health']:.1f}% P2={analysis['final_p2_health']:.1f}%")
        
        print(f"   Frames analyzed: {analysis['total_frames_analyzed']}")
    
    def wait_for_round_ready(self, timeout: float = 30.0) -> bool:
        """
        Wait until the next round is ready (health bars visible and both players at full health)
        
        Args:
            timeout: Maximum time to wait in seconds
            
        Returns:
            True if round is ready, False if timeout or detection unavailable
        """
        if not self.health_available:
            print("⚠️  Health detection not available - skipping wait")
            return True
        
        print(f"⏳ Waiting for round to start (timeout: {timeout}s)...")
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            if self.is_round_ready():
                print("🎬 Round is ready to start!")
                return True
            
            # Brief delay between checks
            time.sleep(0.5)
        
        print("⏰ Timeout waiting for round to start")
        return False
    
    def is_round_ready(self, min_health: float = 95.0) -> bool:
        """
        Check if round is ready to start (both players at near full health)
        
        Args:
            min_health: Minimum health percentage required for both players
            
        Returns:
            True if round is ready to start, False otherwise
        """
        if not self.health_available:
            return True  # Assume ready if no health detection
        
        # Get current health state
        state = self.get_current_state()
        
        # Round is ready when both players have high health
        if (state.p1_health >= min_health and 
            state.p2_health >= min_health):
            return True
        
        return False
    
    def close(self):
        """Clean up resources"""
        print("RoundStateMonitor closed")


def test_state_monitoring():
    """Test the state monitoring functionality"""
    print("🧪 Testing Round State Monitor")
    print("=" * 60)
    print("This will monitor health, positions, and detect winners")
    print("Press Ctrl+C to stop")
    print("-" * 60)
    
    monitor = RoundStateMonitor()
    
    try:
        monitor.reset()
        
        # Monitor loop
        while not monitor.is_round_finished():
            # Get current state
            state = monitor.get_current_state()
            
            # Print state
            monitor.print_state(state)
            
            # Check for winner
            if monitor.is_round_finished():
                winner = monitor.get_winner()
                print(f"\n🎉 ROUND FINISHED! Winner: {winner}")
                break
            
            # Small delay
            time.sleep(0.1)  # 10 FPS monitoring
    
    except KeyboardInterrupt:
        print(f"\n\n⏹️  Monitoring stopped by user")
    except Exception as e:
        print(f"\n❌ Error during monitoring: {e}")
        import traceback
        traceback.print_exc()
    finally:
        monitor.close()


if __name__ == "__main__":
    test_state_monitoring()