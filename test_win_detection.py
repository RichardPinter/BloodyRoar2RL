import time
import numpy as np
from enum import Enum
from dataclasses import dataclass
from window_capture import WindowCapture
from health_detector import HealthDetector, HealthBarConfig
from test_health_pixels import calculate_yellow_percentage

class RoundStatus(Enum):
    """Possible round statuses"""
    ONGOING = "ONGOING"
    P1_WINS = "P1_WINS"
    P2_WINS = "P2_WINS"
    DOUBLE_KO = "DOUBLE_KO"
    WINNER_DECLARED = "WINNER_DECLARED"
    WAITING_FOR_RESET = "WAITING_FOR_RESET"

@dataclass
class MatchRecord:
    """Record for a single match (best of 3 rounds)"""
    opponent_number: int
    rounds_won_p1: int = 0
    rounds_won_p2: int = 0
    match_winner: str = ""
    is_complete: bool = False

@dataclass
class WinDetectionState:
    """State for tracking win detection"""
    p1_zero_frames: int = 0
    p2_zero_frames: int = 0
    status: RoundStatus = RoundStatus.ONGOING
    winner_declared_frame: int = 0
    zero_threshold: int = 10  # Frames of zero health needed to declare death
    round_winner: str = ""  # Who won the round
    celebration_frames: int = 0  # Frames since winner declared
    
    # Match tracking
    current_opponent: int = 1
    current_match: MatchRecord = None
    match_history: list = None  # List of completed matches
    
    def __post_init__(self):
        if self.current_match is None:
            self.current_match = MatchRecord(self.current_opponent)
        if self.match_history is None:
            self.match_history = []

def test_win_detection():
    """
    Test win detection based on consecutive zero-health frames
    Shows real-time detection of round winners
    """
    
    WINDOW_TITLE = "Bloody Roar II (USA) [PlayStation] - BizHawk"
    ZERO_THRESHOLD = 10  # Consecutive zero frames needed for death
    
    # Initialize components
    capture = WindowCapture(WINDOW_TITLE)
    health_detector = HealthDetector()
    config = health_detector.config
    
    if not capture.is_valid:
        print(f"ERROR: Window '{WINDOW_TITLE}' not found!")
        print("Make sure BizHawk is running with Bloody Roar 2")
        return
    
    print("Win Detection Test")
    print("=" * 80)
    print(f"Zero health threshold: {ZERO_THRESHOLD} consecutive frames")
    print(f"Monitoring health bar regions for win conditions")
    print("=" * 80)
    print("Press Ctrl+C to stop, 'r' + Enter to reset")
    print()
    
    # Initialize win detection state
    win_state = WinDetectionState(zero_threshold=ZERO_THRESHOLD)
    frame_count = 0
    start_time = time.time()
    
    try:
        while True:
            frame_count += 1
            
            # Capture health bar regions
            p1_strip = capture.capture_region(
                x=config.p1_x, y=config.bar_y,
                width=config.bar_length, height=config.bar_height
            )
            p2_strip = capture.capture_region(
                x=config.p2_x - config.bar_length, y=config.bar_y,
                width=config.bar_length, height=config.bar_height
            )
            
            if p1_strip is None or p2_strip is None:
                print(f"\rFrame {frame_count:4d} | ERROR: Failed to capture health regions", end='', flush=True)
                time.sleep(0.1)
                continue
            
            # Calculate health percentages
            p1_health_pct, p1_count = calculate_yellow_percentage(p1_strip, config)
            p2_health_pct, p2_count = calculate_yellow_percentage(p2_strip, config)
            
            # Update win detection state and handle different states
            if win_state.status == RoundStatus.ONGOING:
                # Only update detection during ongoing rounds
                update_win_detection(win_state, p1_health_pct, p2_health_pct, frame_count)
                
                # Check if winner was just declared
                if win_state.status != RoundStatus.ONGOING:
                    # Update match record
                    update_match_record(win_state)
                    
                    print()  # New line for winner announcement
                    print("🎉 " + "="*60 + " 🎉")
                    print(f"   ROUND WINNER: {win_state.round_winner}")
                    print(f"   Declared at frame {frame_count} ({elapsed:.1f}s)")
                    print(f"   Match vs Opponent {win_state.current_opponent}: P1={win_state.current_match.rounds_won_p1} P2={win_state.current_match.rounds_won_p2}")
                    
                    # Check if match is complete
                    if win_state.current_match.is_complete:
                        print(f"   🏆 MATCH COMPLETE! Winner: {win_state.current_match.match_winner}")
                        print(f"   Match record added to history (Total matches: {len(win_state.match_history) + 1})")
                    
                    print("🎉 " + "="*60 + " 🎉")
                    print("Entering celebration phase... waiting for new round to start")
                    print("(Press 'r' to reset, 'n' for new opponent, 's' for stats)")
                    win_state.status = RoundStatus.WINNER_DECLARED
                
            elif win_state.status == RoundStatus.WINNER_DECLARED:
                # During celebration, look for new round starting
                win_state.celebration_frames += 1
                
                # Check for new round (both players have good health)
                if p1_health_pct > 80.0 and p2_health_pct > 80.0:
                    print()
                    print("🔄 NEW ROUND DETECTED! Auto-resetting...")
                    reset_win_detection(win_state)
                    frame_count = 0
                    start_time = time.time()
                    print("Reset complete. Starting new round detection...")
                    continue
                
                # Check for manual input (Windows compatible)
                import msvcrt
                if msvcrt.kbhit():
                    try:
                        key = msvcrt.getch().decode('utf-8').lower()
                        if key == 'r':
                            print()
                            print("🔄 MANUAL RESET")
                            reset_win_detection(win_state)
                            frame_count = 0
                            start_time = time.time()
                            print("Reset complete. Starting new round detection...")
                            continue
                        elif key == 'n':
                            print()
                            print("🆕 NEW OPPONENT")
                            start_new_opponent(win_state)
                            reset_win_detection(win_state)
                            frame_count = 0
                            start_time = time.time()
                            print("Ready for new opponent. Starting round detection...")
                            continue
                        elif key == 's':
                            print()
                            show_match_statistics(win_state)
                            print("Press any key to continue...")
                            msvcrt.getch()
                            continue
                    except:
                        pass
            
            # Create status display
            elapsed = time.time() - start_time
            status_display = get_status_display(win_state, p1_health_pct, p2_health_pct, frame_count, elapsed)
            
            print(f"\r{status_display}", end='', flush=True)
            
            time.sleep(0.05)  # ~20fps
            
    except KeyboardInterrupt:
        print(f"\n\nWin detection test stopped after {frame_count} frames ({time.time() - start_time:.1f}s)")
        if win_state.status != RoundStatus.ONGOING:
            print(f"Final result: {win_state.status.value}")

def update_win_detection(win_state: WinDetectionState, p1_health: float, p2_health: float, frame_count: int):
    """
    Update win detection state based on current health values
    
    Args:
        win_state: Current win detection state
        p1_health: P1 health percentage
        p2_health: P2 health percentage
        frame_count: Current frame number
    """
    
    # Only update if round is still ongoing
    if win_state.status != RoundStatus.ONGOING:
        return
    
    # Check P1 health
    if p1_health <= 0.0:
        win_state.p1_zero_frames += 1
    else:
        win_state.p1_zero_frames = 0
    
    # Check P2 health
    if p2_health <= 0.0:
        win_state.p2_zero_frames += 1
    else:
        win_state.p2_zero_frames = 0
    
    # Check for win conditions
    p1_dead = win_state.p1_zero_frames >= win_state.zero_threshold
    p2_dead = win_state.p2_zero_frames >= win_state.zero_threshold
    
    if p1_dead and p2_dead:
        win_state.status = RoundStatus.DOUBLE_KO
        win_state.round_winner = "DOUBLE KO"
        win_state.winner_declared_frame = frame_count
    elif p1_dead:
        win_state.status = RoundStatus.P2_WINS
        win_state.round_winner = "PLAYER 2"
        win_state.winner_declared_frame = frame_count
    elif p2_dead:
        win_state.status = RoundStatus.P1_WINS
        win_state.round_winner = "PLAYER 1"
        win_state.winner_declared_frame = frame_count

def update_match_record(win_state: WinDetectionState):
    """Update match record when a round ends"""
    if win_state.round_winner == "PLAYER 1":
        win_state.current_match.rounds_won_p1 += 1
    elif win_state.round_winner == "PLAYER 2":
        win_state.current_match.rounds_won_p2 += 1
    # Double KO doesn't count as a win for either player
    
    # Check if match is complete (first to 2 wins)
    if win_state.current_match.rounds_won_p1 >= 2:
        win_state.current_match.match_winner = "PLAYER 1"
        win_state.current_match.is_complete = True
        win_state.match_history.append(win_state.current_match)
    elif win_state.current_match.rounds_won_p2 >= 2:
        win_state.current_match.match_winner = "PLAYER 2"
        win_state.current_match.is_complete = True
        win_state.match_history.append(win_state.current_match)

def start_new_opponent(win_state: WinDetectionState):
    """Start a new opponent (complete current match if not finished)"""
    # If current match isn't complete, mark it as incomplete and save
    if not win_state.current_match.is_complete:
        win_state.current_match.match_winner = "INCOMPLETE"
        win_state.match_history.append(win_state.current_match)
    
    # Start new opponent
    win_state.current_opponent += 1
    win_state.current_match = MatchRecord(win_state.current_opponent)
    print(f"    [DEBUG] Started opponent {win_state.current_opponent}")

def show_match_statistics(win_state: WinDetectionState):
    """Show detailed match statistics"""
    print("\n" + "="*80)
    print("🏆 MATCH STATISTICS")
    print("="*80)
    
    if not win_state.match_history and win_state.current_match.rounds_won_p1 == 0 and win_state.current_match.rounds_won_p2 == 0:
        print("No matches completed yet.")
        print("="*80)
        return
    
    # Show completed matches
    p1_match_wins = 0
    p2_match_wins = 0
    
    for i, match in enumerate(win_state.match_history):
        status = "✅" if match.is_complete else "❌"
        print(f"Opponent {match.opponent_number}: P1={match.rounds_won_p1} P2={match.rounds_won_p2} Winner: {match.match_winner} {status}")
        
        if match.match_winner == "PLAYER 1":
            p1_match_wins += 1
        elif match.match_winner == "PLAYER 2":
            p2_match_wins += 1
    
    # Show current match
    if not win_state.current_match.is_complete:
        print(f"Opponent {win_state.current_match.opponent_number}: P1={win_state.current_match.rounds_won_p1} P2={win_state.current_match.rounds_won_p2} Winner: IN PROGRESS ⏳")
    
    print("-"*80)
    print(f"OVERALL RECORD:")
    print(f"  Player 1 match wins: {p1_match_wins}")
    print(f"  Player 2 match wins: {p2_match_wins}")
    print(f"  Total matches: {len(win_state.match_history)}")
    print(f"  Current opponent: {win_state.current_opponent}")
    print("="*80)

def reset_win_detection(win_state: WinDetectionState):
    """Reset win detection state for a new round"""
    print(f"    [DEBUG] Resetting: P1 zeros: {win_state.p1_zero_frames} -> 0, P2 zeros: {win_state.p2_zero_frames} -> 0")
    win_state.p1_zero_frames = 0
    win_state.p2_zero_frames = 0
    win_state.status = RoundStatus.ONGOING
    win_state.winner_declared_frame = 0
    win_state.round_winner = ""
    win_state.celebration_frames = 0
    print(f"    [DEBUG] Reset complete. Status: {win_state.status.value}")

def get_status_display(win_state: WinDetectionState, p1_health: float, p2_health: float, 
                      frame_count: int, elapsed: float) -> str:
    """
    Create status display string for terminal output
    
    Returns:
        Formatted status string
    """
    
    # Health and zero frame info
    p1_info = f"P1: {p1_health:5.1f}% ({win_state.p1_zero_frames} zero)"
    p2_info = f"P2: {p2_health:5.1f}% ({win_state.p2_zero_frames} zero)"
    
    # Match info
    match_info = f"Op{win_state.current_opponent} [{win_state.current_match.rounds_won_p1}-{win_state.current_match.rounds_won_p2}]"
    
    # Status with color indicators
    if win_state.status == RoundStatus.ONGOING:
        status_info = "Status: ONGOING"
        # Add warning if someone is close to death
        if win_state.p1_zero_frames >= win_state.zero_threshold - 3:
            status_info += f" ⚠️ P1 DANGER ({win_state.zero_threshold - win_state.p1_zero_frames} frames left)"
        elif win_state.p2_zero_frames >= win_state.zero_threshold - 3:
            status_info += f" ⚠️ P2 DANGER ({win_state.zero_threshold - win_state.p2_zero_frames} frames left)"
    elif win_state.status == RoundStatus.WINNER_DECLARED:
        status_info = f"🏆 {win_state.round_winner} WINS! (celebration: {win_state.celebration_frames} frames)"
    else:
        status_info = f"*** {win_state.status.value} ***"
    
    return f"Frame {frame_count:4d} ({elapsed:6.1f}s) | {p1_info} | {p2_info} | {match_info} | {status_info}"

def test_win_detection_with_thresholds():
    """
    Test win detection with different zero-frame thresholds
    """
    
    print("Win Detection Threshold Test")
    print("=" * 50)
    print("Choose zero-frame threshold:")
    print("1. 5 frames (fast detection)")
    print("2. 10 frames (default)")
    print("3. 15 frames (conservative)")
    print("4. Custom threshold")
    
    choice = input("\nEnter choice (1-4): ").strip()
    
    if choice == '1':
        threshold = 5
    elif choice == '2':
        threshold = 10
    elif choice == '3':
        threshold = 15
    elif choice == '4':
        try:
            threshold = int(input("Enter custom threshold (frames): "))
            if threshold < 1:
                threshold = 10
        except ValueError:
            threshold = 10
    else:
        threshold = 10
    
    print(f"\nUsing threshold: {threshold} frames")
    print("Starting test...")
    
    # Modify the global threshold for this test
    # (In a real implementation, this would be passed as a parameter)
    global ZERO_THRESHOLD
    original_threshold = globals().get('ZERO_THRESHOLD', 10)
    globals()['ZERO_THRESHOLD'] = threshold
    
    try:
        test_win_detection()
    finally:
        globals()['ZERO_THRESHOLD'] = original_threshold

if __name__ == "__main__":
    print("Win Detection Tests")
    print("=" * 40)
    print("1. Standard win detection test")
    print("2. Test with different thresholds")
    print("3. Exit")
    
    choice = input("\nEnter choice (1-3): ").strip()
    
    if choice == '1':
        test_win_detection()
    elif choice == '2':
        test_win_detection_with_thresholds()
    elif choice == '3':
        print("Exiting...")
    else:
        print("Invalid choice, running standard test...")
        test_win_detection()