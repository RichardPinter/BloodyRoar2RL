import time
import atexit
import comtypes
import dxcam
import numpy as np
import cv2

# ─── CONFIG ────────────────────────────────────────────────────────────────
REGION      = (0, 0, 624, 548)      # x, y, width, height

# Round indicator detection - Working coordinates
ROUND_INDICATORS = {
    'p1_round1': (270, 135, 278, 140),   # Player 1, first round indicator
    'p1_round2': (245, 135, 253, 140),   # Player 1, second round indicator
    'p2_round1': (373, 135, 381, 140),   # Player 2, first round indicator
    'p2_round2': (396, 135, 404, 140),   # Player 2, second round indicator
}

# Red color range for round indicators
RED_BGR_LOWER = np.array([0, 0, 150], dtype=np.uint8)
RED_BGR_UPPER = np.array([60, 60, 255], dtype=np.uint8)

# ─── ROUND STATE LOGIC ─────────────────────────────────────────────────────
class RoundState:
    def __init__(self):
        # Confirmed state (persistent, can only increase)
        self.confirmed_p1_rounds = 0
        self.confirmed_p2_rounds = 0

        # Candidate state (needs confirmation)
        self.candidate_state = None
        self.candidate_start_time = None
        self.CONFIRMATION_TIME = 1.0  # Must see state for 1 second

        print("🔄 RoundState initialized: P1:0 P2:0")

    def update(self, detected_p1, detected_p2):
        """
        Update round state with new detection.
        Returns: None, "round_won", or tuple for winner info
        """
        current_time = time.time()
        new_state = (detected_p1, detected_p2)

        # Check if this is a valid upgrade (monotonic rule)
        if (detected_p1 >= self.confirmed_p1_rounds and
            detected_p2 >= self.confirmed_p2_rounds):

            # Check if this is actually an upgrade (not same state)
            if (detected_p1 > self.confirmed_p1_rounds or
                detected_p2 > self.confirmed_p2_rounds):

                if new_state == self.candidate_state:
                    # Same candidate - check if enough time has passed
                    elapsed = current_time - self.candidate_start_time

                    if elapsed >= self.CONFIRMATION_TIME:
                        # CONFIRMED! Update persistent state
                        old_p1, old_p2 = self.confirmed_p1_rounds, self.confirmed_p2_rounds
                        self.confirmed_p1_rounds = detected_p1
                        self.confirmed_p2_rounds = detected_p2

                        # Determine who won the round
                        if detected_p1 > old_p1:
                            winner = "p1"
                            print(f"🎯 ROUND CONFIRMED: P1 won! (P1:{detected_p1} P2:{detected_p2})")
                        else:
                            winner = "p2"
                            print(f"🎯 ROUND CONFIRMED: P2 won! (P1:{detected_p1} P2:{detected_p2})")

                        # Clear candidate
                        self.candidate_state = None
                        self.candidate_start_time = None

                        return ("round_won", winner, detected_p1, detected_p2)
                    else:
                        # Still waiting for confirmation
                        print(f"⏳ [Candidate] P1:{detected_p1} P2:{detected_p2} ({elapsed:.1f}s) - waiting for confirmation...")

                else:
                    # New candidate state
                    self.candidate_state = new_state
                    self.candidate_start_time = current_time
                    print(f"🔍 [New Candidate] P1:{detected_p1} P2:{detected_p2} - starting confirmation timer")

            # else: same as confirmed state, no action needed

        else:
            # Not an upgrade - ignore (noise/temporary false positive)
            if (detected_p1 < self.confirmed_p1_rounds or
                detected_p2 < self.confirmed_p2_rounds):
                print(f"🚫 [Ignored] P1:{detected_p1} P2:{detected_p2} - not an upgrade from P1:{self.confirmed_p1_rounds} P2:{self.confirmed_p2_rounds}")

        return None

    def get_current_state(self):
        """Get the current confirmed round state"""
        return self.confirmed_p1_rounds, self.confirmed_p2_rounds

    def reset(self):
        """Reset round state for new match"""
        self.confirmed_p1_rounds = 0
        self.confirmed_p2_rounds = 0
        self.candidate_state = None
        self.candidate_start_time = None
        print("🔄 RoundState reset: P1:0 P2:0")

# ─── MATCH TRACKING LOGIC ──────────────────────────────────────────────────
class MatchTracker:
    def __init__(self):
        self.match_number = 1
        self.p1_match_wins = 0
        self.p2_match_wins = 0

        print(f"🏆 MatchTracker initialized: Match #{self.match_number}")

    def check_match_end(self, p1_rounds, p2_rounds):
        """
        Check if current round state indicates match end.
        Returns: None or ("match_over", winner)
        """
        if p1_rounds >= 2:
            self.p1_match_wins += 1
            result = ("match_over", "p1")
            print(f"🏁 MATCH #{self.match_number} OVER: P1 wins {p1_rounds}-{p2_rounds}!")
            print(f"📊 Overall Matches: P1:{self.p1_match_wins} P2:{self.p2_match_wins}")
            self.match_number += 1
            return result

        elif p2_rounds >= 2:
            self.p2_match_wins += 1
            result = ("match_over", "p2")
            print(f"🏁 MATCH #{self.match_number} OVER: P2 wins {p2_rounds}-{p1_rounds}!")
            print(f"📊 Overall Matches: P1:{self.p1_match_wins} P2:{self.p2_match_wins}")
            self.match_number += 1
            return result

        return None

    def get_stats(self):
        """Get current match statistics"""
        return {
            'current_match': self.match_number,
            'p1_wins': self.p1_match_wins,
            'p2_wins': self.p2_match_wins,
            'total_matches': self.p1_match_wins + self.p2_match_wins
        }

# ─── ROUND DETECTION (SAME AS BEFORE) ──────────────────────────────────────
def detect_round_indicators(frame):
    """Simple round detection using same pattern as health bars"""
    results = {}

    for name, (x1, y1, x2, y2) in ROUND_INDICATORS.items():
        # Extract region
        region = frame[y1:y2, x1:x2]

        # Create mask for red pixels
        mask = cv2.inRange(region, RED_BGR_LOWER, RED_BGR_UPPER)

        # Count red pixels and calculate percentage
        total_pixels = region.shape[0] * region.shape[1]
        if total_pixels > 0:
            red_pixels = cv2.countNonZero(mask)
            red_pct = red_pixels / total_pixels * 100.0
        else:
            red_pct = 0.0

        # Simple threshold: >50% red = won round
        results[name] = red_pct > 50.0

    return results

# ─── DXCAM SETUP ────────────────────────────────────────────────────────────
comtypes.CoInitialize()
camera = dxcam.create(output_color="BGR")
camera.start(target_fps=60, region=REGION, video_mode=True)
atexit.register(lambda: (camera.stop(), comtypes.CoUninitialize()))

# ─── MAIN TEST LOOP ────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("ROUND & MATCH LOGIC TEST")
    print("=" * 60)
    print("Round indicators:")
    for name, coords in ROUND_INDICATORS.items():
        print(f"  {name}: {coords}")
    print(f"Red range: {RED_BGR_LOWER} - {RED_BGR_UPPER}")
    print(f"Confirmation time: 1.0 seconds")
    print()
    print("Press Ctrl+C to stop")
    print("=" * 60)

    # Initialize state trackers
    round_state = RoundState()
    match_tracker = MatchTracker()

    last_print_time = 0

    try:
        while True:
            frame = camera.get_latest_frame()
            if frame is None:
                time.sleep(0.01)
                continue

            # Detect round indicators
            round_indicators = detect_round_indicators(frame)

            # Count current rounds for each player
            detected_p1_rounds = sum([round_indicators['p1_round1'], round_indicators['p1_round2']])
            detected_p2_rounds = sum([round_indicators['p2_round1'], round_indicators['p2_round2']])

            # Update round state with new detection
            round_result = round_state.update(detected_p1_rounds, detected_p2_rounds)

            # Check for round win
            if round_result and round_result[0] == "round_won":
                _, winner, p1_rounds, p2_rounds = round_result

                # Check for match end
                match_result = match_tracker.check_match_end(p1_rounds, p2_rounds)

                if match_result and match_result[0] == "match_over":
                    print(f"🆕 Starting Match #{match_tracker.match_number}")
                    round_state.reset()

            # Print current status every 2 seconds
            current_time = time.time()
            if current_time - last_print_time > 2.0:
                confirmed_p1, confirmed_p2 = round_state.get_current_state()
                stats = match_tracker.get_stats()

                print(f"\n[Status] Match #{stats['current_match']} | Rounds P1:{confirmed_p1} P2:{confirmed_p2} | Overall P1:{stats['p1_wins']} P2:{stats['p2_wins']}")
                print(f"[Raw Detection] P1:{detected_p1_rounds} P2:{detected_p2_rounds}")

                last_print_time = current_time

            time.sleep(0.05)  # 20 FPS for testing

    except KeyboardInterrupt:
        print("\n" + "=" * 60)
        print("TEST COMPLETED")

        # Final statistics
        stats = match_tracker.get_stats()
        confirmed_p1, confirmed_p2 = round_state.get_current_state()

        print(f"Final State:")
        print(f"  Current Match: #{stats['current_match']}")
        print(f"  Current Rounds: P1:{confirmed_p1} P2:{confirmed_p2}")
        print(f"  Total Matches Played: {stats['total_matches']}")
        print(f"  Match Record: P1:{stats['p1_wins']} P2:{stats['p2_wins']}")

        if stats['total_matches'] > 0:
            p1_winrate = stats['p1_wins'] / stats['total_matches'] * 100
            print(f"  P1 Win Rate: {p1_winrate:.1f}%")

        print("=" * 60)

    camera.stop()

if __name__ == "__main__":
    main()