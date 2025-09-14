#!/usr/bin/env python3
"""
RoundStateLite — simpler round detection:
- Flags the first side to hit ~0% health.
- Confirms the round on either:
    • both-at-zero timeout (fade/load), or
    • health restoration (next round starts).
- Keeps confirmed round counts (first to 2 wins).
- Cooldown after confirmation to avoid double counts.

Drop-in for your agent:
    from round_state_lite import RoundStateLite as RoundState
"""

import time

try:
    # Prefer your structured logger if available
    from src.logging_utils import log_state, log_round, log_debug
except Exception:
    # Fallback to print to avoid import issues when testing standalone
    def log_state(msg): print(msg)
    def log_round(msg): print(msg)
    def log_debug(msg): print(msg)


class RoundStateLite:
    def __init__(
        self,
        zero_thresh: float = 2.0,            # ≤ this % is considered "zero"
        full_thresh: float = 95.0,           # ≥ this % is considered "full health"
        both_zero_confirm_sec: float = 3.0,  # confirm if both are zero for this long
        cooldown_sec: float = 2.0,           # ignore detections right after confirm
    ):
        self.zero_thresh = zero_thresh
        self.full_thresh = full_thresh
        self.both_zero_confirm_sec = both_zero_confirm_sec
        self.cooldown_sec = cooldown_sec

        self.confirmed = {'p1': 0, 'p2': 0}

        self.first_to_zero_flag = None    # 'p1' or 'p2'
        self.both_at_zero_since = None
        self.prev_pct1 = 100.0
        self.prev_pct2 = 100.0

        self.round_winner_decided = False
        self.last_round_end_time = None

        log_state("RoundStateLite initialized (simple first-to-zero + restoration/timeout)")

    # ------------- helpers -------------
    def has_round_recently_ended(self, timeout=None):
        t = self.cooldown_sec if timeout is None else timeout
        return (self.last_round_end_time is not None
                and (time.time() - self.last_round_end_time) < t)

    def get_current_state(self):
        return self.confirmed['p1'], self.confirmed['p2']

    # --- legacy API compatibility (no-ops) ---
    def clear_candidate(self, player=None):
        """Kept for compatibility with older agent code."""
        return None

    def clear_all_candidates(self):
        """Kept for compatibility with older agent code."""
        return None

    def reset(self):
        self.confirmed = {'p1': 0, 'p2': 0}
        self.first_to_zero_flag = None
        self.both_at_zero_since = None
        self.prev_pct1 = 100.0
        self.prev_pct2 = 100.0
        self.round_winner_decided = False
        self.last_round_end_time = None
        log_state("RoundStateLite reset for new match")

    # ------------- confirmation core -------------
    def _confirm_now(self, reason: str, pct1: float, pct2: float, validation_callback=None):
        if self.first_to_zero_flag is None or self.round_winner_decided:
            return None

        winner = 'p2' if self.first_to_zero_flag == 'p1' else 'p1'
        self.confirmed[winner] += 1

        log_round(f"🎯 ROUND END: {winner.upper()} WINS! [{reason}] "
                  f"(flag={self.first_to_zero_flag}, P1={pct1:.1f}%, P2={pct2:.1f}%)")
        log_round(f"📊 Score: P1={self.confirmed['p1']} P2={self.confirmed['p2']}")

        if validation_callback:
            validation_callback(
                system_prediction=winner.upper(),
                p1_health=pct1,
                p2_health=pct2,
                first_to_zero_flag=self.first_to_zero_flag,
                both_at_zero=True
            )

        self.round_winner_decided = True
        self.last_round_end_time = time.time()

        # clear flags for next round
        self.first_to_zero_flag = None
        self.both_at_zero_since = None

        # remember for tie-breaks
        self.prev_pct1 = pct1
        self.prev_pct2 = pct2

        return ("round_won", winner, self.confirmed['p1'], self.confirmed['p2'])

    # ------------- public API used by your agent -------------
    def finalize_on_restoration(self, pct1: float, pct2: float, validation_callback=None):
        """Call when both bars come back to 'full' after being zero."""
        return self._confirm_now("restoration", pct1, pct2, validation_callback)

    def finalize_on_bars_missing(self, missing_secs: float, validation_callback=None, threshold_sec: float = 10.0):
        """Call when health bars have been missing for a while."""
        if self.first_to_zero_flag is None:
            return None
        if missing_secs >= threshold_sec:
            log_round(f"⏱️ Bars hidden for {missing_secs:.1f}s — confirming by flag.")
            return self._confirm_now("bars_missing_timeout", 0.0, 0.0, validation_callback)
        return None

    def update(self, pct1: float, pct2: float):
        """
        Feed current health percents each frame.
        Returns: None or ("round_won", winner, p1_rounds, p2_rounds)
        """
        # Cooldown after confirmation
        if self.has_round_recently_ended():
            self.prev_pct1, self.prev_pct2 = pct1, pct2
            return None

        # 1) Set first-to-zero flag
        if self.first_to_zero_flag is None:
            if pct1 <= self.zero_thresh and pct2 > self.zero_thresh:
                self.first_to_zero_flag = 'p1'
                log_round(f"🚩 FLAG: P1 reached zero first (P1={pct1:.1f} P2={pct2:.1f})")
            elif pct2 <= self.zero_thresh and pct1 > self.zero_thresh:
                self.first_to_zero_flag = 'p2'
                log_round(f"🚩 FLAG: P2 reached zero first (P1={pct1:.1f} P2={pct2:.1f})")
            elif pct1 <= self.zero_thresh and pct2 <= self.zero_thresh:
                # tie-break using previous frame (who was closer to zero)
                if self.prev_pct1 < self.prev_pct2:
                    self.first_to_zero_flag = 'p1'
                    log_round("🚩 FLAG: simultaneous zero; prev P1 lower → P1 first")
                elif self.prev_pct2 < self.prev_pct1:
                    self.first_to_zero_flag = 'p2'
                    log_round("🚩 FLAG: simultaneous zero; prev P2 lower → P2 first")
                else:
                    self.first_to_zero_flag = 'p1'  # deterministic fallback
                    log_round("🚩 FLAG: exact tie; default P1")

        # 2) Track both-at-zero duration
        if pct1 <= self.zero_thresh and pct2 <= self.zero_thresh:
            if self.both_at_zero_since is None:
                self.both_at_zero_since = time.time()
                log_debug(f"💀 BOTH ZERO — starting timer")
            else:
                if self.first_to_zero_flag and (time.time() - self.both_at_zero_since) >= self.both_zero_confirm_sec:
                    return self._confirm_now("both_zero_timeout", pct1, pct2, None)
        else:
            self.both_at_zero_since = None

        # 3) Allow re-arming after restoration to full
        if self.round_winner_decided and pct1 >= self.full_thresh and pct2 >= self.full_thresh:
            self.round_winner_decided = False
            log_debug("✅ Ready for next round (both full)")

        # Update previous
        self.prev_pct1, self.prev_pct2 = pct1, pct2
        return None


class MatchTrackerLite:
    """Same behavior as your MatchTracker, minimal logging."""
    def __init__(self, start_match_number=1, rounds_to_win=2):
        self.match_number = start_match_number
        self.p1_match_wins = 0
        self.p2_match_wins = 0
        self.rounds_to_win = rounds_to_win
        log_state(f"MatchTrackerLite initialized: Match #{self.match_number}")

    def check_match_end(self, p1_rounds, p2_rounds):
        if p1_rounds >= self.rounds_to_win:
            self.p1_match_wins += 1
            self.match_number += 1
            log_state(f"🏁 MATCH OVER: P1 wins {p1_rounds}-{p2_rounds}")
            return ("match_over", "p1")
        if p2_rounds >= self.rounds_to_win:
            self.p2_match_wins += 1
            self.match_number += 1
            log_state(f"🏁 MATCH OVER: P2 wins {p1_rounds}-{p2_rounds}")
            return ("match_over", "p2")
        return None
