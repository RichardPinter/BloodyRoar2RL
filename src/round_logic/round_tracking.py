#!/usr/bin/env python3
"""
Round and match tracking logic for the game.
Handles round detection, validation, and match progression.
"""
import time
from src.logging_utils import log_state, log_round, log_match, log_debug


class RoundState:
    """Manages round detection and confirmation logic"""

    def __init__(self):
        # Confirmed round counts
        self.confirmed = {'p1': 0, 'p2': 0}

        # Full-health hold path (optional, still supported)
        self.FULL_HOLD_SEC = 0.30       # require brief hold at full before confirming
        self.health_full_since = None   # when both became "full enough"
        self.full_hold_armed = False

        # "First to zero" detection
        self.first_to_zero_flag = None      # 'p1' or 'p2'
        self.both_at_zero = False           # currently both at 0%
        self.round_winner_decided = False   # round already finalized?

        # Previous healths (for simultaneous-zero tie-break)
        self.prev_pct1 = 100.0
        self.prev_pct2 = 100.0

        # Bars-lost timeout confirm (loading screens etc.)
        self.BARS_LOST_CONFIRM_SEC = 10.0   # confirm after long bars-gone period
        self.zero_bars_since = None         # when both bars reached 0%

        # Legacy timer-based indicator fallback support
        self.cand = {
            'p1': {'count': None, 'start': None},
            'p2': {'count': None, 'start': None},
        }
        self.CONFIRMATION_TIME = 0.3

        # Cooldown after confirming a round
        self.last_round_end_time = None

        log_state("🔄 RoundState initialized: P1:0 P2:0 (flag + full-hold + restoration/timeout)")

    # ---------- basic helpers ----------
    def clear_candidate(self, player):
        self.cand[player] = {'count': None, 'start': None}

    def clear_all_candidates(self):
        self.clear_candidate('p1')
        self.clear_candidate('p2')

    def has_round_recently_ended(self, timeout=2.0):
        if self.last_round_end_time is None:
            return False
        return (time.time() - self.last_round_end_time) < timeout

    def get_current_state(self):
        """Return current confirmed round counts (p1, p2)."""
        return self.confirmed['p1'], self.confirmed['p2']

    def reset(self):
        """Reset round state for a new match."""
        self.confirmed = {'p1': 0, 'p2': 0}
        self.first_to_zero_flag = None
        self.both_at_zero = False
        self.round_winner_decided = False
        self.prev_pct1 = 100.0
        self.prev_pct2 = 100.0
        self.clear_all_candidates()
        self.last_round_end_time = None
        self.health_full_since = None
        self.full_hold_armed = False
        self.zero_bars_since = None
        log_state("🔄 RoundState reset for new match (flags/holds cleared)")

    # ---------- finalize helpers ----------
    def _confirm_from_flag(self, reason, pct1, pct2, validation_callback=None):
        """
        Confirm the round now using the existing first_to_zero_flag.
        Winner = NOT the player who went to zero first.
        """
        if self.first_to_zero_flag is None or self.round_winner_decided:
            return None

        winner = 'p2' if self.first_to_zero_flag == 'p1' else 'p1'
        self.confirmed[winner] += 1

        log_round(
            f"ROUND END: {winner.upper()} WINS! "
            f"({self.first_to_zero_flag.upper()} went to zero first) [{reason}]"
        )
        log_round(f"Score: P1={self.confirmed['p1']} P2={self.confirmed['p2']}")

        if validation_callback:
            validation_callback(
                system_prediction=winner.upper(),
                p1_health=pct1,
                p2_health=pct2,
                first_to_zero_flag=self.first_to_zero_flag,
                both_at_zero=True
            )

        # finalize & clean
        self.round_winner_decided = True
        self.last_round_end_time = time.time()
        self.first_to_zero_flag = None
        self.both_at_zero = False
        self.full_hold_armed = False
        self.health_full_since = None
        self.zero_bars_since = None

        # update prev healths
        self.prev_pct1 = pct1
        self.prev_pct2 = pct2

        return ("round_won", winner, self.confirmed['p1'], self.confirmed['p2'])

    def finalize_on_restoration(self, pct1, pct2, validation_callback=None):
        """
        Call this when the agent detects 'health restoration'.
        If there is a pending first_to_zero_flag, confirm the previous round now.
        """
        return self._confirm_from_flag("restoration", pct1, pct2, validation_callback)

    # ---------- primary update (flag + full-hold + timeout) ----------
    def update_first_to_zero(self, pct1, pct2, validation_callback=None):
        """
        Main detection path.
        - Set first_to_zero_flag when one side hits ~0 first.
        - Track both-at-zero; if both at 0 for long enough (loading screen),
          confirm via timeout.
        - Or confirm via full-health brief hold (legacy path).
        """
        # Apply cooldown after a confirmed round
        if self.has_round_recently_ended(timeout=5.0):
            self.prev_pct1 = pct1
            self.prev_pct2 = pct2
            return None

        if not self.round_winner_decided:
            # Establish first-to-zero flag (≤2%)
            if pct1 <= 2.0 and pct2 > 2.0 and self.first_to_zero_flag is None:
                self.first_to_zero_flag = 'p1'
                log_round(f"FLAG SET: P1 went to zero first! (P1={pct1:.1f}% P2={pct2:.1f}%)")
            elif pct2 <= 2.0 and pct1 > 2.0 and self.first_to_zero_flag is None:
                self.first_to_zero_flag = 'p2'
                log_round(f"FLAG SET: P2 went to zero first! (P1={pct1:.1f}% P2={pct2:.1f}%)")
            elif pct1 <= 2.0 and pct2 <= 2.0 and self.first_to_zero_flag is None:
                # Simultaneous → tie-break using previous frame
                log_round(
                    f"SIMULTANEOUS ZERO: P1={pct1:.1f}% P2={pct2:.1f}%, "
                    f"prev P1={self.prev_pct1:.1f}% P2={self.prev_pct2:.1f}%"
                )
                if self.prev_pct1 < self.prev_pct2:
                    self.first_to_zero_flag = 'p1'
                    log_round("FLAG SET: P1 was closer to zero on previous frame")
                elif self.prev_pct2 < self.prev_pct1:
                    self.first_to_zero_flag = 'p2'
                    log_round("FLAG SET: P2 was closer to zero on previous frame")
                else:
                    self.first_to_zero_flag = 'p1'  # deterministic fallback
                    log_round("FLAG SET: Exactly tied; defaulting to P1")

            # Allow pre-both-zero recovery to clear a false alarm
            if self.first_to_zero_flag == 'p1' and pct1 > 50.0 and pct2 > 50.0 and not self.both_at_zero:
                log_debug(f"FLAG RESET: P1 recovered pre-both-zero (P1={pct1:.1f}% P2={pct2:.1f}%)")
                self.first_to_zero_flag = None
            elif self.first_to_zero_flag == 'p2' and pct2 > 50.0 and pct1 > 50.0 and not self.both_at_zero:
                log_debug(f"FLAG RESET: P2 recovered pre-both-zero (P1={pct1:.1f}% P2={pct2:.1f}%)")
                self.first_to_zero_flag = None

            # Track both-at-zero state and confirm on long timeout if flagged
            if pct1 <= 0.0 and pct2 <= 0.0:
                if not self.both_at_zero:
                    self.both_at_zero = True
                    self.zero_bars_since = time.time()
                    log_round(f"BOTH AT ZERO: P1={pct1:.1f}% P2={pct2:.1f}% | Flag={self.first_to_zero_flag}")
                else:
                    if (self.first_to_zero_flag is not None and
                        self.zero_bars_since and
                        (time.time() - self.zero_bars_since) >= self.BARS_LOST_CONFIRM_SEC):
                        result = self._confirm_from_flag("bars_lost_timeout", pct1, pct2, validation_callback)
                        if result:
                            return result
            else:
                # Left both-zero; keep state open for restoration path handled by agent
                if self.both_at_zero:
                    self.both_at_zero = False

            # Full-health brief hold (legacy path still supported)
            if self.zero_bars_since is not None:
                both_full = (pct1 >= 95.0 and pct2 >= 95.0)
                if both_full:
                    if not self.full_hold_armed:
                        self.full_hold_armed = True
                        self.health_full_since = time.time()
                        log_round(
                            f"FULL HEALTH ARMED: P1={pct1:.1f}% P2={pct2:.1f}% "
                            f"| Flag={self.first_to_zero_flag} | Hold={self.FULL_HOLD_SEC:.2f}s"
                        )
                    else:
                        held = time.time() - (self.health_full_since or time.time())
                        if held >= self.FULL_HOLD_SEC:
                            # Confirm if we have a flag; otherwise ask for fallback
                            if self.first_to_zero_flag is not None:
                                return self._confirm_from_flag("full_hold", pct1, pct2, validation_callback)
                            else:
                                log_round("⚠️ FALLBACK: Full health held but no first_to_zero_flag; use indicator fallback")
                                # reset hold to avoid loops
                                self.full_hold_armed = False
                                self.health_full_since = None
                                self.zero_bars_since = None
                                self.prev_pct1 = pct1
                                self.prev_pct2 = pct2
                                return ("fallback_needed", None, self.confirmed['p1'], self.confirmed['p2'])
                else:
                    if self.full_hold_armed:
                        held = time.time() - (self.health_full_since or time.time())
                        log_debug(
                            f"FULL HEALTH ABORTED after {held:.2f}s "
                            f"(need {self.FULL_HOLD_SEC:.2f}s). Now P1={pct1:.1f}% P2={pct2:.1f}%"
                        )
                    self.full_hold_armed = False
                    self.health_full_since = None

        # Ready again once both are high after a decision
        if self.round_winner_decided and pct1 >= 95.0 and pct2 >= 95.0:
            log_debug(f"ROUND RESET: Ready for next round (P1={pct1:.1f}% P2={pct2:.1f}%)")
            self.round_winner_decided = False

        # Update previous healths
        self.prev_pct1 = pct1
        self.prev_pct2 = pct2
        return None
    
    def finalize_on_restoration(self, pct1: float, pct2: float, validation_callback=None):
        """
        Call this when health bars 'come back' (round transition).
        If we had a first_to_zero_flag, lock in the round winner now.
        """
        if self.first_to_zero_flag is None:
            log_round("ℹRestoration seen but no first_to_zero_flag — cannot confirm previous round.")
            return None
        return self._confirm_from_flag("restoration", pct1, pct2, validation_callback)


    def finalize_on_bars_missing(self, missing_secs: float, validation_callback=None, threshold_sec: float = 10.0):
        """
        Call this when bars have been missing for a while (e.g., loading/new opponent).
        If a flag exists and we've exceeded the threshold, confirm the round by flag.
        """
        if self.first_to_zero_flag is None:
            return None
        if missing_secs >= threshold_sec:
            log_round(f"Health bars hidden for {missing_secs:.1f}s — confirming round by flag.")
            # We don't have meaningful pct values here; use zeros for logging.
            return self._confirm_from_flag("long_zero_missing", 0.0, 0.0, validation_callback)
        return None

    # ---------- legacy indicator-timer path (unchanged) ----------
    def update(self, detected_p1, detected_p2):
        """
        LEGACY: Old timer-based round detection system (kept as fallback)
        Returns: None or ("round_won", winner, p1_rounds, p2_rounds)
        """
        now = time.time()
        det = {'p1': detected_p1, 'p2': detected_p2}

        if self.has_round_recently_ended(timeout=5.0):
            log_debug(f"Round cooldown active - ignoring detection for {5.0 - (now - self.last_round_end_time):.1f}s more")
            return None

        if (detected_p1, detected_p2) != (self.confirmed['p1'], self.confirmed['p2']):
            log_debug(f"Round detection: P1={detected_p1}, P2={detected_p2}, "
                      f"confirmed=(P1:{self.confirmed['p1']}, P2:{self.confirmed['p2']})")

        for p in ('p1', 'p2'):
            if det[p] > self.confirmed[p]:
                if det[p] != self.confirmed[p] + 1:
                    log_debug(f"⚠️ INVALID: {p.upper()} rounds jumped from {self.confirmed[p]} to {det[p]} - ignoring!")
                    self.clear_candidate(p)
                    continue
                c = self.cand[p]
                if c['count'] == det[p]:
                    elapsed = now - c['start']
                    if elapsed < 0.1 or int(elapsed * 10) != int((elapsed - 0.016) * 10):
                        log_debug(f"[Candidate {p.upper()}] {det['p1']}-{det['p2']} ({elapsed:.1f}s) - waiting…")
                else:
                    self.cand[p] = {'count': det[p], 'start': now}
                    log_debug(f"🔍 [New Candidate {p.upper()}] count={det[p]} > confirmed={self.confirmed[p]} - starting timer")
            else:
                if self.cand[p]['count'] is not None:
                    log_debug(f"[Cleared Candidate {p.upper()}] detected {det[p]} <= confirmed {self.confirmed[p]}")
                self.clear_candidate(p)

        ready = [
            (p, data['start'])
            for p, data in self.cand.items()
            if data['count'] is not None and (now - data['start']) >= self.CONFIRMATION_TIME
        ]
        if not ready:
            return None

        winner, start_ts = min(ready, key=lambda x: x[1])
        if len(ready) > 1:
            log_debug(f"Both timers ready—choosing first player: {winner.upper()}")

        old_p1, old_p2 = self.confirmed['p1'], self.confirmed['p2']
        new_count = self.cand[winner]['count']
        if new_count != self.confirmed[winner] + 1:
            log_debug(f"FINAL VALIDATION FAILED: {winner.upper()} would jump from {self.confirmed[winner]} to {new_count}")
            self.clear_all_candidates()
            return None

        self.confirmed[winner] = new_count
        self.last_round_end_time = now
        log_round(f"ROUND CONFIRMED: {winner.upper()} won! "
                  f"(P1:{self.confirmed['p1']} P2:{self.confirmed['p2']}) "
                  f"[was P1:{old_p1} P2:{old_p2}]")
        self.clear_all_candidates()
        return ("round_won", winner, self.confirmed['p1'], self.confirmed['p2'])


    
class MatchTracker:
    """Tracks match progression and statistics"""

    def __init__(self, start_match_number=1):
        self.match_number = start_match_number
        self.p1_match_wins = 0
        self.p2_match_wins = 0
        log_match(f"🏆 MatchTracker initialized: Match #{self.match_number}")

    def check_match_end(self, p1_rounds, p2_rounds):
        """
        Check if current round state indicates match end.
        Returns: None or ("match_over", winner)
        """
        log_debug(f'Inside match tracker P1:{p1_rounds}, P2:{p2_rounds}')
        if p1_rounds >= 2:
            self.p1_match_wins += 1
            result = ("match_over", "p1")
            log_match(f"MATCH #{self.match_number} OVER: P1 wins {p1_rounds}-{p2_rounds}!")
            log_match(f"Overall Matches: P1:{self.p1_match_wins} P2:{self.p2_match_wins}")
            self.match_number += 1
            return result
        elif p2_rounds >= 2:
            self.p2_match_wins += 1
            result = ("match_over", "p2")
            log_match(f"MATCH #{self.match_number} OVER: P2 wins {p1_rounds}-{p2_rounds}!")
            log_match(f"Overall Matches: P1:{self.p1_match_wins} P2:{self.p2_match_wins}")
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
