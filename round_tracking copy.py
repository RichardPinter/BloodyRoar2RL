#!/usr/bin/env python3
"""
Round and match tracking logic for the game.
Handles round detection, validation, and match progression.
"""
import time
from logging_utils import log_state, log_round, log_match, log_debug

class RoundState:
    """Manages round detection and confirmation logic"""
    
    def __init__(self):
        # Confirmed round counts
        self.confirmed = {'p1': 0, 'p2': 0}
        
        # Full-health hold (new)
        self.FULL_HOLD_SEC = 0.30      # require 300ms at "full" before confirming
        self.health_full_since = None  # timestamp when both became "full enough"
        self.full_hold_armed = False   # are we currently holding at full?
        
        # "First to zero" detection system
        self.first_to_zero_flag = None     # 'p1' or 'p2' - who went to zero first
        self.both_at_zero = False          # Are both players currently at zero?
        self.round_winner_decided = False  # Has this round been decided?
        
        # Track previous health values for simultaneous zero handling
        self.prev_pct1 = 100.0
        self.prev_pct2 = 100.0
        
        # Timer-based system (kept for legacy indicator fallback)
        self.cand = {
            'p1': {'count': None, 'start': None},
            'p2': {'count': None, 'start': None},
        }
        self.CONFIRMATION_TIME = 0.3
        
        # Track when we last confirmed a round
        self.last_round_end_time = None

        # --- NEW: visibility / intermission tracking ---
        self.last_health_seen_ts = None   # last time any health looked > 2%
        self.disappear_since = None       # when both bars looked "gone"
        self.flag_set_ts = None           # when first_to_zero_flag was set

        # --- NEW: tunables for visibility/intermission ---
        self.VIS_SEEN_THRESH = 2.0         # any health > 2% counts as "seen"
        self.VISIBILITY_RECENCY_SEC = 1.5  # how recent "seen" must be to treat 0/0 as real KO
        self.INTERMISSION_TIMEOUT = 10.0   # bars-gone timeout to award by flag
        self.FLAG_FRESH_SEC = 5.0          # flag must be this fresh when bars disappear
        
        log_state(f"🔄 RoundState initialized: P1:0 P2:0 (first-to-zero + full-hold)")
    
    def clear_candidate(self, player):
        self.cand[player] = {'count': None, 'start': None}
    
    def clear_all_candidates(self):
        self.clear_candidate('p1')
        self.clear_candidate('p2')
        # add inside RoundState
    
    def has_round_recently_ended(self, timeout=2.0):
        if self.last_round_end_time is None:
            return False
        return (time.time() - self.last_round_end_time) < timeout
    
    def get_current_state(self):
        """Return the current confirmed round counts."""
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
        # full-hold flags
        self.health_full_since = None
        self.full_hold_armed = False
        # NEW: visibility/intermission
        self.last_health_seen_ts = None
        self.disappear_since = None
        self.flag_set_ts = None
        log_state(f"🔄 RoundState reset for new match (flags + full-hold cleared)")
    
    def update_first_to_zero(self, pct1, pct2, validation_callback=None):
        """
        Primary: first-to-zero with full-health hold before confirming the round.
        validation_callback: optional function to call for validation GUI
        """
        now = time.time()

        # Track when we last saw "real" health (any bar > threshold)
        if max(pct1, pct2) > self.VIS_SEEN_THRESH:
            self.last_health_seen_ts = now

        # Apply cooldown after a confirmed round
        if self.has_round_recently_ended(timeout=5.0):
            # still update prevs for tie-breaks
            self.prev_pct1 = pct1
            self.prev_pct2 = pct2
            # clear intermission timer while cooling down
            self.disappear_since = None
            return None
        
        if not self.round_winner_decided:
            # Check who goes to ~zero first (≤2%)
            if pct1 <= 2.0 and pct2 > 2.0 and self.first_to_zero_flag is None:
                self.first_to_zero_flag = 'p1'
                self.flag_set_ts = now
                log_round(f"🚩 FLAG SET: P1 went to zero first! (P1={pct1:.1f}% P2={pct2:.1f}%)")
            elif pct2 <= 2.0 and pct1 > 2.0 and self.first_to_zero_flag is None:
                self.first_to_zero_flag = 'p2'
                self.flag_set_ts = now
                log_round(f"🚩 FLAG SET: P2 went to zero first! (P1={pct1:.1f}% P2={pct2:.1f}%)")
            elif pct1 <= 2.0 and pct2 <= 2.0 and self.first_to_zero_flag is None:
                # Simultaneous → tie-break with previous frame
                log_round(
                    f"⚠️ SIMULTANEOUS ZERO: P1={pct1:.1f}% P2={pct2:.1f}%, "
                    f"prev P1={self.prev_pct1:.1f}% P2={self.prev_pct2:.1f}%"
                )
                if self.prev_pct1 < self.prev_pct2:
                    self.first_to_zero_flag = 'p1'
                    log_round("🚩 FLAG SET: P1 was closer to zero on previous frame")
                elif self.prev_pct2 < self.prev_pct1:
                    self.first_to_zero_flag = 'p2'
                    log_round("🚩 FLAG SET: P2 was closer to zero on previous frame")
                else:
                    self.first_to_zero_flag = 'p1'  # deterministic fallback
                    log_round("🚩 FLAG SET: Exactly tied; defaulting to P1")
                self.flag_set_ts = now
            
            # Reset flag only if flagged player clearly recovers before both-at-zero locks in
            if self.first_to_zero_flag == 'p1' and pct1 > 50.0 and pct2 > 50.0 and not self.both_at_zero:
                log_debug(f"FLAG RESET: P1 recovered pre-both-zero (P1={pct1:.1f}% P2={pct2:.1f}%)")
                self.first_to_zero_flag = None
                self.flag_set_ts = None
            elif self.first_to_zero_flag == 'p2' and pct2 > 50.0 and pct1 > 50.0 and not self.both_at_zero:
                log_debug(f"FLAG RESET: P2 recovered pre-both-zero (P1={pct1:.1f}% P2={pct2:.1f}%)")
                self.first_to_zero_flag = None
                self.flag_set_ts = None
            
            # Distinguish "true both-zero" vs "bars disappeared"
            if pct1 <= 0.0 and pct2 <= 0.0:
                recent_seen = (
                    self.last_health_seen_ts is not None and
                    (now - self.last_health_seen_ts) <= self.VISIBILITY_RECENCY_SEC
                )

                if recent_seen:
                    # Treat as real double-zero → normal path
                    if not self.both_at_zero:
                        self.both_at_zero = True
                        log_round(
                            f"💀 BOTH AT ZERO: P1={pct1:.1f}% P2={pct2:.1f}% | "
                            f"Flag={self.first_to_zero_flag}"
                        )
                else:
                    # Bars likely disappeared (intro/intermission). Start/advance the timer.
                    if self.disappear_since is None:
                        self.disappear_since = now
                        log_debug("Health bars disappeared—starting intermission timer")
                    else:
                        waited = now - self.disappear_since
                        if waited >= self.INTERMISSION_TIMEOUT:
                            # Only award if we have a fresh flag from right before the bars vanished
                            if (self.first_to_zero_flag is not None and
                                self.flag_set_ts is not None and
                                (self.disappear_since - self.flag_set_ts) <= self.FLAG_FRESH_SEC and
                                not self.round_winner_decided):

                                winner = 'p2' if self.first_to_zero_flag == 'p1' else 'p1'
                                self.confirmed[winner] += 1
                                log_round(f"⏱️ INTERMISSION TIMEOUT: awarding round to {winner.upper()} by first-to-zero flag")
                                log_round(f"📊 Score: P1={self.confirmed['p1']} P2={self.confirmed['p2']}")

                                # finalize round
                                self.round_winner_decided = True
                                self.last_round_end_time = now

                                # clear transient stuff
                                self.first_to_zero_flag = None
                                self.flag_set_ts = None
                                self.both_at_zero = False
                                self.full_hold_armed = False
                                self.health_full_since = None
                                self.disappear_since = None

                                # update prevs
                                self.prev_pct1 = pct1
                                self.prev_pct2 = pct2

                                if validation_callback:
                                    validation_callback(
                                        system_prediction=winner.upper(),
                                        p1_health=pct1, p2_health=pct2,
                                        first_to_zero_flag="INTERMISSION_FALLBACK",
                                        both_at_zero=False
                                    )
                                return ("round_won", winner, self.confirmed['p1'], self.confirmed['p2'])
                            else:
                                # No fresh flag → don’t guess; drop to indicator fallback
                                log_round("Intermission timeout but no fresh flag—using indicator fallback next")
                                self.disappear_since = None
                                self.prev_pct1 = pct1
                                self.prev_pct2 = pct2
                                return ("fallback_needed", None, self.confirmed['p1'], self.confirmed['p2'])
            else:
                # bars visible again → clear intermission timer
                self.disappear_since = None
            
            # Round end path: after both-at-zero, require both ≥95% and hold briefly
            if self.both_at_zero:
                both_full = (pct1 >= 95.0 and pct2 >= 95.0)
                
                if both_full:
                    if not self.full_hold_armed:
                        self.full_hold_armed = True
                        self.health_full_since = now
                        log_round(
                            f"🟩 FULL HEALTH ARMED: P1={pct1:.1f}% P2={pct2:.1f}% "
                            f"| Flag={self.first_to_zero_flag} | Hold={self.FULL_HOLD_SEC:.2f}s"
                        )
                    else:
                        held = now - (self.health_full_since or now)
                        if held >= self.FULL_HOLD_SEC:
                            log_round(
                                f"💚 FULL HEALTH CONFIRMED ({held:.2f}s ≥ {self.FULL_HOLD_SEC:.2f}s): "
                                f"P1={pct1:.1f}% P2={pct2:.1f}% | Flag={self.first_to_zero_flag}"
                            )
                            if self.first_to_zero_flag is not None:
                                # Winner = NOT the one who hit zero first
                                winner = 'p2' if self.first_to_zero_flag == 'p1' else 'p1'
                                self.confirmed[winner] += 1
                                
                                log_round(f"🎯 ROUND END: {winner.upper()} WINS! ({self.first_to_zero_flag.upper()} went to zero first)")
                                log_round(f"📊 Score: P1={self.confirmed['p1']} P2={self.confirmed['p2']}")
                                
                                # Request manual validation if callback provided
                                if validation_callback:
                                    validation_callback(
                                        system_prediction=winner.upper(),
                                        p1_health=pct1,
                                        p2_health=pct2,
                                        first_to_zero_flag=self.first_to_zero_flag,
                                        both_at_zero=self.both_at_zero
                                    )
                                
                                # Reset for next round
                                self.first_to_zero_flag = None
                                self.flag_set_ts = None
                                self.both_at_zero = False
                                self.round_winner_decided = True
                                self.last_round_end_time = now
                                
                                # Clear full-hold flags
                                self.full_hold_armed = False
                                self.health_full_since = None
                                
                                # Update prevs and return
                                self.prev_pct1 = pct1
                                self.prev_pct2 = pct2
                                return ("round_won", winner, self.confirmed['p1'], self.confirmed['p2'])
                            else:
                                # Full health held, but no first_to_zero flag → fallback
                                log_round("⚠️ FALLBACK: Full health confirmed but no first_to_zero_flag; using indicator fallback...")
                                self.both_at_zero = False
                                self.full_hold_armed = False
                                self.health_full_since = None
                                self.prev_pct1 = pct1
                                self.prev_pct2 = pct2
                                return ("fallback_needed", None, self.confirmed['p1'], self.confirmed['p2'])
                else:
                    # Dipped below full during the hold → abort
                    if self.full_hold_armed:
                        held = now - (self.health_full_since or now)
                        log_debug(
                            f"🟥 FULL HEALTH ABORTED after {held:.2f}s (need {self.FULL_HOLD_SEC:.2f}s). "
                            f"Now P1={pct1:.1f}% P2={pct2:.1f}%"
                        )
                    self.full_hold_armed = False
                    self.health_full_since = None
        
        # Reset round decision readiness when both are high again
        if self.round_winner_decided and pct1 >= 95.0 and pct2 >= 95.0:
            log_debug(f"ROUND RESET: Ready for next round (P1={pct1:.1f}% P2={pct2:.1f}%)")
            self.round_winner_decided = False
        
        # Update previous health values for next frame
        self.prev_pct1 = pct1
        self.prev_pct2 = pct2
        
        return None
    
    def update(self, detected_p1, detected_p2):
        """
        LEGACY: Old timer-based round detection system (kept as fallback)
        Returns: None or ("round_won", winner, p1_rounds, p2_rounds)
        """
        now = time.time()
        det = {'p1': detected_p1, 'p2': detected_p2}
        
        # COOLDOWN: Don't process if we recently confirmed a round
        if self.has_round_recently_ended(timeout=5.0):
            log_debug(f"Round cooldown active - ignoring detection for {5.0 - (now - self.last_round_end_time):.1f}s more")
            return None
        
        # debug: log whenever raw detection differs from confirmed
        if (detected_p1, detected_p2) != (self.confirmed['p1'], self.confirmed['p2']):
            log_debug(f"Round detection: P1={detected_p1}, P2={detected_p2}, "
                      f"confirmed=(P1:{self.confirmed['p1']}, P2:{self.confirmed['p2']})")
        
        # HIGH WATERMARK: Only consider as candidate if HIGHER than confirmed
        for p in ('p1','p2'):
            if det[p] > self.confirmed[p]:
                # VALIDATION: Rounds must increment by exactly 1
                if det[p] != self.confirmed[p] + 1:
                    log_debug(f"⚠️ INVALID: {p.upper()} rounds jumped from {self.confirmed[p]} to {det[p]} - ignoring!")
                    self.clear_candidate(p)
                    continue
                
                c = self.cand[p]
                # same candidate continuing?
                if c['count'] == det[p]:
                    elapsed = now - c['start']
                    # occasional debug heartbeat
                    if elapsed < 0.1 or int(elapsed * 10) != int((elapsed - 0.016) * 10):
                        log_debug(f"⏳ [Candidate {p.upper()}] {det['p1']}-{det['p2']} "
                                  f"({elapsed:.1f}s) - waiting for confirmation...")
                else:
                    # new candidate for this player
                    self.cand[p] = {'count': det[p], 'start': now}
                    log_debug(f"🔍 [New Candidate {p.upper()}] count={det[p]} > confirmed={self.confirmed[p]} - starting timer")
            else:
                # if detected is not higher than confirmed, clear candidate
                if self.cand[p]['count'] is not None:
                    log_debug(f"🚫 [Cleared Candidate {p.upper()}] detected {det[p]} <= "
                              f"confirmed {self.confirmed[p]}")
                self.clear_candidate(p)
        
        # collect any timers past CONFIRMATION_TIME
        ready = [
            (p, data['start'])
            for p, data in self.cand.items()
            if data['count'] is not None and (now - data['start']) >= self.CONFIRMATION_TIME
        ]
        if not ready:
            return None
        
        # pick whoever's timer started first (tie-breaker)
        winner, start_ts = min(ready, key=lambda x: x[1])
        
        # guard against both firing exactly together
        if len(ready) > 1:
            log_debug(f"⚠️ Both timers ready—choosing first player: {winner.upper()}")
        
        # confirm the win - but validate first
        old_p1, old_p2 = self.confirmed['p1'], self.confirmed['p2']
        new_count = self.cand[winner]['count']
        
        # Final validation before confirming
        if new_count != self.confirmed[winner] + 1:
            log_debug(f"⚠️ FINAL VALIDATION FAILED: {winner.upper()} would jump from {self.confirmed[winner]} to {new_count}")
            self.clear_all_candidates()
            return None
        
        self.confirmed[winner] = new_count
        self.last_round_end_time = now
        log_round(f"🎯 ROUND CONFIRMED: {winner.upper()} won! "
                  f"(P1:{self.confirmed['p1']} P2:{self.confirmed['p2']}) "
                  f"[was P1:{old_p1} P2:{old_p2}]")
        
        # clear all so no ghost confirmations
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
            log_match(f"🏁 MATCH #{self.match_number} OVER: P1 wins {p1_rounds}-{p2_rounds}!")
            log_match(f"📊 Overall Matches: P1:{self.p1_match_wins} P2:{self.p2_match_wins}")
            self.match_number += 1
            return result
        
        elif p2_rounds >= 2:
            self.p2_match_wins += 1
            result = ("match_over", "p2")
            log_match(f"🏁 MATCH #{self.match_number} OVER: P2 wins {p1_rounds}-{p2_rounds}!")
            log_match(f"📊 Overall Matches: P1:{self.p1_match_wins} P2:{self.p2_match_wins}")
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
