#!/usr/bin/env python3
import os
import time
import atexit
import csv
import threading
from queue import Queue
from collections import deque
import random
import re
import os
import comtypes
import dxcam
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter  
from PIL import Image
import logging
from logging.handlers import RotatingFileHandler
from datetime import datetime
action_to_send = "start\n"
# ─── LOGGING SETUP ─────────────────────────────────────────────────────────
# EASY TOGGLE: Set to False to disable file logging
ENABLE_LOGGING = True

# Create logs directory
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

# Set up logging with rotation (keeps last 5 files of 50MB each)
log_filename = os.path.join(LOG_DIR, f"game_debug_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

# Create a custom logger
logger = logging.getLogger('GameDebug')
logger.setLevel(logging.DEBUG)

# Console handler (always enabled)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_formatter = logging.Formatter('%(message)s')
console_handler.setFormatter(console_formatter)
logger.addHandler(console_handler)

if ENABLE_LOGGING:
    # File handler with rotation
    file_handler = RotatingFileHandler(
        log_filename, 
        maxBytes=200*1024*1024,  # 200MB per file
        backupCount=10,          # Keep 10 files (2GB total)
        encoding='utf-8'
    )
    file_handler.setLevel(logging.DEBUG)
    
    # Create formatter
    file_formatter = logging.Formatter(
        '%(asctime)s.%(msecs)03d | %(levelname)-8s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Add formatter to handler
    file_handler.setFormatter(file_formatter)
    
    # Add handler to logger
    logger.addHandler(file_handler)

# Create separate loggers for different components
round_logger = logging.getLogger('GameDebug.Round')
match_logger = logging.getLogger('GameDebug.Match')
state_logger = logging.getLogger('GameDebug.State')
learner_logger = logging.getLogger('GameDebug.Learner')

# ─── HELPER FUNCTIONS ─────────────────────────────────────────────────────
def log_round(message, *args, **kwargs):
    """Log round-related messages"""
    round_logger.info(message, *args, **kwargs)

def log_match(message, *args, **kwargs):
    """Log match-related messages"""
    match_logger.info(message, *args, **kwargs)

def log_state(message, *args, **kwargs):
    """Log state changes"""
    state_logger.info(message, *args, **kwargs)

def log_learner(message, *args, **kwargs):
    """Log learner messages"""
    learner_logger.info(message, *args, **kwargs)

def log_debug(message, *args, **kwargs):
    """Log debug messages (only to file)"""
    logger.debug(message, *args, **kwargs)

# ─── CONFIG ────────────────────────────────────────────────────────────────
REGION        = (0, 0, 624, 548)      # x, y, width, height
Y_HEALTH      = 116
X1_P1, X2_P1  = 73, 292
X1_P2, X2_P2  = 355, 574
LEN_P1        = X2_P1 - X1_P1
LEN_P2        = X2_P2 - X1_P2

LOWER_BGR     = np.array([0,150,180], dtype=np.uint8)
UPPER_BGR     = np.array([30,175,220], dtype=np.uint8)

FRAME_STACK     = 10
CNN_SIZE        = (84, 84)
ACTIONS         = ['special', "kick", "transform", "jump", "squat", "left", "right"]
NUM_ACTIONS     = len(ACTIONS)

DEVICE          = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LOG_CSV         = "health_results.csv"
ACTIONS_FILE    = "actions.txt"
MAX_FRAMES      = 1000
GAMMA           = 0.99
BATCH_SIZE      = 32
TARGET_SYNC     = 1000
REPLAY_SIZE     = 10000
MIN_BUFFER_SIZE = 100
HEALTH_LIMIT    = 98.0  # Lowered from 99.0 for better detection

LEARNING_RATE   = 1e-4

MODEL_DIR = "../models"
os.makedirs(MODEL_DIR, exist_ok=True)
MODEL_NUMBER = 126
os.makedirs(MODEL_DIR, exist_ok=True)
LOAD_CHECKPOINT = f"{MODEL_DIR}model_match_{MODEL_NUMBER}.pth"
TEST_MODE       = False

writer = SummaryWriter(log_dir="../logs")

ROUND_INDICATORS = {
    'p1_round1': (270, 135, 278, 140),
    'p1_round2': (245, 135, 253, 140),
    'p2_round1': (373, 135, 381, 140),
    'p2_round2': (396, 135, 404, 140),
}
LOW1 = np.array([0, 70, 50], dtype=np.uint8)
HIGH1 = np.array([10, 255, 255], dtype=np.uint8)
LOW2 = np.array([170, 70, 50], dtype=np.uint8)
HIGH2 = np.array([180, 255, 255], dtype=np.uint8)

# HSV range for blue indicators
BLUE_LOW = np.array([100, 50, 50], dtype=np.uint8)
BLUE_HIGH = np.array([130, 255, 255], dtype=np.uint8)

RED_BGR_LOWER   = np.array([0, 0, 150], dtype=np.uint8)
RED_BGR_UPPER   = np.array([60, 60, 255], dtype=np.uint8)

# ─── TRANSFORM STATE & BLACK‐PIXEL HELPERS ─────────────────────────────────
PIXEL_RECTS = [
    ("P1_R1_pixel", 71, 475, 72, 476),
    ("P2_R2_pixel", 520, 475, 521, 476),
]
STATE_MAP = {
    (200, 200, 200): "can transform",
    ( 48,  48, 248): "transformed",
    (240, 128,   0): "cannot transform",
}

AREA_RECTS = [
    ("P1_R1_area", 71, 480, 177, 481),
    ("P2_R2_area", 469, 480, 575, 481),
]
BLACK_BGR = np.array([0, 0, 8], dtype=np.uint8)

def classify_transform_state(frame):
    """
    Return { 'P1': 'can transform' | 'transformed' | 'cannot transform' | 'unknown',
             'P2': ... }
    """
    out = {}
    for player, x1, y1, x2, y2 in PIXEL_RECTS:
        b, g, r = frame[y1, x1]
        out[player] = STATE_MAP.get((int(b), int(g), int(r)), "unknown")
    return out

def compute_black_stats(frame):
    """
    Return { 'P1': pct_black, 'P2': pct_black },
    and separate channel‐range dict if you like.
    """
    pct_out = {}
    range_out = {}
    for player, x1, y1, x2, y2 in AREA_RECTS:
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            pct_out[player] = None
            range_out[player] = None
            continue

        mask = cv2.inRange(roi, BLACK_BGR, BLACK_BGR)
        cnt = int(cv2.countNonZero(mask))
        total = roi.shape[0] * roi.shape[1]
        pct = cnt / total * 100.0
        pct_out[player] = pct

        B, G, R = roi[:,:,0], roi[:,:,1], roi[:,:,2]
        range_out[player] = {
            'B': (int(B.min()), int(B.max())),
            'G': (int(G.min()), int(G.max())),
            'R': (int(R.min()), int(R.max())),
        }
    return pct_out, range_out

# ─── DQN NET & BUFFER ─────────────────────────────────────────────────────
class DQNNet(nn.Module):
    def __init__(self, in_ch, n_actions, extra_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, 32, 8, 4)
        self.conv2 = nn.Conv2d(32, 64, 4, 2)
        self.conv3 = nn.Conv2d(64, 64, 3, 1)
        conv_out   = 64 * 7 * 7
        # now fc1 expects conv_out + extra_dim
        self.fc1   = nn.Linear(conv_out + extra_dim, 512)
        self.out   = nn.Linear(512, n_actions)

    def forward(self, x, extra):
        # x: (B, in_ch, H, W), extra: (B, extra_dim)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.flatten(1)                   # (B, conv_out)
        x = torch.cat([x, extra], dim=1)   # (B, conv_out+extra_dim)
        x = F.relu(self.fc1(x))
        return self.out(x)

class ReplayBuffer:
    def __init__(self, size, extra_dim):
        self.size      = size
        self.extra_dim = extra_dim
        self.clear()

    def clear(self):
        self.states      = np.zeros((self.size, FRAME_STACK, *CNN_SIZE), dtype=np.float32)
        self.extras      = np.zeros((self.size, self.extra_dim), dtype=np.float32)  # new
        self.actions     = np.zeros(self.size, dtype=np.int64)
        self.rewards     = np.zeros(self.size, dtype=np.float32)
        self.next_states = np.zeros((self.size, FRAME_STACK, *CNN_SIZE), dtype=np.float32)
        self.next_extras  = np.zeros((self.size, self.extra_dim), dtype=np.float32)
        self.dones       = np.zeros(self.size, dtype=bool)
        self.ptr = 0
        self.len = 0

    def add(self, s, extra, a, r, s2, next_extra, done):
        self.states[self.ptr]      = s
        self.extras[self.ptr]      = extra
        self.actions[self.ptr]     = a
        self.rewards[self.ptr]     = r
        self.next_states[self.ptr] = s2
        self.next_extras[self.ptr] = next_extra
        self.dones[self.ptr]       = done
        self.ptr = (self.ptr + 1) % self.size
        self.len = min(self.len + 1, self.size)

    def sample(self, batch_size):
        idx = np.random.randint(0, self.len, size=batch_size)
        return (
            torch.from_numpy(self.states[idx]).to(DEVICE),
            torch.from_numpy(self.extras[idx]).to(DEVICE),     # new
            torch.from_numpy(self.actions[idx]).to(DEVICE),
            torch.from_numpy(self.rewards[idx]).to(DEVICE),
            torch.from_numpy(self.next_states[idx]).to(DEVICE),
            torch.from_numpy(self.next_extras[idx]).to(DEVICE),
            torch.from_numpy(self.dones[idx].astype(np.uint8)).to(DEVICE),
        )

# ─── ROBUST ROUND & MATCH LOGIC ───────────────────────────────────────────
class RoundState:
    def __init__(self):
        # confirmed round counts
        self.confirmed = {'p1': 0, 'p2': 0}
        
        # NEW: Baseline-based round detection
        self.baseline_indicators = None  # Captured at round start
        self.pending_round_changes = {'p1': 0, 'p2': 0}  # Changes detected during death period
        self.in_death_period = False  # Are both players currently at 0% health?
        
        # OLD: Timer-based system (keeping for compatibility/fallback)
        self.cand = {
            'p1': {'count': None, 'start': None},
            'p2': {'count': None, 'start': None},
        }
        self.CONFIRMATION_TIME = 1.5

        # track when we last confirmed a round
        self.last_round_end_time = None

        log_state(f"🔄 RoundState initialized: P1:0 P2:0 (baseline system)")

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
        """Return the current confirmed round counts."""
        return self.confirmed['p1'], self.confirmed['p2']

    def reset(self):
        """Reset round state for a new match."""
        self.confirmed = {'p1': 0, 'p2': 0}
        self.baseline_indicators = None
        self.pending_round_changes = {'p1': 0, 'p2': 0}
        self.in_death_period = False
        self.clear_all_candidates()
        self.last_round_end_time = None
        log_state(f"🔄 RoundState reset for new match: P1:0 P2:0 (baseline cleared)")
        
    def capture_baseline(self, indicator_states):
        """Capture current indicator states as baseline for this round"""
        self.baseline_indicators = indicator_states.copy()
        log_state(f"📸 Baseline captured: {indicator_states}")
        
    def check_for_changes(self, current_indicators, both_at_zero):
        """Compare current indicators to baseline and update pending changes"""
        if self.baseline_indicators is None:
            log_debug("No baseline set - capturing current as baseline")
            self.capture_baseline(current_indicators)
            return
            
        # Update death period status
        self.in_death_period = both_at_zero
        
        # Only process changes during death period
        if not self.in_death_period:
            return
            
        # Check for changes from baseline - use "not blue" as primary method
        # Since blue detection is reliable, any indicator that's not blue is likely won
        p1_baseline_rounds = sum(1 for k, v in self.baseline_indicators.items() 
                                if k.startswith('p1_') and v != 'blue')
        p2_baseline_rounds = sum(1 for k, v in self.baseline_indicators.items() 
                                if k.startswith('p2_') and v != 'blue')
        
        p1_current_rounds = sum(1 for k, v in current_indicators.items() 
                               if k.startswith('p1_') and v != 'blue')
        p2_current_rounds = sum(1 for k, v in current_indicators.items() 
                               if k.startswith('p2_') and v != 'blue')
        
        # Also track red-only for logging
        p1_red_only = sum(1 for k, v in current_indicators.items() 
                         if k.startswith('p1_') and v == 'red')
        p2_red_only = sum(1 for k, v in current_indicators.items() 
                         if k.startswith('p2_') and v == 'red')
        
        # Detect changes
        p1_change = p1_current_rounds - p1_baseline_rounds
        p2_change = p2_current_rounds - p2_baseline_rounds
        
        # Update pending changes if we detect increments
        if p1_change > 0 and p1_change != self.pending_round_changes['p1']:
            log_debug(f"🔍 P1 indicator change detected: +{p1_change} from baseline "
                     f"(not-blue: {p1_current_rounds}, red: {p1_red_only}) during death")
            self.pending_round_changes['p1'] = p1_change
            
        if p2_change > 0 and p2_change != self.pending_round_changes['p2']:
            log_debug(f"🔍 P2 indicator change detected: +{p2_change} from baseline "
                     f"(not-blue: {p2_current_rounds}, red: {p2_red_only}) during death")
            self.pending_round_changes['p2'] = p2_change
            
    def confirm_pending_changes(self):
        """Apply any pending changes and return winner info"""
        if self.pending_round_changes['p1'] == 0 and self.pending_round_changes['p2'] == 0:
            return None
            
        # Validate that only one player can win a round
        if self.pending_round_changes['p1'] > 0 and self.pending_round_changes['p2'] > 0:
            log_debug("⚠️ INVALID: Both players have pending changes - ignoring all")
            self.pending_round_changes = {'p1': 0, 'p2': 0}
            return None
            
        # Apply the change
        winner = None
        if self.pending_round_changes['p1'] > 0:
            # Validate increment is exactly 1
            if self.pending_round_changes['p1'] == 1:
                self.confirmed['p1'] += 1
                winner = 'p1'
                log_round(f"🎯 ROUND CONFIRMED: P1 won! (P1:{self.confirmed['p1']} P2:{self.confirmed['p2']})")
            else:
                log_debug(f"⚠️ INVALID: P1 pending change is {self.pending_round_changes['p1']}, not 1")
                
        if self.pending_round_changes['p2'] > 0:
            # Validate increment is exactly 1
            if self.pending_round_changes['p2'] == 1:
                self.confirmed['p2'] += 1
                winner = 'p2'
                log_round(f"🎯 ROUND CONFIRMED: P2 won! (P1:{self.confirmed['p1']} P2:{self.confirmed['p2']})")
            else:
                log_debug(f"⚠️ INVALID: P2 pending change is {self.pending_round_changes['p2']}, not 1")
        
        # Clear pending changes
        self.pending_round_changes = {'p1': 0, 'p2': 0}
        
        # Update timing
        if winner:
            self.last_round_end_time = time.time()
            return ("round_won", winner, self.confirmed['p1'], self.confirmed['p2'])
        
        return None
    
    def update_with_baseline(self, indicator_states, both_at_zero):
        """NEW: Baseline-based update method"""
        # Apply 5-second cooldown
        if self.has_round_recently_ended(timeout=5.0):
            return None
            
        # Check for changes against baseline
        self.check_for_changes(indicator_states, both_at_zero)
        
        # Don't return anything here - confirmation happens at round start
        return None
    
    def on_round_start(self, indicator_states):
        """Called when health restoration detected - confirms pending and captures new baseline"""
        # First, confirm any pending changes from previous round
        result = self.confirm_pending_changes()
        
        # Then capture new baseline for starting round
        self.capture_baseline(indicator_states)
        
        return result

    def update(self, detected_p1, detected_p2):
        """
        Update round state with new detection.
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

        # 1) HIGH WATERMARK: Only consider as candidate if HIGHER than confirmed
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

        # 2) collect any timers past CONFIRMATION_TIME
        ready = [
            (p, data['start'])
            for p, data in self.cand.items()
            if data['count'] is not None and (now - data['start']) >= self.CONFIRMATION_TIME
        ]
        if not ready:
            return None

        # 3) pick whoever's timer started first (tie-breaker)
        winner, start_ts = min(ready, key=lambda x: x[1])

        # guard against both firing exactly together
        if len(ready) > 1:
            log_debug(f"⚠️ Both timers ready—choosing first player: {winner.upper()}")

        # 4) confirm the win - but validate first
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

# ─── MatchTracker ROUND & MATCH LOGIC ───────────────────────────────────────────
class MatchTracker:
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

# ─── MATCH LOGGER FOR PERSISTENT STATISTICS ─────────────────────────────────
class MatchLogger:
    def __init__(self, log_file="match_history.csv"):
        self.log_file = log_file
        self.ensure_csv_exists()
        
    def ensure_csv_exists(self):
        """Create CSV with headers if it doesn't exist"""
        if not os.path.exists(self.log_file):
            with open(self.log_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'timestamp', 'match_number', 'round_number', 
                    'round_winner', 'p1_rounds', 'p2_rounds',
                    'match_winner', 'match_complete', 'notes'
                ])
            log_state(f"📊 Created match history log: {self.log_file}")
    
    def log_round(self, match_number, round_number, winner, p1_rounds, p2_rounds, notes=""):
        """Log a round result"""
        with open(self.log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                datetime.now().isoformat(),
                match_number,
                round_number,
                winner,
                p1_rounds,
                p2_rounds,
                "",  # match_winner (empty for round logs)
                False,  # match_complete
                notes
            ])
    
    def log_match(self, match_number, winner, p1_rounds, p2_rounds):
        """Log a match result"""
        with open(self.log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                datetime.now().isoformat(),
                match_number,
                "",  # round_number (empty for match logs)
                "",  # round_winner (empty for match logs)
                p1_rounds,
                p2_rounds,
                winner,
                True,  # match_complete
                f"Match ended {p1_rounds}-{p2_rounds}"
            ])
    
    def get_statistics(self):
        """Read the CSV and return cumulative statistics"""
        if not os.path.exists(self.log_file):
            return None
            
        p1_matches = 0
        p2_matches = 0
        total_rounds = 0
        
        with open(self.log_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row['match_complete'] == 'True':
                    if row['match_winner'] == 'p1':
                        p1_matches += 1
                    elif row['match_winner'] == 'p2':
                        p2_matches += 1
                elif row['round_winner']:
                    total_rounds += 1
        
        return {
            'p1_match_wins': p1_matches,
            'p2_match_wins': p2_matches,
            'total_matches': p1_matches + p2_matches,
            'total_rounds': total_rounds,
            'p1_win_rate': p1_matches / (p1_matches + p2_matches) if (p1_matches + p2_matches) > 0 else 0
        }

def detect_round_indicators(frame):
    """Round detection using HSV color space, returns color states (blue/red)"""
    results = {}
    states = {}
    debug_info = []

    for name, (x1, y1, x2, y2) in ROUND_INDICATORS.items():
        # Extract region
        region = frame[y1:y2, x1:x2]

        # Convert to HSV for more reliable color detection
        hsv_region = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)

        # Red detection: wraps around in HSV, so we need two masks
        red_mask1 = cv2.inRange(hsv_region, LOW1, HIGH1)
        red_mask2 = cv2.inRange(hsv_region, LOW2, HIGH2)
        red_mask = cv2.bitwise_or(red_mask1, red_mask2)
        
        # Blue detection
        blue_mask = cv2.inRange(hsv_region, BLUE_LOW, BLUE_HIGH)

        # Count pixels and calculate percentages
        total_pixels = region.shape[0] * region.shape[1]
        if total_pixels > 0:
            red_pixels = cv2.countNonZero(red_mask)
            red_pct = red_pixels / total_pixels * 100.0
            
            blue_pixels = cv2.countNonZero(blue_mask)
            blue_pct = blue_pixels / total_pixels * 100.0
        else:
            red_pct = 0.0
            blue_pct = 0.0

        # Determine state: prioritize red over blue with improved thresholds
        if red_pct > 50.0:  # Lowered from 85% for better detection
            states[name] = 'red'
            results[name] = True  # Keep backward compatibility
        elif blue_pct > 30.0:  # Blue detection works well
            states[name] = 'blue'
            results[name] = False
        else:
            # FALLBACK: If not clearly blue and has some red, assume it's red
            if blue_pct < 10.0 and red_pct > 20.0:
                states[name] = 'red'  # Likely red but weak signal
                results[name] = True
                debug_info.append(f"{name}: FALLBACK RED - R:{red_pct:.1f}% B:{blue_pct:.1f}%")
            else:
                states[name] = 'unknown'
                results[name] = False
                # Log unknowns for debugging
                if red_pct > 5 or blue_pct > 5:
                    log_debug(f"⚠️ Unknown state {name}: R:{red_pct:.1f}% B:{blue_pct:.1f}% - needs tuning")

        # Debug logging for calibration
        if red_pct > 5 or blue_pct > 5:
            debug_info.append(f"{name}: R:{red_pct:.1f}% B:{blue_pct:.1f}% -> {states[name]}")
    
    # Log all debug info at once to reduce file writes
    if debug_info:
        log_debug(f"Round indicators: {', '.join(debug_info)}")

    # Return both for compatibility: results (bool) and states (color)
    return results, states

# ─── SETUP ───────────────────────────────────────────────────────────────
import re

EXTRA_DIM = 4  # [P1_state, P2_state, P1_black_pct, P2_black_pct]
policy_net = DQNNet(FRAME_STACK, NUM_ACTIONS, EXTRA_DIM).to(DEVICE).train()
target_net = DQNNet(FRAME_STACK, NUM_ACTIONS, EXTRA_DIM).to(DEVICE)

# Load checkpoint if specified
start_match_number = 1
if LOAD_CHECKPOINT and os.path.exists(LOAD_CHECKPOINT):
    checkpoint = torch.load(LOAD_CHECKPOINT, map_location=DEVICE)
    policy_net.load_state_dict(checkpoint)
    target_net.load_state_dict(checkpoint)
    log_state(f"✅ Loaded checkpoint from {LOAD_CHECKPOINT}")

    # Extract match number from filename if possible
    match = re.search(r'model_match_(\d+)', LOAD_CHECKPOINT)
    if match:
        start_match_number = int(match.group(1)) + 1
        log_state(f"   Continuing from match {start_match_number}")
else:
    log_debug(f"Checkpoint exists: {os.path.exists(LOAD_CHECKPOINT)}")
    log_debug(f"Checkpoint path: {LOAD_CHECKPOINT}")
    target_net.load_state_dict(policy_net.state_dict())
    if LOAD_CHECKPOINT:
        log_state(f"⚠️  Checkpoint {LOAD_CHECKPOINT} not found, training from scratch")

optimizer  = torch.optim.Adam(policy_net.parameters(), lr=LEARNING_RATE)
buffer = ReplayBuffer(REPLAY_SIZE, EXTRA_DIM)

# ─── DXCAM SETUP ────────────────────────────────────────────────────────────
comtypes.CoInitialize()
camera = dxcam.create(output_color="BGR")
camera.start(target_fps=60, region=REGION, video_mode=True)
atexit.register(lambda: (camera.stop(), comtypes.CoUninitialize()))

# ─── SHARED STATE ───────────────────────────────────────────────────────────
frame_q     = Queue(maxsize=16)
frame_stack = deque(maxlen=FRAME_STACK)
results     = []
screenshots = []
stop_event  = threading.Event()
round_end_event = threading.Event()
match_end_event = threading.Event()
match_number = start_match_number

# Global tracking for TensorBoard
global_step = 0
episode_number = 0

# ─── PRODUCER ────────────────────────────────────────────────────────────────
def producer():
    start = time.perf_counter()
    while not stop_event.is_set():
        frm = camera.get_latest_frame()
        if frm is not None:
            ts = time.perf_counter() - start
            frame_q.put((frm.copy(), ts))

# ─── SINGLE CONSUMER w/ FIXED ROUND LOGIC ────────────────────────────────────
def consumer():
    global match_number, global_step, episode_number
    
    # Log startup
    log_state("Consumer thread started")
    log_debug(f"Initial match_number: {match_number}")
    
    # Track how many rounds (episodes) we've seen
    episode_count = 1

    # STATE MACHINE with new fallback state:
    # - "waiting_for_match": Waiting for first round of a new match
    # - "waiting_for_round": Previous round ended, waiting for next round
    # - "active": Round is in progress
    # - "post_match_waiting": Match ended, navigating menus
    # - "health_detection_lost": Cannot detect health bars, fallback mode
    state = "waiting_for_match"
    
    # IMPORTANT: Track state before entering fallback
    state_before_fallback = None
    
    alive_since  = None
    death_since  = None
    
    # Health detection fallback tracking
    health_lost_since = None
    HEALTH_LOST_TIMEOUT = 5.0  # 5 seconds before entering fallback mode
    fallback_action_count = 0

    # Proper state tracking for rewards
    prev_state   = None
    prev_action  = None
    prev_extra_feats = None
    prev_pct1    = 100.0
    prev_pct2    = 100.0
    
    # Health-based round detection
    both_at_zero_since = None
    DEATH_THRESHOLD = 0.0      # Both must be exactly 0%
    ALIVE_THRESHOLD = 98.0     # Both must be >98% to confirm round start (matches HEALTH_LIMIT)
    ZERO_HEALTH_DURATION = 1.0 # Must be at 0% for 1 second

    # Initialize robust round and match tracking
    round_state = RoundState()
    match_tracker = MatchTracker(start_match_number=match_number)
    match_logger = MatchLogger()  # Persistent CSV logging
    log_debug(f"MatchTracker initialized with match_number: {match_number}")
    
    # Log startup statistics
    stats = match_logger.get_statistics()
    if stats:
        log_state(f"📊 Historical stats: P1 wins: {stats['p1_match_wins']}, P2 wins: {stats['p2_match_wins']}, "
                 f"Win rate: {stats['p1_win_rate']:.1%}, Total rounds: {stats['total_rounds']}")
    
    # Step 1: create a named window for Q‑values
    cv2.namedWindow("Q-Values", cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow("Rewards",  cv2.WINDOW_AUTOSIZE)

    # compute where to put it:
    x, y, w, h = REGION     # REGION = (0,0,624,548)
    win_x = x
    win_y = y + h + 100     # leave a 100px buffer below the capture area

    cv2.moveWindow("Q-Values", win_x, win_y)
    cv2.moveWindow("Rewards", win_x+300, win_y)

    def write_action(text: str):
        try:
            with open(ACTIONS_FILE, "w") as f:
                f.write(text)
        except Exception as e:
            log_debug(f"ERROR writing action: {e}")

    # ── TRAINING HOOKS ────
    def on_round_end():
        round_end_event.set()  # Just signal the learner thread

    def on_match_end():
        match_end_event.set()  # Signal the learner thread for match end
    # ──────────────  Reward Tracking ─────────────────
    reward_history = deque(maxlen=100)

    round_reward = 0.0
    round_start_time = time.time()
    round_steps = 0
    
    # Action tracking for distribution
    action_counts = np.zeros(NUM_ACTIONS)
    
    # ────────────────────────────────────────────────
    hold_counter   = 0       # frames left to hold the current action
    current_action = None  
    
    # In consumer(), before while loop
    write_count = 0
    
    # Post-match state tracking
    post_match_entry_logged = False
    post_match_action_count = 0
    
    # Health logging timer
    last_health_log_time = time.time()
    
    while not stop_event.is_set():
        # Initialize flags for this frame
        health_restoration_detected = False
        
        # Get frame with timeout to avoid blocking during post-match and fallback
        frame = None
        ts = None
        try:
            frame, ts = frame_q.get(timeout=0.1)
        except:
            # No frame available - handle post_match and fallback actions
            if state == "post_match_waiting" or state == "health_detection_lost":
                if  action_to_send == "start\n":
                     action_to_send = "kick\n"
                else:
                    action_to_send = "kick\n"
                write_action(action_to_send)
            continue
        
        # If we got here, we have a frame - process it normally
        global_step += 1
        
        # 1) Compute health % once
        strip = frame[Y_HEALTH:Y_HEALTH+1]
        m1    = cv2.inRange(strip[:, X1_P1:X2_P1], LOWER_BGR, UPPER_BGR)
        m2    = cv2.inRange(strip[:, X1_P2:X2_P2], LOWER_BGR, UPPER_BGR)
        pct1  = cv2.countNonZero(m1) / LEN_P1 * 100.0
        pct2  = cv2.countNonZero(m2) / LEN_P2 * 100.0

        # Log health every second
        current_time = time.time()
        if current_time - last_health_log_time >= 1.0:
            log_state(f"Health: P1={pct1:.1f}%, P2={pct2:.1f}%")
            last_health_log_time = current_time

        # HEALTH-BASED ROUND DETECTION
        # Check if both players are at 0% health
        if pct1 <= DEATH_THRESHOLD and pct2 <= DEATH_THRESHOLD:
            if both_at_zero_since is None:
                both_at_zero_since = time.time()
                log_state(f"⚠️ Both players at 0% health - tracking... (P1:{pct1:.1f}%, P2:{pct2:.1f}%)")
        else:
            # Check if we're coming back from a confirmed death period
            if both_at_zero_since is not None:
                time_at_zero = time.time() - both_at_zero_since
                
                # Log current health values when they're no longer both at zero
                if time_at_zero < ZERO_HEALTH_DURATION:
                    log_debug(f"Health recovery too quick: P1:{pct1:.1f}%, P2:{pct2:.1f}% after {time_at_zero:.2f}s")
                
                # If both are now alive AND were dead for required duration
                if (pct1 >= ALIVE_THRESHOLD and pct2 >= ALIVE_THRESHOLD and 
                    time_at_zero >= ZERO_HEALTH_DURATION):
                    
                    log_round(f"🎯 ROUND START DETECTED via health restoration! "
                             f"(were at 0% for {time_at_zero:.1f}s, now P1:{pct1:.1f}% P2:{pct2:.1f}%)")
                    
                    # We need to get the indicator states first - will be handled later in frame processing
                    health_restoration_detected = True
                    
                both_at_zero_since = None
                
        # Debug log health values periodically
        if global_step % 60 == 0:
            log_debug(f"Health check: P1={pct1:.1f}%, P2={pct2:.1f}%, State={state}, "
                     f"Zero tracking={'Yes' if both_at_zero_since else 'No'}")

        # HEALTH DETECTION FALLBACK LOGIC
        # Check if both health bars are undetectable (0%)
        if pct1 == 0.0 and pct2 == 0.0:
            if health_lost_since is None:
                health_lost_since = time.time()
                log_state("⚠️ Health bars lost - starting timer...")
            else:
                time_lost = time.time() - health_lost_since
                if time_lost >= HEALTH_LOST_TIMEOUT and state != "health_detection_lost":
                    # SAVE THE CURRENT STATE BEFORE ENTERING FALLBACK
                    state_before_fallback = state
                    log_state(f"🚨 HEALTH DETECTION LOST for {time_lost:.1f}s - Entering fallback mode! (was in state: {state})")
                    state = "health_detection_lost"
                    fallback_action_count = 0
                    # Clear any active round state
                    alive_since = None
                    death_since = None
                    current_action = None
                    hold_counter = 0
        else:
            # Health bars detected - check if we need to exit fallback mode
            if state == "health_detection_lost":
                if pct1 >= HEALTH_LIMIT and pct2 >= HEALTH_LIMIT:
                    log_state(f"✅ Health bars restored! P1={pct1:.1f}% P2={pct2:.1f}% - Exiting fallback mode")

                    # Clear any in-flight round candidates
                    round_state.clear_all_candidates()

                    # Re-compute indicators
                    round_indicators, indicator_states = detect_round_indicators(frame)
                    detected_p1_rounds = sum([round_indicators['p1_round1'], round_indicators['p1_round2']])
                    detected_p2_rounds = sum([round_indicators['p2_round1'], round_indicators['p2_round2']])

                    # Special handling if we were in post_match_waiting before fallback
                    if state_before_fallback == "post_match_waiting":
                        # We were navigating post-match menus, so check if we're at a new match
                        if detected_p1_rounds == 0 and detected_p2_rounds == 0:
                            log_state("   → Post-match navigation complete, new match detected")
                            round_state.reset()  # CRITICAL: Reset for new match!
                            state = "waiting_for_match"
                        else:
                            # Still in post-match state
                            log_state("   → Returning to post_match_waiting")
                            state = "post_match_waiting"
                    else:
                        # Normal fallback exit logic for other states
                        confirmed_p1, confirmed_p2 = round_state.get_current_state()
                        
                        if detected_p1_rounds == 0 and detected_p2_rounds == 0:
                            # Check if round state needs reset (shouldn't happen in normal flow)
                            if confirmed_p1 > 0 or confirmed_p2 > 0:
                                log_state("   → WARNING: Clearing round indicators with non-zero confirmed state")
                            state = "waiting_for_match"
                            log_state("   → Returning to waiting_for_match (no round indicators)")
                        elif (detected_p1_rounds > confirmed_p1 or
                            detected_p2_rounds > confirmed_p2):
                            state = "active"
                            log_state("   → Returning to active (round in progress)")
                        else:
                            # Detected rounds match confirmed - we're between rounds
                            state = "waiting_for_round"
                            log_state(f"   → Returning to waiting_for_round (P1:{detected_p1_rounds} P2:{detected_p2_rounds})")

                    health_lost_since = None
                    fallback_action_count = 0
                    state_before_fallback = None
                else:
                    # still partial health—stay in fallback
                    if fallback_action_count % 50 == 0:
                        log_debug(
                        f"Health partially restored (P1={pct1:.1f}% P2={pct2:.1f}%) but not at limit"
                        )
                        
            elif health_lost_since is not None:
                # Health bars are back but not in fallback mode yet
                log_debug(f"Health bars restored before timeout (was lost for {time.time() - health_lost_since:.1f}s)")
                health_lost_since = None

        # Handle fallback state actions
        if state == "health_detection_lost":
            # Alternate between kick and start
            time_rounded = round(time.time(), 2)
            time_int = int(time_rounded * 100)
            
            if time_int % 2 == 0:
                action_to_write = "kick\n"
            else:
                action_to_write = "start\n"
            
            # Debug: Log every 50th action
            fallback_action_count += 1
            if fallback_action_count % 50 == 0:
                log_debug(f"Fallback action #{fallback_action_count}: {action_to_write.strip()} (P1={pct1:.1f}% P2={pct2:.1f}%)")
            
            write_action(action_to_write)
            
            # Skip the rest of the processing while in fallback mode
            try:
                frame_q.task_done()
            except:
                pass
            continue

        # 2) ALWAYS detect rounds - STATE INDEPENDENT!
        round_indicators, indicator_states = detect_round_indicators(frame)
        detected_p1_rounds = sum([round_indicators['p1_round1'], round_indicators['p1_round2']])
        detected_p2_rounds = sum([round_indicators['p2_round1'], round_indicators['p2_round2']])
        
        # Log periodically
        if global_step % 30 == 0:
            log_debug(f"Frame health: P1={pct1:.2f}%, P2={pct2:.2f}%")
            log_debug(f"Round indicators: {round_indicators}")
            log_debug(f"Indicator states: {indicator_states}")
            log_debug(f"Detected rounds: P1={detected_p1_rounds}, P2={detected_p2_rounds}")
            log_debug(f"Current state: {state}")
            log_debug(f"Baseline system - Confirmed: P1={round_state.confirmed['p1']} P2={round_state.confirmed['p2']}, "
                     f"Pending: P1={round_state.pending_round_changes['p1']} P2={round_state.pending_round_changes['p2']}, "
                     f"Death period: {round_state.in_death_period}")

        # 3) Update round state using NEW baseline system
        both_at_zero = (pct1 <= DEATH_THRESHOLD and pct2 <= DEATH_THRESHOLD)
        
        # Handle health restoration detected earlier
        round_result = None
        if health_restoration_detected:
            # ROUND START: Confirm pending changes and capture new baseline
            round_result = round_state.on_round_start(indicator_states)
            
            # Handle state transitions that were removed earlier
            if state in ["waiting_for_match", "waiting_for_round"]:
                log_state("Health restoration confirms round start - transitioning to active")
                state = "active"
                frame_stack.clear()
                # Reset tracking variables
                prev_state = None
                prev_action = None
                prev_pct1 = pct1
                prev_pct2 = pct2
                round_start_time = time.time()
                round_steps = 0
                round_reward = 0.0
                action_counts.fill(0)
                alive_since = None
                write_action("start\n")
        else:
            # DURING ROUND: Check for indicator changes during death period
            round_state.update_with_baseline(indicator_states, both_at_zero)

        # 4) Handle round completion if detected
        if round_result and round_result[0] == "round_won":
            _, winner, p1_rounds, p2_rounds = round_result
            
            log_round(f"[Episode] Round #{episode_count} total_reward={round_reward:.2f}")
            
            # Log to persistent CSV
            match_logger.log_round(
                match_tracker.match_number,
                episode_count,
                winner,
                p1_rounds,
                p2_rounds,
                f"reward={round_reward:.2f}"
            )
            
            episode_count += 1
            episode_number += 1
            
            # Log episode metrics
            round_duration = time.time() - round_start_time
            writer.add_scalar("episode/reward", round_reward, episode_number)
            writer.add_scalar("episode/length_steps", round_steps, episode_number)
            writer.add_scalar("episode/length_seconds", round_duration, episode_number)
            
            # Log action distribution for this episode
            if action_counts.sum() > 0:
                action_probs = action_counts / action_counts.sum()
                for i, action_name in enumerate(ACTIONS):
                    writer.add_scalar(f"actions/episode_distribution/{action_name}", 
                                    action_probs[i], episode_number)
            
            # 1) push this round's total reward
            reward_history.append(round_reward)
            round_reward = 0.0

            # 2) draw a small line‐chart of reward_history
            h, w = 150, 300
            graph = np.zeros((h, w, 3), dtype=np.uint8)
            if len(reward_history) > 1:
                mn = min(reward_history)
                mx = max(reward_history)
                span = mx - mn if mx != mn else 1.0
                pts = []
                for i, r in enumerate(reward_history):
                    x = int(i * (w-1) / (len(reward_history)-1))
                    y = h - 1 - int((r - mn) * (h-1) / span)
                    pts.append((x, y))
                cv2.polylines(graph, [np.array(pts, np.int32)], False, (0,255,0), 2)

                # optional: draw min/max labels
                cv2.putText(graph, f"{mx:.1f}", (5,15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180,180,180), 1)
                cv2.putText(graph, f"{mn:.1f}", (5,h-5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180,180,180), 1)

            cv2.imshow("Rewards", graph)
            cv2.waitKey(1)
            
            # Final transition with done=True and terminal reward
            if prev_state is not None and prev_action is not None:
                final_reward = 10.0 if winner == "p1" else -10.0
                buffer.add(prev_state, prev_extra_feats, prev_action, final_reward, prev_state, prev_extra_feats, True)
                
                # Log round outcome
                writer.add_scalar("episode/win", 1.0 if winner == "p1" else 0.0, episode_number)

            # round-ended hook
            on_round_end()

            # Check for match end
            match_result = match_tracker.check_match_end(p1_rounds, p2_rounds)
            log_debug(f"Match check result: {match_result} for rounds P1:{p1_rounds} P2:{p2_rounds}")

            if match_result and match_result[0] == "match_over":
                # Log match result to CSV
                _, match_winner = match_result
                match_logger.log_match(
                    match_tracker.match_number - 1,  # -1 because match_tracker already incremented
                    match_winner,
                    p1_rounds,
                    p2_rounds
                )
                
                # Match ended - save model and enter post-match navigation
                on_match_end()
                log_state(f"🎯 Match over! Entering post-match navigation mode...")
                state = "post_match_waiting"
                round_state.clear_all_candidates()
                alive_since = None
                death_since = None
            else:
                # Round ended but match continues
                log_state(f"Round ended, waiting for next round...")
                state = "waiting_for_round"
                alive_since = None
                death_since = None
                # Reset post-match tracking in case we were in that state
                post_match_entry_logged = False
                post_match_action_count = 0
                # DON'T RESET round_state here! Keep the confirmed counts for high watermark

        # 5) State-specific logic
        if state == "waiting_for_match":
            # Debug log every 30 frames
            if global_step % 30 == 0:
                log_debug(f"waiting_for_match: P1={pct1:.1f}% P2={pct2:.1f}% "
                         f"alive_since={alive_since is not None} "
                         f"recent_round_end={round_state.has_round_recently_ended(timeout=1.0)}")
            
            # Waiting for first round of a new match
            if pct1 >= HEALTH_LIMIT and pct2 >= HEALTH_LIMIT:
                # Don't start if we just confirmed a round end
                if not round_state.has_round_recently_ended(timeout=1.0):
                    alive_since = alive_since or time.time()
                    if time.time() - alive_since >= 0.5:
                        log_round("🚀 FIRST ROUND OF MATCH STARTED!")
                        write_action("start\n")
                        state = "active"
                        frame_stack.clear()
                        # Reset tracking variables
                        prev_state = None
                        prev_action = None
                        prev_pct1 = pct1
                        prev_pct2 = pct2
                        round_start_time = time.time()
                        round_steps = 0
                        round_reward = 0.0
                        action_counts.fill(0)
                        # Reset post-match tracking
                        post_match_entry_logged = False
                        post_match_action_count = 0
                else:
                    alive_since = None
            else:
                alive_since = None

        elif state == "waiting_for_round":
            # We KNOW previous round ended, waiting for next round
            if pct1 >= HEALTH_LIMIT and pct2 >= HEALTH_LIMIT:
                alive_since = alive_since or time.time()
                if time.time() - alive_since >= 0.3:  # Can be faster here
                    log_round("🚀 NEXT ROUND STARTED!")
                    write_action("start\n")
                    state = "active"
                    frame_stack.clear()
                    # Reset tracking variables
                    prev_state = None
                    prev_action = None
                    prev_pct1 = pct1
                    prev_pct2 = pct2
                    round_start_time = time.time()
                    round_steps = 0
                    round_reward = 0.0
                    action_counts.fill(0)
                    # Reset post-match tracking
                    post_match_entry_logged = False
                    post_match_action_count = 0
            else:
                alive_since = None

        elif state == "post_match_waiting":
            # Debug: Track if we're actually entering this state
            if not post_match_entry_logged:
                post_match_entry_logged = True
                log_state("ENTERED POST_MATCH_WAITING STATE")
                post_match_action_count = 0
            
            # Check for match reset (indicators clear + full health)
            indicators_clear = (detected_p1_rounds == 0 and detected_p2_rounds == 0)
            
            # Check if all indicators are blue (new match ready)
            all_blue = all(state == 'blue' for state in indicator_states.values())
            
            # Debug log the current state
            if post_match_action_count % 100 == 0:
                log_debug(f"Post-match check: indicators_clear={indicators_clear}, all_blue={all_blue}, "
                         f"P1:{detected_p1_rounds} P2:{detected_p2_rounds}, "
                         f"health P1:{pct1:.1f}% P2:{pct2:.1f}%")
            
            if pct1 >= HEALTH_LIMIT and pct2 >= HEALTH_LIMIT and (indicators_clear or all_blue):
                alive_since = alive_since or time.time()
                if time.time() - alive_since >= 0.5:
                    log_state(f"🆕 NEW MATCH DETECTED! Starting Match #{match_tracker.match_number} "
                             f"(indicators: {'all blue' if all_blue else 'cleared'})")
                    round_state.reset()  # NOW we reset for the new match
                    
                    # Reset the debug flags
                    post_match_entry_logged = False
                    post_match_action_count = 0
                    
                    state = "waiting_for_match"
                    alive_since = None
            else:
                alive_since = None
                
            # Continue alternating actions
            time_rounded = round(time.time(), 2)
            time_int = int(time_rounded * 100)
            
            if time_int % 2 == 0:
                action_to_write = "start\n"
            else:
                action_to_write = "kick\n"
            
            # Debug: Log every 50th action
            post_match_action_count += 1
            if post_match_action_count % 50 == 0:
                log_debug(f"Post-match action #{post_match_action_count}: {action_to_write.strip()}")
            
            write_action(action_to_write)

        elif state == "active":
            # ACTIVE → DQN actions
            round_steps += 1
            
            # 1) Always append the new pre-processed frame
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            img = cv2.resize(gray, CNN_SIZE, interpolation=cv2.INTER_NEAREST)
            frame_stack.append(img.astype(np.float32) / 255.0)

            # 2) Only make a new decision when we have enough frames and hold counter expired
            if len(frame_stack) == FRAME_STACK and hold_counter <= 0:
                # A) Snapshot the next state (sₜ₊₁)
                next_state = np.stack(frame_stack, 0)  # shape (4,84,84)

                # B) Compute extras for this new frame (extrasₜ₊₁)
                ts = classify_transform_state(frame)
                bp, _ = compute_black_stats(frame)
                code = {'can transform':0, 'transformed':1, 'cannot transform':2, 'unknown':3}
                next_extras = np.array([
                    code[ts['P1_R1_pixel']],
                    code[ts['P2_R2_pixel']],
                    bp['P1_R1_area'] / 100.0,
                    bp['P2_R2_area'] / 100.0
                ], dtype=np.float32)

                # C) Compute reward rₜ₊₁ from the last action
                reward = 0.0
                if prev_state is not None and prev_action is not None:
                    our_damage = prev_pct1 - pct1
                    opp_damage = prev_pct2 - pct2
                    reward = opp_damage - our_damage
                    round_reward += reward
                    
                    # Log step reward
                    writer.add_scalar("reward/step", reward, global_step)
                    writer.add_scalar("health/p1", pct1, global_step)
                    writer.add_scalar("health/p2", pct2, global_step)

                # D) Store the full transition in replay buffer
                #    (prev_state, prev_extras, prev_action, reward, next_state, next_extras, done=False)
                if prev_state is not None and prev_action is not None:
                    buffer.add(
                        prev_state,
                        prev_extra_feats,
                        prev_action,
                        reward,
                        next_state,
                        next_extras,
                        False
                    )

                # E) ε-greedy action selection on the new state
                state_img    = torch.from_numpy(next_state).unsqueeze(0).to(DEVICE)
                extras_tensor= torch.from_numpy(next_extras).unsqueeze(0).to(DEVICE)
                eps = 0.01 if TEST_MODE else max(0.01, 0.1 - 0.1 * (buffer.len / 10000))
                
                # Log epsilon
                writer.add_scalar("exploration/epsilon", eps, global_step)
                
                if random.random() < eps:
                    chosen = random.randrange(NUM_ACTIONS)
                else:
                    with torch.no_grad():
                        q = policy_net(state_img, extras_tensor)
                        
                        # Log Q-values for each action
                        for i, action_name in enumerate(ACTIONS):
                            writer.add_scalar(f"q_values/{action_name}", q[0, i].item(), global_step)
                        
                        # Log Q-value statistics
                        writer.add_scalar("q_values/mean", q.mean().item(), global_step)
                        writer.add_scalar("q_values/max", q.max().item(), global_step)
                        writer.add_scalar("q_values/min", q.min().item(), global_step)
                        writer.add_scalar("q_values/std", q.std().item(), global_step)
                        
                        chosen = int(q.argmax(1).item())
                
                current_action = chosen
                action_counts[chosen] += 1
                hold_counter   = FRAME_STACK

                # F) Roll forward for next decision
                prev_state       = next_state
                prev_extra_feats = next_extras
                prev_action      = current_action
                prev_pct1        = pct1
                prev_pct2        = pct2

                # G) Slide the window by one frame
                frame_stack.popleft()
                write_count += 1

            # 3) Write (hold) the selected action every frame
            if current_action is not None:
                write_action(ACTIONS[current_action] + "\n")
                hold_counter -= 1

            # 4) Display Q-values for debugging
            if 'state_img' in locals():
                with torch.no_grad():
                    q_vals = policy_net(state_img, extras_tensor).squeeze(0).cpu().numpy()
                disp = np.zeros((20 * NUM_ACTIONS, 200, 3), dtype=np.uint8)
                for i, (act, val) in enumerate(zip(ACTIONS, q_vals)):
                    y = 15 + i * 20
                    cv2.putText(disp, f"{act[:6]:6s}: {val:6.2f}", (5, y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180,180,180), 1)
                cv2.imshow("Q-Values", disp)
                cv2.waitKey(1)

        # Save results (always track health regardless of state)
        if frame is not None:
            results.append((time.time(), pct1, pct2))

        # Save screenshots (only if we have a frame)
        if frame is not None and len(screenshots) < MAX_FRAMES:
            screenshots.append(frame.copy())

        # Mark task as done only if we got a frame
        try:
            frame_q.task_done()
        except:
            pass
# ─── ENHANCED LEARNER WITH CONTINUOUS TRAINING ─────────────────────────────
def learner():
    global match_number
    train_steps = 0

    if TEST_MODE:
        log_learner("[Learner] TEST MODE - Training disabled")
        while not stop_event.is_set():
            # Still check for match end to save current model state
            if match_end_event.wait(timeout=1.0):
                match_end_event.clear()
                log_learner(f"[Learner] Match ended in test mode, match {match_number}")
                match_number += 1
        return

    while not stop_event.is_set():
        # Train continuously when buffer is ready
        if buffer.len >= MIN_BUFFER_SIZE:
            # Train for a batch
            states, extras, actions, rewards, next_states, next_extras, dones = buffer.sample(BATCH_SIZE)

            # Analyze buffer periodically
            if train_steps % 1000 == 0 and buffer.len > 0:
                # Log buffer statistics
                rewards_in_buffer = buffer.rewards[:buffer.len]
                writer.add_scalar("buffer/reward_mean", rewards_in_buffer.mean(), train_steps)
                writer.add_scalar("buffer/reward_std", rewards_in_buffer.std(), train_steps)
                writer.add_scalar("buffer/reward_min", rewards_in_buffer.min(), train_steps)
                writer.add_scalar("buffer/reward_max", rewards_in_buffer.max(), train_steps)
                
                # Log action distribution in buffer
                actions_in_buffer = buffer.actions[:buffer.len]
                for i, action_name in enumerate(ACTIONS):
                    action_pct = (actions_in_buffer == i).sum() / len(actions_in_buffer)
                    writer.add_scalar(f"buffer/action_distribution/{action_name}", 
                                    action_pct, train_steps)
                
                # Log done percentage
                dones_in_buffer = buffer.dones[:buffer.len]
                writer.add_scalar("buffer/done_percentage", 
                                dones_in_buffer.sum() / len(dones_in_buffer), train_steps)

            with torch.no_grad():
                next_q = target_net(next_states, next_extras).max(1)[0]
                target = rewards + GAMMA * next_q * (1 - dones.float())
                
                # Log TD error statistics
                current_q = policy_net(states, extras).gather(1, actions.unsqueeze(1)).squeeze(1)
                td_error = (target - current_q).abs()
                writer.add_scalar("training/td_error_mean", td_error.mean().item(), train_steps)
                writer.add_scalar("training/td_error_max", td_error.max().item(), train_steps)

            q_vals = policy_net(states, extras).gather(1, actions.unsqueeze(1)).squeeze(1)
            loss   = F.mse_loss(q_vals, target)

            optimizer.zero_grad()
            loss.backward()
            
            # Log gradient statistics before clipping
            total_grad_norm = 0
            for p in policy_net.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2).item()
                    total_grad_norm += param_norm ** 2
            total_grad_norm = total_grad_norm ** 0.5
            writer.add_scalar("gradients/norm_before_clip", total_grad_norm, train_steps)
            
            # Clip gradients
            clipped_norm = torch.nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=1.0)
            writer.add_scalar("gradients/norm_after_clip", clipped_norm.item(), train_steps)
            
            optimizer.step()
            
            # Log comprehensive training metrics
            writer.add_scalar("loss/train", loss.item(), train_steps)
            writer.add_scalar("training/learning_rate", LEARNING_RATE, train_steps)
            writer.add_scalar("buffer/size", buffer.len, train_steps)
            writer.add_scalar("buffer/utilization", buffer.len / buffer.size, train_steps)

            # Log layer-wise statistics periodically
            if train_steps % 500 == 0:
                for name, param in policy_net.named_parameters():
                    if param.requires_grad and param.grad is not None:
                        writer.add_histogram(f"weights/{name}", param.data, train_steps)
                        writer.add_histogram(f"gradients/{name}", param.grad.data, train_steps)
                        writer.add_scalar(f"weights/{name}/mean", param.data.mean().item(), train_steps)
                        writer.add_scalar(f"weights/{name}/std", param.data.std().item(), train_steps)

            train_steps += 1
            # Log the training loss and buffer occupancy periodically
            if train_steps % 100 == 0:
                log_learner(f"[Learner] step={train_steps:6d} "
                    f"loss={loss.item():.4f} "
                    f"buffer_len={buffer.len} "
                    f"grad_norm={total_grad_norm:.4f}")

            # Sync target network every N steps (not per round)
            if train_steps % TARGET_SYNC == 0:
                target_net.load_state_dict(policy_net.state_dict())
                log_learner(f"[Learner] Synced target network at step {train_steps}")
                writer.add_scalar("training/target_sync", 1, train_steps)

            # Log training progress
            if train_steps % 100 == 0:
                log_learner(f"[Learner] Step {train_steps}, Loss: {loss.item():.4f}, Buffer: {buffer.len}")

        # Check for match end to save model
        if match_end_event.wait(timeout=0.01):
            match_end_event.clear()
            model_path = f"{MODEL_DIR}/model_match_{match_number}.pth"
            torch.save(policy_net.state_dict(), model_path)
            log_learner(f"[Learner] Saved model to {model_path}")
            
            # Log match completion
            writer.add_scalar("training/match_completed", match_number, train_steps)
            
            match_number += 1
            # Don't clear buffer - preserve experience!
            log_learner(f"[Learner] Buffer preserved with {buffer.len} samples")

        # Small sleep to prevent CPU spinning
        time.sleep(0.001)

# ─── MAIN ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    log_state("="*60)
    if ENABLE_LOGGING:
        log_state(f"Starting RL agent - Log file: {log_filename}")
    else:
        log_state("Starting RL agent - File logging DISABLED")
    log_state(f"Device: {DEVICE}")
    log_state(f"Region: {REGION}")
    log_state(f"Health bar locations - P1: {X1_P1}-{X2_P1}, P2: {X1_P2}-{X2_P2}")
    log_state(f"Learning rate: {LEARNING_RATE}")
    log_state(f"Min buffer size: {MIN_BUFFER_SIZE}, Target sync: {TARGET_SYNC} steps")
    log_state(f"Round detection: State-independent with confirmation")
    log_state(f"Round indicators: {len(ROUND_INDICATORS)} positions")
    if LOAD_CHECKPOINT:
        log_state(f"Checkpoint: {LOAD_CHECKPOINT}")
    if TEST_MODE:
        log_state("MODE: TEST (training disabled)")
    else:
        log_state("MODE: TRAINING")
    log_state("="*60)

    # start threads
    t_p = threading.Thread(target=producer, name="Producer", daemon=True)
    t_c = threading.Thread(target=consumer, name="Consumer", daemon=True)
    t_l = threading.Thread(target=learner,  name="Learner",  daemon=True)
    t_p.start(); t_c.start(); t_l.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        log_state("\nShutting down...")
        stop_event.set()

    writer.close()
    
    # drain & join
    t_p.join()
    frame_q.join()
    t_c.join()
    t_l.join()

    # save outputs
    # os.makedirs("screenshots", exist_ok=True)
    # for i, f in enumerate(screenshots):
    #     Image.fromarray(f[..., ::-1]).save(f"screenshots/frame_{i:04d}.png")

    with open(LOG_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["time_s","p1_pct","p2_pct"])
        w.writerows(results)

    camera.stop()
    cv2.destroyAllWindows()  # Close any debug windows
    log_state(f"\nSaved {len(screenshots)} screenshots and {len(results)} health readings")
    if ENABLE_LOGGING:
        log_state(f"Log file saved to: {log_filename}")