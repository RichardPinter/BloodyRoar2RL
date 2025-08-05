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
HEALTH_LIMIT    = 99.0

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
        # Confirmed state (persistent, can only increase)
        self.confirmed_p1_rounds = 0
        self.confirmed_p2_rounds = 0

        # Candidate state (needs confirmation)
        self.candidate_state = None
        self.candidate_start_time = None
        self.CONFIRMATION_TIME = 0.5  # Keep at 0.5s for stability

        # Track if we've recently confirmed a round
        self.last_round_end_time = None
        
        log_state("🔄 RoundState initialized: P1:0 P2:0")

    def update(self, detected_p1, detected_p2):
        """
        Update round state with new detection.
        Returns: None or ("round_won", winner, p1_rounds, p2_rounds)
        """
        current_time = time.time()
        new_state = (detected_p1, detected_p2)

        # Log all detections to file for debugging (only if changed)
        if (detected_p1, detected_p2) != (self.confirmed_p1_rounds, self.confirmed_p2_rounds):
            log_debug(f"Round detection: P1={detected_p1}, P2={detected_p2}, confirmed=(P1:{self.confirmed_p1_rounds}, P2:{self.confirmed_p2_rounds})")

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
                        # Guard against both players "winning" simultaneously
                        if detected_p1 > self.confirmed_p1_rounds and detected_p2 > self.confirmed_p2_rounds:
                            log_debug(f"⚠️ Ambiguous round: both P1 and P2 advanced ({detected_p1}-{detected_p2}); ignoring")
                            self.candidate_state = None
                            self.candidate_start_time = None
                            return None

                        # CONFIRMED! Update persistent state
                        old_p1, old_p2 = self.confirmed_p1_rounds, self.confirmed_p2_rounds
                        self.confirmed_p1_rounds = detected_p1
                        self.confirmed_p2_rounds = detected_p2
                        
                        # Track when this round ended
                        self.last_round_end_time = current_time

                        # Determine who won the round (check P2 first)
                        if detected_p2 > old_p2:
                            winner = "p2"
                            log_round(f"🎯 ROUND CONFIRMED: P2 won! (P1:{detected_p1} P2:{detected_p2})")
                        elif detected_p1 > old_p1:
                            winner = "p1"
                            log_round(f"🎯 ROUND CONFIRMED: P1 won! (P1:{detected_p1} P2:{detected_p2})")
                        else:
                            # This shouldn't happen due to earlier checks, but just in case
                            log_debug(f"⚠️ Round confirmed but no winner detected?")
                            winner = "unknown"

                        # Clear candidate
                        self.candidate_state = None
                        self.candidate_start_time = None

                        value_tuple = ("round_won", winner, detected_p1, detected_p2)
                        log_round(f'Round confirmation successful: {value_tuple}')
                        return value_tuple
                    else:
                        # Still waiting for confirmation
                        if elapsed < 0.1 or int(elapsed * 10) != int((elapsed - 0.016) * 10):
                            log_debug(f"⏳ [Candidate] P1:{detected_p1} P2:{detected_p2} ({elapsed:.1f}s) - waiting for confirmation...")
                else:
                    # New candidate state
                    self.candidate_state = new_state
                    self.candidate_start_time = current_time
                    log_debug(f"🔍 [New Candidate] P1:{detected_p1} P2:{detected_p2} - starting confirmation timer")

        else:
            # Not an upgrade - ignore (noise/temporary false positive)
            if (detected_p1 < self.confirmed_p1_rounds or
                detected_p2 < self.confirmed_p2_rounds):
                log_debug(f"🚫 [Ignored] P1:{detected_p1} P2:{detected_p2} - not an upgrade from P1:{self.confirmed_p1_rounds} P2:{self.confirmed_p2_rounds}")

        return None

    def has_round_recently_ended(self, timeout=2.0):
        """Check if a round ended within the timeout period"""
        if self.last_round_end_time is None:
            return False
        return (time.time() - self.last_round_end_time) < timeout

    def get_current_state(self):
        """Get the current confirmed round state"""
        return self.confirmed_p1_rounds, self.confirmed_p2_rounds

    def reset(self):
        """Reset round state for new match"""
        self.confirmed_p1_rounds = 0
        self.confirmed_p2_rounds = 0
        self.candidate_state = None
        self.candidate_start_time = None
        self.last_round_end_time = None
        log_state("🔄 RoundState reset: P1:0 P2:0")

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

def detect_round_indicators(frame):
    """Simple round detection using same pattern as health bars"""
    results = {}
    debug_info = []

    for name, (x1, y1, x2, y2) in ROUND_INDICATORS.items():
        # Extract region (same as health bar pattern)
        region = frame[y1:y2, x1:x2]

        # Create mask for red pixels (same pattern as health bar)
        mask = cv2.inRange(region, RED_BGR_LOWER, RED_BGR_UPPER)

        # Count red pixels and calculate percentage
        total_pixels = region.shape[0] * region.shape[1]
        if total_pixels > 0:
            red_pixels = cv2.countNonZero(mask)
            red_pct = red_pixels / total_pixels * 100.0
        else:
            red_pct = 0.0

        # Log P2 detection values to file when they might be active
        if 'p2' in name and red_pct > 10:
            debug_info.append(f"{name}:{red_pct:.1f}%")

        # Simple threshold: >30% red = won round
        results[name] = red_pct > 30.0
    
    # Log all P2 debug info at once to reduce file writes
    if debug_info:
        log_debug(f"P2 indicators: {', '.join(debug_info)}")

    return results

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

    # MODIFIED STATE MACHINE:
    # - "waiting_for_match": Waiting for first round of a new match
    # - "waiting_for_round": Previous round ended, waiting for next round
    # - "active": Round is in progress
    # - "post_match_waiting": Match ended, navigating menus
    state = "waiting_for_match"
    
    alive_since  = None
    death_since  = None

    # Proper state tracking for rewards
    prev_state   = None
    prev_action  = None
    prev_extra_feats = None
    prev_pct1    = 100.0
    prev_pct2    = 100.0
    

    # Initialize robust round and match tracking
    round_state = RoundState()
    match_tracker = MatchTracker(start_match_number=match_number)
    log_debug(f"MatchTracker initialized with match_number: {match_number}")
    
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
    
    while not stop_event.is_set():
        # Get frame with timeout to avoid blocking during post-match
        frame = None
        ts = None
        try:
            frame, ts = frame_q.get(timeout=0.1)
        except:
            # No frame available - handle post_match actions
            if state == "post_match_waiting":
                time_rounded = round(time.time(), 2)
                time_int = int(time_rounded * 100)
                if time_int % 2 == 0:
                    write_action("start\n")
                else:
                    write_action("kick\n")
            continue
        
        # If we got here, we have a frame - process it normally
        
        # 1) Compute health % once
        strip = frame[Y_HEALTH:Y_HEALTH+1]
        m1    = cv2.inRange(strip[:, X1_P1:X2_P1], LOWER_BGR, UPPER_BGR)
        m2    = cv2.inRange(strip[:, X1_P2:X2_P2], LOWER_BGR, UPPER_BGR)
        pct1  = cv2.countNonZero(m1) / LEN_P1 * 100.0
        pct2  = cv2.countNonZero(m2) / LEN_P2 * 100.0

        # 2) ALWAYS detect rounds - STATE INDEPENDENT!
        round_indicators = detect_round_indicators(frame)
        detected_p1_rounds = sum([round_indicators['p1_round1'], round_indicators['p1_round2']])
        detected_p2_rounds = sum([round_indicators['p2_round1'], round_indicators['p2_round2']])
        
        # Log periodically
        if global_step % 30 == 0:
            log_debug(f"Frame health: P1={pct1:.2f}%, P2={pct2:.2f}%")
            log_debug(f"Round indicators: {round_indicators}")
            log_debug(f"Detected rounds: P1={detected_p1_rounds}, P2={detected_p2_rounds}")
            log_debug(f"Current state: {state}")

        # 3) Update round state - ALWAYS, regardless of game state
        round_result = round_state.update(detected_p1_rounds, detected_p2_rounds)

        # 4) Handle round completion if detected
        if round_result and round_result[0] == "round_won":
            _, winner, p1_rounds, p2_rounds = round_result
            
            log_round(f"[Episode] Round #{episode_count} total_reward={round_reward:.2f}")
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
                # Match ended - save model and enter post-match navigation
                on_match_end()
                log_state(f"🎯 Match over! Entering post-match navigation mode...")
                state = "post_match_waiting"
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

        # 5) State-specific logic
        if state == "waiting_for_match":
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
            
            if pct1 >= HEALTH_LIMIT and pct2 >= HEALTH_LIMIT and indicators_clear:
                alive_since = alive_since or time.time()
                if time.time() - alive_since >= 0.5:
                    log_state(f"🆕 NEW MATCH DETECTED! Starting Match #{match_tracker.match_number}")
                    round_state.reset()
                    
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
            global_step += 1
            
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