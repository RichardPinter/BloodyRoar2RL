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

import comtypes
import dxcam
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

# ─── CONFIG ────────────────────────────────────────────────────────────────
REGION        = (0, 0, 624, 548)      # x, y, width, height
Y_HEALTH      = 116
X1_P1, X2_P1  = 73, 292
X1_P2, X2_P2  = 355, 574
LEN_P1        = X2_P1 - X1_P1
LEN_P2        = X2_P2 - X1_P2

LOWER_BGR     = np.array([0,150,180], dtype=np.uint8)
UPPER_BGR     = np.array([30,175,220], dtype=np.uint8)

FRAME_STACK     = 4
CNN_SIZE        = (84, 84)
ACTIONS         = ["punch",'special', "kick", "transform", "jump", "squat", "left", "right"]
NUM_ACTIONS     = len(ACTIONS)

DEVICE          = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LOG_CSV         = "health_results.csv"
ACTIONS_FILE    = "../actions.txt"
MAX_FRAMES      = 1000
GAMMA           = 0.99
BATCH_SIZE      = 32
TARGET_SYNC     = 1000
REPLAY_SIZE     = 10000
MIN_BUFFER_SIZE = 1000
HEALTH_LIMIT    = 99.0
LEARNING_RATE   = 1e-4

LOAD_CHECKPOINT = "C:/Users/richa/bro2_rl/screenshot_test/model_match_34.pth"
TEST_MODE       = False

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
        self.dones       = np.zeros(self.size, dtype=bool)
        self.ptr = 0
        self.len = 0

    def add(self, s, extra, a, r, s2, done):
        self.states[self.ptr]      = s
        self.extras[self.ptr]      = extra
        self.actions[self.ptr]     = a
        self.rewards[self.ptr]     = r
        self.next_states[self.ptr] = s2
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
        self.CONFIRMATION_TIME = 1.0  # Must see state for 1 second

        print("🔄 RoundState initialized: P1:0 P2:0")

    def update(self, detected_p1, detected_p2):
        """
        Update round state with new detection.
        Returns: None or ("round_won", winner, p1_rounds, p2_rounds)
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

def detect_round_indicators(frame):
    """Simple round detection using same pattern as health bars"""
    results = {}

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

        # Simple threshold: >50% red = won round
        results[name] = red_pct > 50.0

    return results

# ─── SETUP ───────────────────────────────────────────────────────────────────
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
    print(f"✅ Loaded checkpoint from {LOAD_CHECKPOINT}")

    # Extract match number from filename if possible
    match = re.search(r'model_match_(\d+)', LOAD_CHECKPOINT)
    if match:
        start_match_number = int(match.group(1)) + 1
        print(f"   Continuing from match {start_match_number}")
else:
    print(os.path.exists(LOAD_CHECKPOINT))
    print(LOAD_CHECKPOINT)
    target_net.load_state_dict(policy_net.state_dict())
    if LOAD_CHECKPOINT:
        print(f"⚠️  Checkpoint {LOAD_CHECKPOINT} not found, training from scratch")

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

# ─── PRODUCER ────────────────────────────────────────────────────────────────
def producer():
    start = time.perf_counter()
    while not stop_event.is_set():
        frm = camera.get_latest_frame()
        if frm is not None:
            ts = time.perf_counter() - start
            frame_q.put((frm.copy(), ts))

# ─── SINGLE CONSUMER w/ HEALTH & ROUND LOGIC ────────────────────────────────
def consumer():
    global match_number

    state        = "waiting"
    alive_since  = None
    death_since  = None

    # Proper state tracking for rewards
    prev_state   = None
    prev_action  = None
    prev_pct1    = 100.0
    prev_pct2    = 100.0

    # Initialize robust round and match tracking
    round_state = RoundState()
    match_tracker = MatchTracker()
    
    # Step 1: create a named window for Q‑values
    cv2.namedWindow("Q-Values", cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow("Rewards",  cv2.WINDOW_AUTOSIZE)

    # compute where to put it:
    x, y, w, h = REGION     # REGION = (0,0,624,548)
    win_x = x
    win_y = y + h + 100     # leave a 100px buffer below the capture area

    cv2.moveWindow("Q-Values", win_x, win_y)
    cv2.moveWindow("Rewards", win_x+300, win_y)

    def write_action(text: str):
        with open(ACTIONS_FILE, "w") as f:
            f.write(text)

    # ── TRAINING HOOKS ────
    def on_round_end():
        round_end_event.set()  # Just signal the learner thread

    def on_match_end():
        match_end_event.set()  # Signal the learner thread for match end
    # ──────────────  Reward Tracking ─────────────────
    reward_history = deque(maxlen=100)

    round_reward = 0.0
    # ────────────────────────────────────────────────
    
    
    # In consumer(), before while loop
    debug_file = open("debug_transitions.txt", "a")
    write_count = 0
    
    while not stop_event.is_set():
        frame, ts = frame_q.get()

        # 1) Compute health % once
        strip = frame[Y_HEALTH:Y_HEALTH+1]
        m1    = cv2.inRange(strip[:, X1_P1:X2_P1], LOWER_BGR, UPPER_BGR)
        m2    = cv2.inRange(strip[:, X1_P2:X2_P2], LOWER_BGR, UPPER_BGR)
        pct1  = cv2.countNonZero(m1) / LEN_P1 * 100.0
        pct2  = cv2.countNonZero(m2) / LEN_P2 * 100.0

        # 2) Robust round detection with confirmation (only during active rounds)
        round_result = None
        if state in ["waiting", "active"]:  # Only check rounds when not in post-match
            round_indicators = detect_round_indicators(frame)

            # Count current rounds for each player
            detected_p1_rounds = sum([round_indicators['p1_round1'], round_indicators['p1_round2']])
            detected_p2_rounds = sum([round_indicators['p2_round1'], round_indicators['p2_round2']])

            # Update round state with new detection
            round_result = round_state.update(detected_p1_rounds, detected_p2_rounds)
        elif state == "post_match_waiting":
            # During post-match, skip round detection to avoid noise
            detected_p1_rounds = detected_p2_rounds = 0

        # 3) WAITING → detect round start
        if state == "waiting":
            if pct1 >= HEALTH_LIMIT and pct2 >= HEALTH_LIMIT:
                alive_since = alive_since or time.time()
                if time.time() - alive_since >= 0.5:
                    print("🚀 ROUND STARTED: Both players at 99%+ health!")
                    write_action("start\n")
                    state = "active"
                    frame_stack.clear()
                    # Reset tracking variables
                    prev_state = None
                    prev_action = None
                    prev_pct1 = pct1
                    prev_pct2 = pct2
            else:
                alive_since = None

        # 4) POST-MATCH WAITING → continuously alternate start/kick until new round
        elif state == "post_match_waiting":
            # Continuously alternate between start and kick based on current time
            time_rounded = round(time.time(), 2)
            time_int = int(time_rounded * 100)

            if time_int % 2 == 0:
                write_action("start\n")
            else:
                write_action("kick\n")

            # Wait for both health bars to reach 99%+ (indicating new round started)
            if pct1 >= HEALTH_LIMIT and pct2 >= HEALTH_LIMIT:
                alive_since = alive_since or time.time()
                if time.time() - alive_since >= 0.5:
                    print(f"🆕 NEW ROUND DETECTED: Both players at 99%+ health! Starting Match #{match_tracker.match_number}")

                    # Reset round state for the new match
                    round_state.reset()

                    # Transition back to normal flow
                    state = "waiting"
                    alive_since = None
            else:
                alive_since = None

            # Skip frame processing and learning during post-match
            # (Nothing meaningful to learn from menu screens)

        # 5) ACTIVE → DQN actions + death detection
        elif state == "active":
            
            # 0) Compute transform & black‐pixel diagnostics
            ts = classify_transform_state(frame)          # {'P1_R1_pixel':..., 'P2_R2_pixel':...}
            bp, _ = compute_black_stats(frame)            # ({'P1_R1_area':pct,...}, {...})
            code = {'can transform':0, 'transformed':1, 'cannot transform':2, 'unknown':3}
            f_transform = [
                code[ ts['P1_R1_pixel'] ],
                code[ ts['P2_R2_pixel'] ]
            ]
            f_black_pct = [
                bp['P1_R1_area'] / 100.0,
                bp['P2_R2_area'] / 100.0
            ]
            extra_feats = np.array(f_transform + f_black_pct, dtype=np.float32)
            # print diagnostics
            # print(f"Transform states: {ts}")
            # print(f"Black %        : {{'P1':{bp['P1_R1_area']:.1f}, 'P2':{bp['P2_R2_area']:.1f}}}")

            # 1) Preprocess frame
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            img  = cv2.resize(gray, CNN_SIZE, interpolation=cv2.INTER_NEAREST)
            frame_stack.append(img.astype(np.float32) / 255.0)

            if len(frame_stack) == FRAME_STACK:
                current_state = np.stack(frame_stack, 0)  # (4,84,84)

                # 2) Compute reward from prev action
                if prev_state is not None and prev_action is not None:
                    our_damage = prev_pct1 - pct1
                    opp_damage = prev_pct2 - pct2
                    reward     = opp_damage - our_damage
                    round_reward += reward
                    # store transition with extras
                    
                    buffer.add(
                        prev_state,
                        prev_extra_feats,
                        prev_action,
                        reward,
                        current_state,
                        False
                    )
                    print(prev_extra_feats)
                    write_count += 1

                # Update reward
                 
                
                # 3) Convert to tensors
                state_img    = torch.from_numpy(current_state).unsqueeze(0).to(DEVICE)   # (1,4,84,84)
                extra_tensor = torch.from_numpy(extra_feats).unsqueeze(0).to(DEVICE)      # (1,4)

                # 4) ε-greedy action selection
                eps = 0.01 if TEST_MODE else max(0.01, 0.1 - 0.1 * (buffer.len / 10000))
                if random.random() < eps:
                    action = random.randrange(NUM_ACTIONS)
                else:
                    with torch.no_grad():
                        q = policy_net(state_img, extra_tensor)
                        q_vals = q.squeeze(0).cpu().numpy()                  # (NUM_ACTIONS,)
                        height = 20 * len(ACTIONS)
                        width  = 200
                        disp   = np.zeros((height, width, 3), dtype=np.uint8)

                        for i, (act, val) in enumerate(zip(ACTIONS, q_vals)):
                            y = 15 + i * 20
                            text = f"{act[:6]:6s}: {val:6.2f}"
                            cv2.putText(disp, text, (5, y),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180,180,180), 1, cv2.LINE_AA)

                        cv2.imshow("Q-Values", disp)
                        cv2.waitKey(1)
                    action = int(q.argmax(1).item())
                
                write_action(ACTIONS[action] + "\n")
                debug_file.write(f" {ACTIONS[action]}.")
                if write_count % 100 == 0:
                    debug_file.flush()
                # 5) Roll forward for next step
                prev_state        = current_state
                prev_extra_feats  = extra_feats
                prev_action       = action
                prev_pct1         = pct1
                prev_pct2         = pct2
                frame_stack.popleft()

        # Check for confirmed round wins
        if state == "active" and round_result and round_result[0] == "round_won":
            
            print(round_reward)
            # 1) push this round’s total reward
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
            
            _, winner, p1_rounds, p2_rounds = round_result

            # Final transition with done=True and terminal reward
            if prev_state is not None and prev_action is not None:
                final_reward = 10.0 if winner == "p1" else -10.0
                buffer.add(prev_state, prev_extra_feats, prev_action, final_reward, prev_state, True)

            # round-ended hook
            on_round_end()

            # Check for match end
            match_result = match_tracker.check_match_end(p1_rounds, p2_rounds)

            if match_result and match_result[0] == "match_over":
                # Match ended - save model and enter post-match navigation
                on_match_end()
                print(f"🎯 Match over! Entering post-match navigation mode...")

                state = "post_match_waiting"
                alive_since = death_since = None
            else:
                # Round ended but match continues - back to waiting
                state = "waiting"
                alive_since = death_since = None

        # Save results (always track health regardless of state)
        results.append((time.time(), pct1, pct2))

        # Save screenshots
        if len(screenshots) < MAX_FRAMES:
            screenshots.append(frame.copy())

        frame_q.task_done()

# ─── ENHANCED LEARNER WITH CONTINUOUS TRAINING ─────────────────────────────
def learner():
    global match_number
    train_steps = 0

    if TEST_MODE:
        print("[Learner] TEST MODE - Training disabled")
        while not stop_event.is_set():
            # Still check for match end to save current model state
            if match_end_event.wait(timeout=1.0):
                match_end_event.clear()
                print(f"[Learner] Match ended in test mode, match {match_number}")
                match_number += 1
        return

    while not stop_event.is_set():
        # Train continuously when buffer is ready
        if buffer.len >= MIN_BUFFER_SIZE:
            # Train for a batch
            states, extras, actions, rewards, next_states, dones = buffer.sample(BATCH_SIZE)

            with torch.no_grad():
                next_q = target_net(next_states, extras).max(1)[0]
                target = rewards + GAMMA * next_q * (1 - dones.float())

            q_vals = policy_net(states, extras).gather(1, actions.unsqueeze(1)).squeeze(1)
            loss   = F.mse_loss(q_vals, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_steps += 1

            # Sync target network every N steps (not per round)
            if train_steps % TARGET_SYNC == 0:
                target_net.load_state_dict(policy_net.state_dict())
                print(f"[Learner] Synced target network at step {train_steps}")

            # Log training progress
            if train_steps % 100 == 0:
                print(f"[Learner] Step {train_steps}, Loss: {loss.item():.4f}, Buffer: {buffer.len}")

        # Check for match end to save model
        if match_end_event.wait(timeout=0.01):
            match_end_event.clear()
            model_path = f"model_match_{match_number}.pth"
            torch.save(policy_net.state_dict(), model_path)
            print(f"[Learner] Saved model to {model_path}")
            match_number += 1
            # Don't clear buffer - preserve experience!
            print(f"[Learner] Buffer preserved with {buffer.len} samples")

        # Small sleep to prevent CPU spinning
        time.sleep(0.001)

# ─── MAIN ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"Starting RL agent with device: {DEVICE}")
    print(f"Region: {REGION}")
    print(f"Health bar locations - P1: {X1_P1}-{X2_P1}, P2: {X1_P2}-{X2_P2}")
    print(f"Learning rate: {LEARNING_RATE}")
    print(f"Min buffer size: {MIN_BUFFER_SIZE}, Target sync: {TARGET_SYNC} steps")
    print(f"Round detection: Simplified rectangle monitoring")
    print(f"Round indicators: {len(ROUND_INDICATORS)} positions")
    if LOAD_CHECKPOINT:
        print(f"Checkpoint: {LOAD_CHECKPOINT}")
    if TEST_MODE:
        print("MODE: TEST (training disabled)")
    else:
        print("MODE: TRAINING")
    print("-" * 50)

    # start threads
    t_p = threading.Thread(target=producer, name="Producer", daemon=True)
    t_c = threading.Thread(target=consumer, name="Consumer", daemon=True)
    t_l = threading.Thread(target=learner,  name="Learner",  daemon=True)
    t_p.start(); t_c.start(); t_l.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down...")
        stop_event.set()

    # drain & join
    t_p.join()
    frame_q.join()
    t_c.join()
    t_l.join()

    # save outputs
    os.makedirs("screenshots", exist_ok=True)
    for i, f in enumerate(screenshots):
        Image.fromarray(f[..., ::-1]).save(f"screenshots/frame_{i:04d}.png")

    with open(LOG_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["time_s","p1_pct","p2_pct"])
        w.writerows(results)

    camera.stop()
    cv2.destroyAllWindows()  # Close any debug windows
    print(f"\nSaved {len(screenshots)} screenshots and {len(results)} health readings")
