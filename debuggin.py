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

FRAME_STACK     = 10
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

START_TIME = time.perf_counter()
def now():
    """Seconds since START_TIME as a float."""
    return time.perf_counter() - START_TIME

BLACK_BGR = np.array([0, 0, 8], dtype=np.uint8)
results = []
screenshots = []
debug_transitions = []
action_log = []  
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


import cv2

def overlay_timestamp(frame, ts):
    """
    Draw ts (seconds) in the top-right corner of frame.
    """
    h, w = frame.shape[:2]
    label = f"{ts:0.3f}s"
    font, scale, thickness = cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
    (text_w, text_h), _ = cv2.getTextSize(label, font, scale, thickness)
    x = w - text_w - 5
    y = text_h + 5
    cv2.putText(frame, label, (x, y), font, scale, (0,255,0), thickness, cv2.LINE_AA)

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
            try:
                frame_q.put((frm.copy(), ts), timeout=0.5)
            except:
                # Queue is full or shutting down
                pass

# ─── SINGLE CONSUMER w/ HEALTH & ROUND LOGIC ────────────────────────────────
def consumer():
    global match_number

    def write_action(text: str):
        with open(ACTIONS_FILE, "w") as f:
            f.write(text)

    state        = "waiting"
    alive_since  = None
    death_since  = None

    prev_state      = None
    prev_action     = None
    prev_pct1       = 100.0
    prev_pct2       = 100.0
    prev_extra_feats= None

    frame_stack     = deque(maxlen=FRAME_STACK)
    write_count     = 0

    hold_counter    = 0       # frames left to hold the current action
    current_action  = None    # index of the held action

    print("[Test Mode] Consumer started — collecting 1000 transitions...")

    while not stop_event.is_set():
        frame, _ = frame_q.get()
        ts = time.perf_counter()

        # Overlay the same timestamp you’ll log
        overlay_timestamp(frame, ts)

        # ─── 1) Compute health % ───────────────────────────────────
        strip = frame[Y_HEALTH:Y_HEALTH+1]
        m1    = cv2.inRange(strip[:, X1_P1:X2_P1], LOWER_BGR, UPPER_BGR)
        m2    = cv2.inRange(strip[:, X1_P2:X2_P2], LOWER_BGR, UPPER_BGR)
        pct1  = cv2.countNonZero(m1) / LEN_P1 * 100.0
        pct2  = cv2.countNonZero(m2) / LEN_P2 * 100.0

        # ─── 2) Feature extraction ─────────────────────────────────
        ts_dict   = classify_transform_state(frame)
        bp_dict,_ = compute_black_stats(frame)
        code = {'can transform':0, 'transformed':1, 'cannot transform':2, 'unknown':3}
        f_transform = [ code[ts_dict['P1_R1_pixel']], code[ts_dict['P2_R2_pixel']] ]
        f_black_pct = [ bp_dict['P1_R1_area']/100.0, bp_dict['P2_R2_area']/100.0 ]
        extra_feats = np.array(f_transform + f_black_pct, dtype=np.float32)

        # ─── 3) Frame stack preprocessing ───────────────────────────
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        img  = cv2.resize(gray, CNN_SIZE, interpolation=cv2.INTER_NEAREST)
        frame_stack.append(img.astype(np.float32) / 255.0)

        # ─── 4) Compute reward from prev action ────────────────────
        if prev_state is not None and prev_action is not None:
            our_damage = prev_pct1 - pct1
            opp_damage = prev_pct2 - pct2
            reward     = opp_damage - our_damage

            buffer.add(prev_state, prev_extra_feats, prev_action,
                       reward, np.stack(frame_stack,0), False)

            debug_transitions.append({
                'time':    ts,
                'step':    write_count,
                'action':  ACTIONS[prev_action],
                'reward':  reward,
                'p1_pct':  prev_pct1,
                'p2_pct':  prev_pct2,
                'extras':  prev_extra_feats.tolist(),
            })
        else:
            reward = 0.0

        # ─── 5) Decide new action every FRAME_STACK frames ──────────
        if len(frame_stack) == FRAME_STACK and hold_counter == 0:
            state_img    = torch.from_numpy(np.stack(frame_stack,0)).unsqueeze(0).to(DEVICE)
            extra_tensor = torch.from_numpy(extra_feats).unsqueeze(0).to(DEVICE)

            eps = 0.5
            if random.random() < eps:
                chosen = random.randrange(NUM_ACTIONS)
            else:
                with torch.no_grad():
                    q = policy_net(state_img, extra_tensor)
                    chosen = int(q.argmax(1).item())

            current_action = chosen
            hold_counter   = FRAME_STACK

            action_log.append({
                'time':   ts,
                'step':   write_count,
                'action': ACTIONS[chosen]
            })

        # ─── 6) Write (or hold) the action every frame ─────────────
        if current_action is not None:
            write_action(ACTIONS[current_action] + "\n")
            hold_counter -= 1

        # ─── 7) Update prev_* for next reward calc ────────────────
        if current_action is not None:
            prev_state       = np.stack(frame_stack,0)
            prev_extra_feats = extra_feats
            prev_action      = current_action
            prev_pct1        = pct1
            prev_pct2        = pct2

        # ─── 8) Increment counter & check exit ────────────────────
        write_count += 1
        if write_count >= 1000:
            print("[Test Mode] Collected 1000 transitions. Exiting...")
            camera.stop()
            comtypes.CoUninitialize()
            cv2.destroyAllWindows()
            stop_event.set()
            break

        # ─── 9) Always save screenshots & health readings ─────────
        if len(screenshots) < MAX_FRAMES:
            screenshots.append(frame.copy())
        results.append((ts, pct1, pct2))

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

    # Start threads
    t_p = threading.Thread(target=producer, name="Producer")
    t_c = threading.Thread(target=consumer, name="Consumer")
    t_l = threading.Thread(target=learner,  name="Learner")
    t_p.start(); t_c.start(); t_l.start()

    try:
        print("[Main] Waiting for consumer to finish...")
        t_c.join(timeout=10)
        if t_c.is_alive():
            print("⚠️ Consumer thread still alive after timeout.")
        else:
            print("✅ Consumer exited cleanly.")
        stop_event.set()  # Signal producer and learner to exit
    except KeyboardInterrupt:
        print("\nShutting down via KeyboardInterrupt...")
        stop_event.set()

    # Join producer and learner with timeout
    for t in (t_p, t_l):
        print(f"[Main] Waiting for {t.name} to exit...")
        t.join(timeout=5)
        if t.is_alive():
            print(f"⚠️ Thread {t.name} still alive after timeout.")
        else:
            print(f"✅ Thread {t.name} exited.")

    # Optional: skip frame_q.join() unless you're certain all .task_done() are called
    print("[Main] Skipping frame_q.join() for safety.")

    # Fallback cleanup
    try:
        camera.stop()
        print("[Main] Camera stopped.")
    except Exception as e:
        print(f"[Main] Failed to stop camera: {e}")
    try:
        comtypes.CoUninitialize()
        print("[Main] COM uninitialized.")
    except Exception as e:
        print(f"[Main] Failed to uninitialize COM: {e}")
    try:
        cv2.destroyAllWindows()
        print("[Main] OpenCV windows closed.")
    except Exception as e:
        print(f"[Main] Failed to close OpenCV windows: {e}")

    # ─── SAVE ALL DATA AFTER THREADS SHUT DOWN ──────────────────────────────
    import json
    import csv
    import os
    from PIL import Image
    import sys

    # 1. Save debug transitions
    # Save results with timestamps
    with open("health_results.csv","w",newline="") as f:
        w = csv.writer(f)
        w.writerow(["time_s","p1_pct","p2_pct"])
        w.writerows(results)

    # Save debug_transitions
    with open("debug_transitions.json","w") as f:
        json.dump(debug_transitions, f, indent=2)

    # Save action_log if you made it
    with open("action_log.csv","w",newline="") as f:
        w = csv.DictWriter(f, fieldnames=['time','step','action'])
        w.writeheader()
        w.writerows(action_log)

    # 3. Save screenshots
    os.makedirs("screenshots", exist_ok=True)
    for i, frame in enumerate(screenshots):
        Image.fromarray(frame[..., ::-1]).save(f"screenshots/frame_{i:04d}.png")
    print(f"✅ Saved {len(screenshots)} screenshots")

    # 4. Print active threads (debugging)
    import threading
    print("\n🧵 Final alive threads:")
    for t in threading.enumerate():
        if t.name != "MainThread":
            print(f" - {t.name:10s} | alive={t.is_alive()} | daemon={t.daemon}")

    # 5. Exit cleanly
    print("🛑 Exiting after successful logging.")
    sys.exit(0)

    # Optional: Force kill if stuck
    # import os
    # os._exit(0)
