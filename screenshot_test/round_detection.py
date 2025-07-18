import os
import time
import atexit
import csv
import threading
from queue import Queue
from collections import deque
import random

import comtypes
import dxcam
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

# ─── CONFIG ────────────────────────────────────────────────────────────────
REGION      = (0, 0, 624, 548)      # x, y, width, height
Y_HEALTH    = 116
X1_P1, X2_P1 = 73, 292
X1_P2, X2_P2 = 355, 574
LEN_P1        = X2_P1 - X1_P1
LEN_P2        = X2_P2 - X1_P2

LOWER_BGR   = np.array([0,150,180], dtype=np.uint8)
UPPER_BGR   = np.array([30,175,220], dtype=np.uint8)

FRAME_STACK   = 4
CNN_SIZE      = (84, 84)
ACTIONS       = ["jump", "kick", "transform", "squat", "left", "right"]
NUM_ACTIONS   = len(ACTIONS)
ACTION_REPEAT = 4

DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LOG_CSV       = "health_results.csv"
ACTIONS_FILE  = "../actions.txt"
MAX_FRAMES    = 100
GAMMA         = 0.99
BATCH_SIZE    = 32
TARGET_SYNC   = 1000
REPLAY_SIZE   = 10000
MIN_BUFFER_SIZE = 1000  # Minimum samples before training starts
HEALTH_LIMIT = 99.0  # Health percentage to consider "full"
LEARNING_RATE = 1e-4

# Model loading and test mode
LOAD_CHECKPOINT = "model_match_20.pth"  # Set to None to train from scratch
TEST_MODE = False  # Set to True to disable training and just play

# Round indicator monitoring
# IMPORTANT: Update these coordinates to match your game's round indicators!
# These are small rectangles that turn red when a player wins a round
ROUND_INDICATORS = {
    'p1_round1': ( 270, 135, 278, 140),   # Player 1, first round indicator
    'p1_round2': ( 245, 135, 253, 140),   # Player 1, second round indicator
    'p2_round1': ( 373, 135, 381, 140),   # Player 2, first round indicator
    'p2_round2': ( 396, 135, 404, 140),   # Player 2, second round indicator
}
RED_BGR_LOWER = np.array([0, 0, 150], dtype=np.uint8)    # Lower red threshold
RED_BGR_UPPER = np.array([60, 60, 255], dtype=np.uint8)  # Upper red threshold
RED_THRESHOLD = 0.3  # 30% of rectangle must be red
DEBUG_RECTANGLES = True  # Show debug visualization

# ─── DQN NET ───────────────────────────────────────────────────────────────
class DQNNet(nn.Module):
    def __init__(self, in_ch, n_actions):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, 32, 8, 4)
        self.conv2 = nn.Conv2d(32, 64, 4, 2)
        self.conv3 = nn.Conv2d(64, 64, 3, 1)
        conv_out = 64 * 7 * 7
        self.fc1   = nn.Linear(conv_out, 512)
        self.out   = nn.Linear(512, n_actions)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.flatten(1)
        x = F.relu(self.fc1(x))
        return self.out(x)

# ─── REPLAY BUFFER ─────────────────────────────────────────────────────────
class ReplayBuffer:
    def __init__(self, size):
        self.size = size
        self.clear()

    def clear(self):
        self.states      = np.zeros((self.size, FRAME_STACK, *CNN_SIZE), dtype=np.float32)
        self.actions     = np.zeros(self.size, dtype=np.int64)
        self.rewards     = np.zeros(self.size, dtype=np.float32)
        self.next_states = np.zeros((self.size, FRAME_STACK, *CNN_SIZE), dtype=np.float32)
        self.dones       = np.zeros(self.size, dtype=bool)
        self.ptr = 0
        self.len = 0

    def add(self, s, a, r, s2, done):
        self.states[self.ptr]      = s
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
            torch.from_numpy(self.actions[idx]).to(DEVICE),
            torch.from_numpy(self.rewards[idx]).to(DEVICE),
            torch.from_numpy(self.next_states[idx]).to(DEVICE),
            torch.from_numpy(self.dones[idx].astype(np.uint8)).to(DEVICE),
        )

# ─── ROUND INDICATOR MONITOR ───────────────────────────────────────────────
class RoundIndicatorMonitor:
    def __init__(self):
        self.rectangles = ROUND_INDICATORS
        self.red_lower = RED_BGR_LOWER
        self.red_upper = RED_BGR_UPPER
        self.threshold = RED_THRESHOLD

        # State tracking
        self.last_state = {'p1_rounds': 0, 'p2_rounds': 0}
        self.change_counter = 0
        self.confirmed_state = {'p1_rounds': 0, 'p2_rounds': 0}

    def check_indicator(self, frame, rect_key):
        """Check if a specific rectangle indicator is red (won)"""
        if rect_key not in self.rectangles:
            return False

        x1, y1, x2, y2 = self.rectangles[rect_key]

        # Ensure coordinates are within frame bounds
        h, w = frame.shape[:2]
        x1, x2 = max(0, x1), min(w, x2)
        y1, y2 = max(0, y1), min(h, y2)

        if x2 <= x1 or y2 <= y1:
            return False

        region = frame[y1:y2, x1:x2]

        # Create mask for red pixels
        mask = cv2.inRange(region, self.red_lower, self.red_upper)

        # Calculate percentage
        total_pixels = region.shape[0] * region.shape[1]
        if total_pixels == 0:
            return False

        red_pixels = np.count_nonzero(mask)
        red_percentage = red_pixels / total_pixels

        return red_percentage > self.threshold

    def get_red_percentage(self, frame, rect_key):
        """Get the percentage of red pixels in a rectangle (for debugging)"""
        if rect_key not in self.rectangles:
            return 0.0

        x1, y1, x2, y2 = self.rectangles[rect_key]

        # Ensure coordinates are within frame bounds
        h, w = frame.shape[:2]
        x1, x2 = max(0, x1), min(w, x2)
        y1, y2 = max(0, y1), min(h, y2)

        if x2 <= x1 or y2 <= y1:
            return 0.0

        region = frame[y1:y2, x1:x2]
        mask = cv2.inRange(region, self.red_lower, self.red_upper)

        total_pixels = region.shape[0] * region.shape[1]
        if total_pixels == 0:
            return 0.0

        red_pixels = np.count_nonzero(mask)
        return red_pixels / total_pixels

    def detect_round_changes(self, frame):
        """
        Detect if a round just ended and who won.
        Returns: (round_winner, current_state, debug_info)
        """
        # Check all indicators
        current_state = {
            'p1_rounds': sum([
                self.check_indicator(frame, 'p1_round1'),
                self.check_indicator(frame, 'p1_round2')
            ]),
            'p2_rounds': sum([
                self.check_indicator(frame, 'p2_round1'),
                self.check_indicator(frame, 'p2_round2')
            ])
        }

        # Debug info
        debug_info = {
            'p1_r1_pct': self.get_red_percentage(frame, 'p1_round1'),
            'p1_r2_pct': self.get_red_percentage(frame, 'p1_round2'),
            'p2_r1_pct': self.get_red_percentage(frame, 'p2_round1'),
            'p2_r2_pct': self.get_red_percentage(frame, 'p2_round2'),
        }

        # Debouncing: require consistent state for 3 frames
        if current_state != self.last_state:
            self.change_counter += 1
            if self.change_counter >= 3:  # 3 frames of consistent change
                # Determine winner
                round_winner = None
                if current_state['p1_rounds'] > self.confirmed_state['p1_rounds']:
                    round_winner = 'p1'
                elif current_state['p2_rounds'] > self.confirmed_state['p2_rounds']:
                    round_winner = 'p2'

                self.confirmed_state = current_state.copy()
                self.change_counter = 0
                return round_winner, current_state, debug_info
        else:
            self.change_counter = 0

        self.last_state = current_state.copy()
        return None, self.confirmed_state, debug_info

    def draw_debug_rectangles(self, frame):
        """Draw debug rectangles and percentages on frame"""
        debug_frame = frame.copy()

        for name, (x1, y1, x2, y2) in self.rectangles.items():
            is_red = self.check_indicator(frame, name)
            pct = self.get_red_percentage(frame, name)

            # Color: Red if indicator is "won", green otherwise
            color = (0, 0, 255) if is_red else (0, 255, 0)

            # Draw rectangle
            cv2.rectangle(debug_frame, (x1, y1), (x2, y2), color, 2)

            # Draw percentage text
            text = f"{name}: {pct:.1%}"
            cv2.putText(debug_frame, text, (x1, y1-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        return debug_frame

# ─── SETUP ───────────────────────────────────────────────────────────────────
import re

policy_net = DQNNet(FRAME_STACK, NUM_ACTIONS).to(DEVICE).train()
target_net = DQNNet(FRAME_STACK, NUM_ACTIONS).to(DEVICE)

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
    target_net.load_state_dict(policy_net.state_dict())
    if LOAD_CHECKPOINT:
        print(f"⚠️  Checkpoint {LOAD_CHECKPOINT} not found, training from scratch")

optimizer  = torch.optim.Adam(policy_net.parameters(), lr=LEARNING_RATE)
buffer     = ReplayBuffer(REPLAY_SIZE)

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

    p1_wins      = 0
    p2_wins      = 0
    match_over   = False
    p1_losses    = 0

    spam_counter = 0
    spam_done    = False

    # Proper state tracking for rewards
    prev_state   = None
    prev_action  = None
    prev_pct1    = 100.0
    prev_pct2    = 100.0

    # Round indicator monitoring
    round_monitor = RoundIndicatorMonitor()

    def write_action(text: str):
        with open(ACTIONS_FILE, "w") as f:
            f.write(text)

    # ── TRAINING HOOKS (SIMPLIFIED - NO TRAINING HERE) ────
    def on_round_end():
        round_end_event.set()  # Just signal the learner thread
        print("[Consumer] Round ended, signaled learner")

    def on_match_end():
        match_end_event.set()  # Signal the learner thread for match end
        print(f"[Consumer] Match ended, signaled learner")
    # ────────────────────────────────────────────────

    last_log_time = 0

    while not stop_event.is_set():
        frame, ts = frame_q.get()

        # 1) Compute health % once
        strip = frame[Y_HEALTH:Y_HEALTH+1]
        m1    = cv2.inRange(strip[:, X1_P1:X2_P1], LOWER_BGR, UPPER_BGR)
        m2    = cv2.inRange(strip[:, X1_P2:X2_P2], LOWER_BGR, UPPER_BGR)
        pct1  = cv2.countNonZero(m1) / LEN_P1 * 100.0
        pct2  = cv2.countNonZero(m2) / LEN_P2 * 100.0

        # Log health periodically
        if time.time() - last_log_time > 2.0:
            print(f"[Health] P1: {pct1:.1f}%, P2: {pct2:.1f}%, State: {state}, Buffer: {buffer.len}")
            last_log_time = time.time()

        # 2) Check round indicators for round end detection
        round_winner, round_state, debug_info = round_monitor.detect_round_changes(frame)

        # Debug output for rectangle monitoring
        if time.time() - last_log_time > 2.0:
            print(f"[Rounds] P1: {round_state['p1_rounds']}, P2: {round_state['p2_rounds']}")
            if DEBUG_RECTANGLES:
                for key, pct in debug_info.items():
                    print(f"  {key}: {pct:.1%}")

        # Optional: Show debug rectangles (for tuning)
        if DEBUG_RECTANGLES:
            debug_frame = round_monitor.draw_debug_rectangles(frame)
            cv2.imshow("Round Indicators Debug", debug_frame)
            cv2.waitKey(1)

        # 3) WAITING → detect round start
        if state == "waiting":
            if pct1 >= HEALTH_LIMIT and pct2 >= HEALTH_LIMIT:
                alive_since = alive_since or time.time()
                if time.time() - alive_since >= 0.5:
                    print("→ Round started!")
                    write_action("start\n")
                    state = "active"
                    frame_stack.clear()
                    # Reset tracking variables
                    prev_state = None
                    prev_action = None
                    prev_pct1 = pct1
                    prev_pct2 = pct2
                    # reset spam if coming out of a match
                    if match_over:
                        spam_counter = 0
                        spam_done    = False
                        match_over   = False
            else:
                alive_since = None

        # 4) ACTIVE → DQN actions + death detection
        elif state == "active":
            # a) Prepare frame
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            img  = cv2.resize(gray, CNN_SIZE, interpolation=cv2.INTER_NEAREST)
            frame_stack.append(img.astype(np.float32) / 255.0)

            if len(frame_stack) == FRAME_STACK:
                current_state = np.stack(frame_stack, 0)

                # Calculate reward from previous action
                if prev_state is not None and prev_action is not None:
                    # Reward: opponent damage - our damage
                    our_damage = prev_pct1 - pct1
                    opp_damage = prev_pct2 - pct2
                    reward = opp_damage - our_damage

                    # Store transition with proper reward
                    buffer.add(prev_state, prev_action, reward, current_state, False)

                # Select action
                if TEST_MODE:
                    eps = 0.01  # Very low exploration in test mode
                else:
                    eps = max(0.01, 0.1 - 0.1 * (buffer.len / 10000))

                if random.random() < eps:
                    action = random.randrange(NUM_ACTIONS)
                else:
                    with torch.no_grad():
                        q = policy_net(torch.from_numpy(current_state).unsqueeze(0).to(DEVICE))
                    action = int(q.argmax(1).item())

                write_action(ACTIONS[action] + "\n")

                # Store for next frame
                prev_state = current_state
                prev_action = action
                prev_pct1 = pct1
                prev_pct2 = pct2

                frame_stack.popleft()

        # Check for round end via rectangle monitoring (only during active gameplay)
        if state == "active" and round_winner:
            print(f"← Round ended → Winner: {round_winner}")

            # Final transition with done=True and terminal reward
            if prev_state is not None and prev_action is not None:
                final_reward = 10.0 if round_winner == "p1" else -10.0
                buffer.add(prev_state, prev_action, final_reward, prev_state, True)

            # round-ended hook
            on_round_end()

            # Update match score using rectangle monitoring results
            p1_wins = round_state['p1_rounds']
            p2_wins = round_state['p2_rounds']

            # match-over: first to 2 wins
            if p1_wins >= 2 or p2_wins >= 2:
                match_over = True
                spam_done  = False
                spam_counter = 0
                print(f"🏁 Match over! Score P1:{p1_wins} P2:{p2_wins}")
                if p2_wins > p1_wins:
                    p1_losses += 1
                    print(f"❌ Player 1 lost the match ({p1_losses} losses)")
                # match-ended hook
                on_match_end()
                # Don't reset p1_wins/p2_wins - they're tracked by rectangle monitoring

            # back to waiting
            state = "waiting"
            alive_since = death_since = None

        # 5) POST-MATCH spam (only if P2 won the match)
        if match_over and state == 'waiting' and p1_losses > 0:
            text = "start\n" if (spam_counter % 2) == 0 else "kick\n"
            write_action(text)
            spam_counter += 1

        # Save results
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
            states, actions, rewards, next_states, dones = buffer.sample(BATCH_SIZE)

            with torch.no_grad():
                next_q = target_net(next_states).max(1)[0]
                target = rewards + GAMMA * next_q * (1 - dones.float())

            q_vals = policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
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
    print(f"Round detection: Rectangle monitoring (threshold: {RED_THRESHOLD:.1%})")
    if DEBUG_RECTANGLES:
        print("Debug rectangles: ENABLED")
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