#!/usr/bin/env python3
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
HEALTH_LIMIT = 99.0  # Health percentage to consider "full"
SPAM_START_FRAMES = 5   # send "start\n" for 5 frames
SPAM_KICK_FRAMES  = 3 

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

# ─── SETUP ───────────────────────────────────────────────────────────────────
policy_net = DQNNet(FRAME_STACK, NUM_ACTIONS).to(DEVICE).train()
target_net = DQNNet(FRAME_STACK, NUM_ACTIONS).to(DEVICE)
target_net.load_state_dict(policy_net.state_dict())
optimizer  = torch.optim.Adam(policy_net.parameters())
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
match_number = 1
# ─── PRODUCER ────────────────────────────────────────────────────────────────
def producer():
    start = time.perf_counter()
    while not stop_event.is_set():
        frm = camera.get_latest_frame()
        if frm is not None:
            ts = time.perf_counter() - start
            frame_q.put((frm.copy(), ts))

# ─── SINGLE CONSUMER w/ HEALTH & ROUND LOGIC ────────────────────────────────
match_number = 1


def consumer():
    global match_number    # now refers to module‐level `match_number`

    state        = "waiting"
    alive_since  = None
    death_since  = None

    p1_wins      = 0
    p2_wins      = 0
    match_over   = False
    p1_losses    = 0

    spam_counter = 0
    spam_done    = False

    YEL_HSV_LOW  = np.array([20,  50,  50], dtype=np.uint8)
    YEL_HSV_HIGH = np.array([40, 255, 255], dtype=np.uint8)

    def write_action(text: str):
        with open(ACTIONS_FILE, "w") as f:
            f.write(text)

    # ── TRAINING HOOKS ───────────────────────────────
    def on_round_end():
        # Drain buffer in BATCH_SIZE chunks
        while buffer.len >= BATCH_SIZE:
            states, actions, rewards, next_states, dones = buffer.sample(BATCH_SIZE)
            with torch.no_grad():
                next_q   = target_net(next_states).max(1)[0]
                target_q = rewards + GAMMA * next_q * (1 - dones.float())

            q_vals = policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
            loss   = F.mse_loss(q_vals, target_q)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Only after *all* updates, sync target_net
        target_net.load_state_dict(policy_net.state_dict())
        print("🛠️  [TRAIN] Completed per‐round updates")

    def on_match_end():
        global match_number
        print(f"🛠️  [TRAIN] on_match_end() — saving model #{match_number}")
        torch.save(policy_net.state_dict(),
                   f"model_match_{match_number}.pth")
        match_number += 1
    # ────────────────────────────────────────────────

    while not stop_event.is_set():
        frame, ts = frame_q.get()

        # 1) Compute health % once
        strip = frame[Y_HEALTH:Y_HEALTH+1]
        m1    = cv2.inRange(strip[:, X1_P1:X2_P1], LOWER_BGR, UPPER_BGR)
        m2    = cv2.inRange(strip[:, X1_P2:X2_P2], LOWER_BGR, UPPER_BGR)
        pct1  = cv2.countNonZero(m1) / LEN_P1 * 100.0
        pct2  = cv2.countNonZero(m2) / LEN_P2 * 100.0
        

        # 2) Compute any-yellow via HSV for death detection
        hsv_strip = cv2.cvtColor(strip, cv2.COLOR_BGR2HSV)
        any1 = cv2.countNonZero(cv2.inRange(
            hsv_strip[:, X1_P1:X2_P1], YEL_HSV_LOW, YEL_HSV_HIGH))
        any2 = cv2.countNonZero(cv2.inRange(
            hsv_strip[:, X1_P2:X2_P2], YEL_HSV_LOW, YEL_HSV_HIGH))

        # 3) WAITING → detect round start
        if state == "waiting":
            if pct1 >= HEALTH_LIMIT and pct2 >= HEALTH_LIMIT:
                alive_since = alive_since or time.time()
                if time.time() - alive_since >= 0.5:
                    print("→ Round started!")
                    write_action("start\n")
                    state = "active"
                    frame_stack.clear()
                    # reset spam if coming out of a match
                    if match_over:
                        spam_counter = 0
                        spam_done    = False
            else:
                alive_since = None

        # 4) ACTIVE → DQN actions + death detection
        elif state == "active":
            # a) choose & write a DQN action
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            img  = cv2.resize(gray, CNN_SIZE, interpolation=cv2.INTER_NEAREST)
            frame_stack.append(img.astype(np.float32) / 255.0)

            if len(frame_stack) == FRAME_STACK:
                st  = np.stack(frame_stack, 0)
                eps = max(0.01, 0.1 - 0.1 * (buffer.len / 10000))
                if random.random() < eps:
                    a = random.randrange(NUM_ACTIONS)
                else:
                    with torch.no_grad():
                        q = policy_net(torch.from_numpy(st)
                                       .unsqueeze(0).to(DEVICE))
                    a = int(q.argmax(1).item())
                write_action(ACTIONS[a] + "\n")
                buffer.add(frame_stack[0], a, 0.0, st, False)
                frame_stack.popleft()

            # b) detect death when no yellow remains ≥ 1s
            if any1 == 0 or any2 == 0:
                death_since = death_since or time.time()
                if time.time() - death_since >= 1.0:
                    loser = 1 if any1 == 0 else 2
                    winner = "p2" if loser == 1 else "p1"
                    print(f"← Player {loser} died → Winner: {winner}")

                    # round-ended hook
                    on_round_end()

                    # update match score
                    if winner == "p1":
                        p1_wins += 1
                    else:
                        p2_wins += 1

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
                        # reset for next match
                        p1_wins = p2_wins = 0

                    # back to waiting
                    state        = "waiting"
                    alive_since  = death_since = None
            else:
                death_since = None

        # 5) POST-MATCH spam (only if P2 won the match)
        if match_over and state == 'waiting' and p1_losses > 0:
            text = "start\n" if (spam_counter % 2) == 0 else "kick\n"
            write_action(text)
            spam_counter += 1
            frame_q.task_done()
            continue

        frame_q.task_done()


# ─── LEARNER ────────────────────────────────────────────────────────────────
def learner():
    round_num = 1
    while not stop_event.is_set():
        if buffer.len < BATCH_SIZE:
            time.sleep(0.01)
            continue

        # train until empty
        while buffer.len >= BATCH_SIZE:
            states, actions, rewards, next_states, dones = buffer.sample(BATCH_SIZE)
            with torch.no_grad():
                next_q = target_net(next_states).max(1)[0]
                target = rewards + GAMMA*next_q*(1-dones.float())

            q_vals = policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
            loss   = F.mse_loss(q_vals, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        target_net.load_state_dict(policy_net.state_dict())
        torch.save(policy_net.state_dict(), f"model_round_{round_num}.pth")
        buffer.clear()
        round_num += 1

# ─── MAIN ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # start threads
    t_p = threading.Thread(target=producer, name="Producer", daemon=True)
    t_c = threading.Thread(target=consumer, name="Consumer", daemon=True)
    t_l = threading.Thread(target=learner,  name="Learner",  daemon=True)
    t_p.start(); t_c.start(); t_l.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
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
