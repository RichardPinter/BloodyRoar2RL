#!/usr/bin/env python3
import os
import time
import atexit
import csv
import threading
import logging
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
REGION        = (0, 0, 680, 540)
Y_HEALTH      = 115
X1_P1, X2_P1  = 78, 298
X1_P2, X2_P2  = 358, 578
LEN_P1        = X2_P1 - X1_P1
LEN_P2        = X2_P2 - X1_P2

LOWER_BGR     = np.array([15, 205, 230], dtype=np.uint8)
UPPER_BGR     = np.array([30, 220, 245], dtype=np.uint8)

FRAME_STACK   = 4
CNN_SIZE      = (84, 84)
ACTIONS       = ["jump", "kick", "transform", "squat", "left", "right"]
NUM_ACTIONS   = len(ACTIONS)
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ACTION_REPEAT = 4

DURATION      = 60.0
LOG_CSV       = "health_results.csv"
ACTIONS_FILE  = "../actions.txt"
MAX_FRAMES    = 100
LR            = 1e-4
GAMMA         = 0.99
BATCH_SIZE    = 32
TARGET_SYNC   = 1000
REPLAY_SIZE   = 10000

# ─── LOGGING ───────────────────────────────────────────────────────────────
logging.basicConfig(
    format="%(asctime)s [%(threadName)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO
)
logger        = logging.getLogger()
learner_logger = logging.getLogger("Learner")
learner_logger.setLevel(logging.INFO)

# ─── DQN NET ───────────────────────────────────────────────────────────────
class DQNNet(nn.Module):
    def __init__(self, in_ch, n_actions):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, 32, 8, 4)
        self.conv2 = nn.Conv2d(32,   64, 4, 2)
        self.conv3 = nn.Conv2d(64,   64, 3, 1)
        conv_out    = 64 * 7 * 7
        self.fc1    = nn.Linear(conv_out, 512)
        self.out    = nn.Linear(512, n_actions)

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
        self.states      = np.zeros((size, FRAME_STACK, *CNN_SIZE), dtype=np.float32)
        self.actions     = np.zeros(size, dtype=np.int64)
        self.rewards     = np.zeros(size, dtype=np.float32)
        self.next_states = np.zeros((size, FRAME_STACK, *CNN_SIZE), dtype=np.float32)
        self.dones       = np.zeros(size, dtype=np.bool_)
        self.max_size    = size
        self.ptr         = 0
        self.len         = 0

    def add(self, s, a, r, s2, done):
        self.states[self.ptr]      = s
        self.actions[self.ptr]     = a
        self.rewards[self.ptr]     = r
        self.next_states[self.ptr] = s2
        self.dones[self.ptr]       = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.len = min(self.len + 1, self.max_size)

    def sample(self, batch_size):
        idx = np.random.randint(0, self.len, size=batch_size)
        return (
            torch.from_numpy(self.states[idx]).to(DEVICE),
            torch.from_numpy(self.actions[idx]).to(DEVICE),
            torch.from_numpy(self.rewards[idx]).to(DEVICE),
            torch.from_numpy(self.next_states[idx]).to(DEVICE),
            torch.from_numpy(self.dones[idx].astype(np.uint8)).to(DEVICE),
        )

# ─── SETUP NETS & OPT ───────────────────────────────────────────────────────
policy_net = DQNNet(FRAME_STACK, NUM_ACTIONS).to(DEVICE).train()
target_net = DQNNet(FRAME_STACK, NUM_ACTIONS).to(DEVICE)
target_net.load_state_dict(policy_net.state_dict())
optimizer  = torch.optim.Adam(policy_net.parameters(), lr=LR)

# warm-up
with torch.no_grad():
    dummy = torch.zeros(1, FRAME_STACK, *CNN_SIZE, device=DEVICE)
    policy_net(dummy)

# ─── DXCAM SETUP ────────────────────────────────────────────────────────────
comtypes.CoInitialize()
def cleanup():
    camera.stop()
    comtypes.CoUninitialize()
atexit.register(cleanup)

camera = dxcam.create(output_color="BGR")
camera.start(target_fps=60, region=REGION, video_mode=True)

# ─── SHARED STATE ───────────────────────────────────────────────────────────
slice_p1    = (slice(Y_HEALTH, Y_HEALTH+1), slice(X1_P1, X2_P1), slice(None))
slice_p2    = (slice(Y_HEALTH, Y_HEALTH+1), slice(X1_P2, X2_P2), slice(None))
mask1       = np.empty((1, LEN_P1), dtype=np.uint8)
mask2       = np.empty((1, LEN_P2), dtype=np.uint8)
frame_q     = Queue(maxsize=16)
frame_stack = deque(maxlen=FRAME_STACK)
results     = []
screenshots = []
buffer      = ReplayBuffer(REPLAY_SIZE)
stop_event  = threading.Event()

# ─── PRODUCER ────────────────────────────────────────────────────────────────
def producer():
    logger.info("Producer started")
    start = time.perf_counter()
    while not stop_event.is_set():
        frm = camera.get_latest_frame()
        if frm is not None:
            ts = time.perf_counter() - start
            frame_q.put((frm.copy(), ts))
    logger.info("Producer stopped")

# ─── CONSUMER ────────────────────────────────────────────────────────────────
def consumer():
    logger.info("Consumer started")
    step = 0
    prev_stack = None
    prev_pct1 = prev_pct2 = 0.0
    frame_count = 0

    open(ACTIONS_FILE, "w").close()

    while not stop_event.is_set() or not frame_q.empty():
        frame, ts = frame_q.get()

        cv2.inRange(frame[slice_p1], LOWER_BGR, UPPER_BGR, mask1)
        pct1 = cv2.countNonZero(mask1) / LEN_P1 * 100.0
        cv2.inRange(frame[slice_p2], LOWER_BGR, UPPER_BGR, mask2)
        pct2 = cv2.countNonZero(mask2) / LEN_P2 * 100.0

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        img  = cv2.resize(gray, CNN_SIZE, interpolation=cv2.INTER_NEAREST)
        frame_stack.append(img.astype(np.float32) / 255.0)

        frame_count += 1

        if frame_count % FRAME_STACK == 0 and len(frame_stack) == FRAME_STACK:
            state = np.stack(frame_stack, 0)
            eps = max(0.01, 0.1 - 0.1 * (step / 10000))

            if random.random() < eps:
                act_idx = random.randrange(NUM_ACTIONS)
            else:
                with torch.no_grad():
                    qv = policy_net(torch.from_numpy(state).unsqueeze(0).to(DEVICE))
                act_idx = int(qv.argmax(1).item())

            with open(ACTIONS_FILE, "w") as f:
                f.write(ACTIONS[act_idx])
            # logger.info(f"[{ts:.3f}s] Wrote action '{ACTIONS[act_idx]}'")

            reward = ((prev_pct1 - pct1) - (prev_pct2 - pct2)
                      if prev_stack is not None else 0.0)
            buffer.add(
                prev_stack if prev_stack is not None else state,
                act_idx,
                reward,
                state,
                False
            )
            step += 1

            prev_stack, prev_pct1, prev_pct2 = state, pct1, pct2

            frame_stack.popleft()

        if len(screenshots) < MAX_FRAMES:
            screenshots.append(frame.copy())

        frame_q.task_done()

    logger.info("Consumer stopped")


# ─── LEARNER ────────────────────────────────────────────────────────────────
def learner():
    learner_logger.info("Learner started")
    update_count = 0
    while not stop_event.is_set():
        if buffer.len < BATCH_SIZE:
            learner_logger.debug(f"Buffer {buffer.len}/{BATCH_SIZE}, waiting…")
            time.sleep(0.01)
            continue

        if update_count == 0:
            learner_logger.info(f"Buffer ready at {buffer.len} samples, starting training")

        states, actions, rewards, next_states, dones = buffer.sample(BATCH_SIZE)
        with torch.no_grad():
            next_q = target_net(next_states).max(1)[0]
            target = rewards + GAMMA * next_q * (1 - dones.float())

        q_vals = policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        loss   = F.mse_loss(q_vals, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        update_count += 1

        learner_logger.debug(f"Step {update_count} loss={loss.item():.4f}")

        if update_count % TARGET_SYNC == 0:
            target_net.load_state_dict(policy_net.state_dict())
            learner_logger.info(f"Synced target network @ step {update_count}")

    learner_logger.info("Learner stopped")

# ─── MAIN ───────────────────────────────────────────────────────────────────
t_p = threading.Thread(target=producer, name="Producer", daemon=True)
t_c = threading.Thread(target=consumer, name="Consumer", daemon=True)
t_l = threading.Thread(target=learner, name="Learner", daemon=True)

logger.info("Launching threads")
t_p.start()
t_c.start()
t_l.start()

time.sleep(DURATION)
stop_event.set()

t_p.join()
frame_q.join()
t_c.join()
t_l.join()

# save screenshots
os.makedirs("screenshots", exist_ok=True)
for i, f in enumerate(screenshots):
    Image.fromarray(f[..., ::-1]).save(f"screenshots/frame_{i:04d}.png")

camera.stop()

# write out health CSV
with open(LOG_CSV, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["time_s", "p1_pct", "p2_pct"])
    writer.writerows(results)

logger.info("Done: health CSV='%s', screenshots in ./screenshots/, actions in '%s'",
            LOG_CSV, ACTIONS_FILE)
