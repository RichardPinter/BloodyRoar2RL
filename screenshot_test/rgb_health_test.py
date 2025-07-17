#!/usr/bin/env python3
import os
import time
import atexit
import csv
import threading
from queue import Queue
from collections import deque  # For efficient frame stacking
import hashlib  # For detecting new frames via hash

import comtypes
import dxcam
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

# ─── CONFIGURATION ───────────────────────────────────────────────────────────
REGION      = (0, 0, 680, 540)            # Cropped fight screen; ensure this includes the frame counter if needed
Y_HEALTH    = 115                          # Health bar vertical position
X1_P1, X2_P1 = 78, 298                     # P1 bar horizontal bounds (220 px)
X1_P2, X2_P2 = 358, 578                    # P2 bar horizontal bounds (220 px)
LEN_P1      = X2_P1 - X1_P1
LEN_P2      = X2_P2 - X1_P2

# Yellow-bar BGR thresholds (tuned)
LOWER_BGR   = np.array([15, 205, 230], dtype=np.uint8)
UPPER_BGR   = np.array([30, 220, 245], dtype=np.uint8)

# RL & CNN settings
FRAME_STACK   = 4
CNN_SIZE      = (84, 84)  # H×W for network input
NUM_ACTIONS   = 6
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DURATION      = 1.0        # seconds to capture
LOG_CSV       = "health_results.csv"
MAX_SCREENSHOTS = 100      # Limit screenshots to save

# ─── DQN NETWORK ─────────────────────────────────────────────────────────────
class DQNNet(nn.Module):
    def __init__(self, in_ch, n_actions):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, 32, 8, 4)
        self.conv2 = nn.Conv2d(32,   64, 4, 2)
        self.conv3 = nn.Conv2d(64,   64, 3, 1)
        conv_out = 64 * 7 * 7  # for 84×84 input
        self.fc1   = nn.Linear(conv_out, 512)
        self.out   = nn.Linear(512, n_actions)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.flatten(start_dim=1)
        x = F.relu(self.fc1(x))
        return self.out(x)

# Instantiate CNN
net = DQNNet(FRAME_STACK, NUM_ACTIONS).to(DEVICE).eval()
with torch.no_grad():
    dummy = torch.zeros(1, FRAME_STACK, *CNN_SIZE, device=DEVICE)
    for _ in range(5):
        net(dummy)

# ─── INITIALIZE COM & DXCAM ─────────────────────────────────────────────────
comtypes.CoInitialize()
def cleanup():
    camera.stop()
    comtypes.CoUninitialize()
atexit.register(cleanup)

camera = dxcam.create(output_color="BGR")
camera.start(
    target_fps=60,  # Aim for emulator-like rate
    region=REGION,
    video_mode=True  # Continuous high-rate capture (ignores change detection)
)

# Precompute slices and mask buffers
slice_p1 = (slice(Y_HEALTH, Y_HEALTH+1), slice(X1_P1, X2_P1), slice(None))
slice_p2 = (slice(Y_HEALTH, Y_HEALTH+1), slice(X1_P2, X2_P2), slice(None))
mask1    = np.empty((1, LEN_P1), dtype=np.uint8)
mask2    = np.empty((1, LEN_P2), dtype=np.uint8)

# Shared queue for frames (with timestamp)
frame_q = Queue(maxsize=16)  # Buffer for high-rate; increase if queue fills

# Frame stack as deque for O(1) pops
frame_stack = deque(maxlen=FRAME_STACK)

# Storage for results and screenshot count
results = []
screenshot_count = 0

# Global running flag
running = True

# For detecting unique frames in producer
prev_hash = None

# ─── PRODUCER THREAD: Capture continuously, save only unique screenshots ─────
def producer():
    global running, screenshot_count, prev_hash
    start = time.perf_counter()
    os.makedirs("screenshots", exist_ok=True)
    while running:
        frame = camera.get_latest_frame()
        if frame is not None:
            ts = time.perf_counter() - start
            # Hash to check uniqueness
            curr_hash = hashlib.md5(frame.tobytes()).digest()
            if curr_hash != prev_hash and screenshot_count < MAX_SCREENSHOTS:
                rgb = frame[..., ::-1]  # BGR→RGB
                Image.fromarray(rgb).save(f"screenshots/frame_{screenshot_count:04d}.png")
                screenshot_count += 1
                prev_hash = curr_hash
            # Queue every frame for processing
            frame_q.put((frame, ts))

# ─── CONSUMER THREAD: Process health, stack, CNN, log ────────────────────────
def consumer():
    global running
    while running or not frame_q.empty():
        frame, ts = frame_q.get()

        # --- Health detection ---
        strip1 = frame[slice_p1]
        cv2.inRange(strip1, LOWER_BGR, UPPER_BGR, mask1)
        cnt1   = int(cv2.countNonZero(mask1))
        pct1   = cnt1 / LEN_P1 * 100.0

        strip2 = frame[slice_p2]
        cv2.inRange(strip2, LOWER_BGR, UPPER_BGR, mask2)
        cnt2   = int(cv2.countNonZero(mask2))
        pct2   = cnt2 / LEN_P2 * 100.0

        # Record health
        results.append((f"{ts:.3f}", f"{pct1:.1f}", f"{pct2:.1f}"))

        # --- Preprocessing for CNN ---
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        img  = cv2.resize(gray, CNN_SIZE, interpolation=cv2.INTER_NEAREST)
        frame_stack.append(img.astype(np.float32) / 255.0)

        # When we have enough frames, run CNN
        if len(frame_stack) == FRAME_STACK:
            state = np.stack(frame_stack, axis=0)
            tensor = torch.from_numpy(state).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                q = net(tensor)
                if DEVICE.type == 'cuda':
                    torch.cuda.synchronize()
            action = int(q.argmax(dim=1).item())
            print(f"[{ts:.3f}s] Health P1={pct1:.1f}% P2={pct2:.1f}% -> Action={action}")
            frame_stack.popleft()

        frame_q.task_done()

# ─── MAIN: Start threads, run for duration, shutdown ─────────────────────────
print(f"Starting capture for {DURATION}s...")
t_prod = threading.Thread(target=producer, daemon=True)
t_cons = threading.Thread(target=consumer, daemon=True)
t_prod.start()
t_cons.start()

time.sleep(DURATION)
running = False

frame_q.join()

# ─── TEARDOWN & WRITE CSV ────────────────────────────────────────────────────
camera.stop()
with open(LOG_CSV, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["time_s", "p1_pct", "p2_pct"])
    writer.writerows(results)

print(f"Captured {len(results)} frames, saved {screenshot_count} screenshots, and processed CNN steps → {LOG_CSV}")