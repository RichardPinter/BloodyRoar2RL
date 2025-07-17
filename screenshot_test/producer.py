import os
import time
import atexit
import csv
import threading
from queue import Queue
from collections import deque  # For efficient frame stacking

import comtypes
import dxcam
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

# ─── CONFIGURATION ───────────────────────────────────────────────────────────
REGION      = (0, 0, 680, 540)            # Cropped fight screen
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
ACTIONS = [
    "jump", "kick", "transform", "squat"
]
NUM_ACTIONS   = len(ACTIONS)
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ACTION_REPEAT = 4  # Repeat/hold each action for 4 frames
ACTIONS_FILE  = r"C:\Users\richa\Desktop\Personal\Uni\ShenLong\actions.txt"

DURATION      = 5.0        # Increased duration for testing the sequence
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

# Storage for results and screenshot frames
results = []
screenshot_frames = []  # up to MAX_SCREENSHOTS

# Thread-safe stop event
stop_event = threading.Event()

# ─── PRODUCER THREAD: Pure capture, no IO ───────────────────────────────────
def producer():
    start = time.perf_counter()
    while not stop_event.is_set():
        frame = camera.get_latest_frame()
        if frame is not None:
            ts = time.perf_counter() - start
            frame_q.put((frame.copy(), ts))

# ─── CONSUMER THREAD: Process health, stack, CNN, log; collect frames for saving
def consumer():
    frame_count = 0
    action_index = 0
    test_sequence = ACTIONS
    current_action = test_sequence[0]

    while not stop_event.is_set() or not frame_q.empty():
        frame, ts = frame_q.get()

        # Capture screenshot frames
        if len(screenshot_frames) < MAX_SCREENSHOTS:
            screenshot_frames.append(frame.copy())

        # Health detection
        strip1 = frame[slice_p1]
        cv2.inRange(strip1, LOWER_BGR, UPPER_BGR, mask1)
        cnt1   = int(cv2.countNonZero(mask1))
        pct1   = cnt1 / LEN_P1 * 100.0

        strip2 = frame[slice_p2]
        cv2.inRange(strip2, LOWER_BGR, UPPER_BGR, mask2)
        cnt2   = int(cv2.countNonZero(mask2))
        pct2   = cnt2 / LEN_P2 * 100.0

        results.append((f"{ts:.3f}", f"{pct1:.1f}", f"{pct2:.1f}"))

        # CNN preprocessing (optional)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        img  = cv2.resize(gray, CNN_SIZE, interpolation=cv2.INTER_NEAREST)
        frame_stack.append(img.astype(np.float32) / 255.0)

        frame_count += 1
        # Action switching
        if len(frame_stack) == FRAME_STACK:
            if frame_count % ACTION_REPEAT == 0:
                action_index = (action_index + 1) % len(test_sequence)
                current_action = test_sequence[action_index]
                # Write action once when it changes
                with open(ACTIONS_FILE, "w") as f:
                    f.write(current_action)
                print(f"[{ts:.3f}s] Health P1={pct1:.1f}% P2={pct2:.1f}% -> Action={current_action}")

            frame_stack.popleft()
        frame_q.task_done()

# ─── MAIN: Start threads, run, shutdown ───────────────────────────────────────
t_prod = threading.Thread(target=producer, daemon=True)
t_cons = threading.Thread(target=consumer, daemon=True)
t_prod.start()
t_cons.start()

print(f"Starting capture for {DURATION}s...")
time.sleep(DURATION)
stop_event.set()

# Wait for all frames to be processed
frame_q.join()

# Save screenshots
os.makedirs("screenshots", exist_ok=True)
for i, frame in enumerate(screenshot_frames):
    rgb = frame[..., ::-1]
    Image.fromarray(rgb).save(f"screenshots/frame_{i:04d}.png")

# Teardown and write CSV
camera.stop()
with open(LOG_CSV, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["time_s", "p1_pct", "p2_pct"])
    writer.writerows(results)

print(f"Captured {len(results)} frames, saved {len(screenshot_frames)} screenshots, results → {LOG_CSV}")
