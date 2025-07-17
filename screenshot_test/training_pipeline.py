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

# ─── CONFIGURATION ───────────────────────────────────────────────────────────
REGION        = (0, 0, 680, 540)
Y_HEALTH      = 115
X1_P1, X2_P1 = 78, 298
X1_P2, X2_P2 = 358, 578
LEN_P1        = X2_P1 - X1_P1
LEN_P2        = X2_P2 - X1_P2

LOWER_BGR   = np.array([15, 205, 230], dtype=np.uint8)
UPPER_BGR   = np.array([30, 220, 245], dtype=np.uint8)

FRAME_STACK   = 4
CNN_SIZE      = (84, 84)
ACTIONS       = ["jump", "kick", "transform", "squat"]
NUM_ACTIONS   = len(ACTIONS)
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ACTION_REPEAT = 4
ACTIONS_FILE  = os.path.abspath("../actions.txt")  # write in current working directory

EPS_START     = 1.0
EPS_END       = 0.1
EPS_DECAY     = 10000

DURATION        = 5.0
LOG_CSV         = "health_results.csv"
MAX_SCREENSHOTS = 100

# ─── DQN NETWORK ─────────────────────────────────────────────────────────────
class DQNNet(nn.Module):
    def __init__(self, in_ch, n_actions):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, 32, 8, 4)
        self.conv2 = nn.Conv2d(32,  64, 4, 2)
        self.conv3 = nn.Conv2d(64,  64, 3, 1)
        conv_out = 64 * 7 * 7
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

# ─── SETUP COM & DXCAM ───────────────────────────────────────────────────────
comtypes.CoInitialize()

def cleanup():
    camera.stop()
    comtypes.CoUninitialize()

atexit.register(cleanup)

camera = dxcam.create(output_color="BGR")
camera.start(
    target_fps=60,
    region=REGION,
    video_mode=True
)

# Precompute slices, masks, queue, stack, event
slice_p1    = (slice(Y_HEALTH, Y_HEALTH+1), slice(X1_P1, X2_P1), slice(None))
slice_p2    = (slice(Y_HEALTH, Y_HEALTH+1), slice(X1_P2, X2_P2), slice(None))
mask1       = np.empty((1, LEN_P1), dtype=np.uint8)
mask2       = np.empty((1, LEN_P2), dtype=np.uint8)
frame_q     = Queue(maxsize=16)
frame_stack = deque(maxlen=FRAME_STACK)
stop_event  = threading.Event()
results     = []
screenshots = []
total_steps = 0
prev_action = None

# ─── PRODUCER THREAD ─────────────────────────────────────────────────────────
def producer():
    start = time.perf_counter()
    try:
        while not stop_event.is_set():
            frame = camera.get_latest_frame()
            if frame is not None:
                ts = time.perf_counter() - start
                frame_q.put((frame.copy(), ts))
    except Exception as e:
        print("Producer error:", e)
        stop_event.set()

# ─── CONSUMER THREAD ─────────────────────────────────────────────────────────
def consumer():
    global total_steps, prev_action
    try:
        while not stop_event.is_set() or not frame_q.empty():
            frame, ts = frame_q.get()
            # collect screenshots
            if len(screenshots) < MAX_SCREENSHOTS:
                screenshots.append(frame.copy())
            # health detection
            strip1 = frame[slice_p1]
            cv2.inRange(strip1, LOWER_BGR, UPPER_BGR, mask1)
            pct1 = cv2.countNonZero(mask1) / LEN_P1 * 100.0
            strip2 = frame[slice_p2]
            cv2.inRange(strip2, LOWER_BGR, UPPER_BGR, mask2)
            pct2 = cv2.countNonZero(mask2) / LEN_P2 * 100.0
            results.append((f"{ts:.3f}", f"{pct1:.1f}", f"{pct2:.1f}"))
            # preprocess and stack
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            img  = cv2.resize(gray, CNN_SIZE, interpolation=cv2.INTER_NEAREST)
            frame_stack.append(img.astype(np.float32)/255.0)
            # decision step
            if len(frame_stack) == FRAME_STACK:
                total_steps += 1
                eps = EPS_END + (EPS_START - EPS_END) * np.exp(-total_steps / EPS_DECAY)
                state = torch.tensor([frame_stack], device=DEVICE)
                with torch.no_grad():
                    q = net(state)
                if random.random() < eps:
                    action_idx = random.randrange(NUM_ACTIONS)
                else:
                    action_idx = int(q.argmax(dim=1).item())
                action = ACTIONS[action_idx]
                # write every frame to actions.txt
                with open(ACTIONS_FILE, 'w') as f:
                    f.write(action)
                print(f"[{ts:.3f}s] P1={pct1:.1f}% P2={pct2:.1f}% EPS={eps:.3f} -> {action}")
                frame_stack.clear()
            frame_q.task_done()
    except Exception as e:
        print("Consumer error:", e)
        stop_event.set()

# ─── MAIN ────────────────────────────────────────────────────────────────────
t_p = threading.Thread(target=producer, daemon=True)
t_c = threading.Thread(target=consumer, daemon=True)
t_p.start()
t_c.start()
print(f"Running for {DURATION}s...")
time.sleep(DURATION)
stop_event.set()
t_p.join()
frame_q.join()
t_c.join()

# ─── SAVE SCREENSHOTS & WRITE CSV ─────────────────────────────────────────────
os.makedirs("screenshots", exist_ok=True)
for i, frm in enumerate(screenshots):
    rgb = frm[..., ::-1]
    Image.fromarray(rgb).save(f"screenshots/frame_{i:04d}.png")

camera.stop()
with open(LOG_CSV, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["time_s", "p1_pct", "p2_pct"])
    writer.writerows(results)
print(f"Done: {len(results)} frames, {len(screenshots)} screenshots → {LOG_CSV}")

