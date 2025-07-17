#!/usr/bin/env python3
import time
import threading
import atexit
from collections import deque

import comtypes, dxcam
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F

# ─── CONFIG ────────────────────────────────────────────────────────────────
left, top = 0, 0
w, h      = 680, 540
region    = (left, top, left + w, top + h)

stack_size  = 4
cnn_size    = (84, 84)
n_actions   = 6
device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_decisions = 5

# ─── SIMPLE DQNNet ─────────────────────────────────────────────────────────
class DQNNet(nn.Module):
    def __init__(self, in_ch, n_a):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, 32, 8, 4)
        self.conv2 = nn.Conv2d(32,   64, 4, 2)
        self.conv3 = nn.Conv2d(64,   64, 3, 1)
        out_sz    = 64 * 7 * 7
        self.fc1  = nn.Linear(out_sz, 512)
        self.out  = nn.Linear(512, n_a)
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.flatten(1)
        x = F.relu(self.fc1(x))
        return self.out(x)

net = DQNNet(stack_size, n_actions).to(device).eval()
# warm‐up
with torch.no_grad():
    dummy = torch.zeros(1, stack_size, *cnn_size, device=device)
    for _ in range(3): net(dummy)

print(f"→ Device: {device}")
print(f"→ Capturing region {region} in GRAY, stacking {stack_size} frames\n")

# ─── CAPTURE THREAD ────────────────────────────────────────────────────────
comtypes.CoInitialize()
def cleanup():
    camera.stop()
    comtypes.CoUninitialize()
atexit.register(cleanup)

camera = dxcam.create(output_color="GRAY")
camera.start(target_fps=60, region=region, video_mode=True)

ring = deque(maxlen=stack_size)
lock = threading.Lock()
new_frame = threading.Event()
running   = True

def capture_loop():
    while running:
        f = camera.get_latest_frame()
        if f is not None:
            gray = f[...,0]
            with lock:
                ring.append(gray)
                # debug: print buffer length
                print(f"[CAPTURE] Buffer size now {len(ring)}")
            new_frame.set()

t = threading.Thread(target=capture_loop, daemon=True)
t.start()

# ─── LOGIC / INFERENCE LOOP ───────────────────────────────────────────────
for step in range(1, n_decisions+1):
    # 1) wait until we've captured at least one new frame
    new_frame.wait(); new_frame.clear()

    # 2) snapshot exactly stack_size frames
    with lock:
        if len(ring) < stack_size:
            print(f"[WARN] only {len(ring)} frames in buffer, skipping")
            continue
        frames = list(ring)
    print(f"[STEP {step}] Pulled {len(frames)} frames for inference")

    # 3) preprocess & stack
    proc = []
    for i, img in enumerate(frames):
        pil = Image.fromarray(img).resize(cnn_size).convert("L")
        arr = np.array(pil, np.float32)/255.0
        proc.append(arr)
        print(f"  • Prepped frame {i+1} shape={arr.shape} min={arr.min():.3f} max={arr.max():.3f}")
    state = np.stack(proc,0)[None,...]  # shape = (1,4,84,84)
    tensor = torch.from_numpy(state).to(device)
    print(f"  • State tensor shape: {tuple(tensor.shape)}")

    # 4) inference
    with torch.no_grad():
        qs = net(tensor)
    qs = qs.cpu().numpy().flatten()
    print(f"  • Q-values: {np.round(qs,3)}  → argmax = {qs.argmax()}")

print("\n✅ Pipeline check complete—every stage logged above.")

# ─── TEARDOWN ─────────────────────────────────────────────────────────────
running = False
t.join()
camera.stop()
