#!/usr/bin/env python3
import os
import time
import atexit

import comtypes
import dxcam
import numpy as np
from PIL import Image

# ─── 1) CONFIGURE YOUR CROP REGION & RESOLUTION ───────────────────────────────
# Crop on your primary screen:
left, top = 0, 0
width, height = 680, 540
region = (left, top, left + width, top + height)

# How long to capture for (seconds)
duration_s = 1.0

# OUTPUT RESOLUTION:
#   - None → keep full (680×540)
#   - (168, 168) → 4× larger than 84×84
#   - (224, 224) → common CNN size
#   - etc.
target_size = None  # e.g. None, or (168,168), or (224,224)

# ─── 2) INIT COM + DXCAM ─────────────────────────────────────────────────────
comtypes.CoInitialize()
def _cleanup():
    camera.stop()
    comtypes.CoUninitialize()
atexit.register(_cleanup)

# grayscale + only on real screen changes
camera = dxcam.create(output_color="GRAY")
camera.start(
    target_fps=0,       # C++ pumps as fast as possible
    region=region,      # only copy this sub-rectangle
    video_mode=False    # only enqueue on real desktop changes
)

# ─── 3) CAPTURE + PREPROCESS LOOP ─────────────────────────────────────────────
frames = []
start = time.perf_counter()
while time.perf_counter() - start < duration_s:
    f = camera.get_latest_frame()      # None until desktop actually changes
    if f is None:
        continue

    gray = f[:, :, 0]                  # (540,680)
    img  = Image.fromarray(gray)       # PIL Image

    # resize if desired
    if target_size is not None:
        img = img.resize(target_size)

    arr = np.array(img, dtype=np.uint8) # shape depends on target_size or region
    frames.append(arr)

# ─── 4) TEARDOWN & SAVE ───────────────────────────────────────────────────────
camera.stop()

os.makedirs("screenshots", exist_ok=True)
for i, arr in enumerate(frames):
    Image.fromarray(arr).save(f"screenshots/frame_{i:04d}.png")

total = time.perf_counter() - start
print(f"Captured {len(frames)} fresh frames in {total:.3f}s → ./screenshots/")
print(f"Each frame is {arr.shape[1]}×{arr.shape[0]} pixels.")
