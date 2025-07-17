import os
import time
import atexit

import comtypes
import dxcam
import numpy as np
from PIL import Image

# — CONFIGURE YOUR REGION & DURATION —
left, top = 0, 0
width, height = 680, 540
region = (left, top, left + width, top + height)  # (l, t, r, b)
duration_s = 1.0
max_frames = 2000  # adjust upward if you expect >2000 grabs in 1 s

# — 1) Initialize COM & schedule cleanup —
comtypes.CoInitialize()
def _cleanup():
    camera.stop()
    comtypes.CoUninitialize()
atexit.register(_cleanup)

# — 2) Create a grayscale camera and start threaded capture —
camera = dxcam.create(output_color="GRAY")
camera.start(
    target_fps=0,       # 0 = as fast as possible
    region=region,
    video_mode=True     # duplicate last frame when static
)

# — 3) Pre-allocate buffer for in-place copies —
# shape = (n_frames, height, width), dtype=uint8
buffer = np.empty((max_frames, height, width), dtype=np.uint8)

# — 4) Drain frames for the specified duration —
count = 0
start = time.perf_counter()
while time.perf_counter() - start < duration_s and count < max_frames:
    frame = camera.get_latest_frame()       # returns (h, w, 1) uint8 or None
    if frame is not None:
        # squeeze out the last dim and copy into our buffer
        buffer[count] = frame[:, :, 0]
        count += 1

# — 5) Stop capture & save out to PNGs —
camera.stop()
os.makedirs("screenshots", exist_ok=True)
for i in range(count):
    Image.fromarray(buffer[i]).save(f"screenshots/frame_{i:04d}.png")

print(f"Captured and saved {count} frames in {duration_s:.1f} s → ./screenshots/")
