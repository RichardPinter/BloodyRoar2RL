import os
import time
import atexit

import comtypes
import dxcam
from PIL import Image

# — CONFIGURE YOUR CROP REGION HERE —
# left, top = top-left corner of your crop
# width, height = size of the crop
left, top = 0, 0
width, height = 680, 540
region = (left, top, left + width, top + height)  # (l, t, r, b)

# — 1) Initialize COM once and schedule cleanup —
comtypes.CoInitialize()
def _cleanup():
    camera.stop()
    comtypes.CoUninitialize()
atexit.register(_cleanup)

# — 2) Create your DXCam grabber and start the background thread —
camera = dxcam.create()
camera.start(
    target_fps=0,       # as fast as possible
    region=region,      # only copy this sub-rectangle
    video_mode=False    # only enqueue on real screen-changes
)

# — 3) Collect “latest” cropped frames for 1 second —
frames = []
start = time.perf_counter()
while time.perf_counter() - start < 1.0:
    frame = camera.get_latest_frame()   # None until a new cropped frame arrives
    if frame is not None:
        frames.append(frame)

# — 4) Stop capture and save out to PNGs —
camera.stop()
os.makedirs("screenshots", exist_ok=True)
for idx, frame in enumerate(frames):
    # frame is now an (height×width×3) RGB array of your crop
    Image.fromarray(frame).save(os.path.join("screenshots", f"crop_{idx:04d}.png"))

print(f"Captured {len(frames)} distinct cropped frames in 1.0 s → ./screenshots/")
