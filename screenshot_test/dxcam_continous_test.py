import os
import time
import atexit

import comtypes
import dxcam
from PIL import Image

# — 1) Initialize COM once and schedule cleanup — 
comtypes.CoInitialize()
def _cleanup():
    camera.stop()
    comtypes.CoUninitialize()
atexit.register(_cleanup)

# — 2) Create your DXCam grabber on the primary display — 
#    (by default, dxcam.create() uses device_idx=0, output_idx=0)
camera = dxcam.create()

# — 3) Burst-capture for 1 second into a list — 
frames = []
start = time.perf_counter()
while time.perf_counter() - start < 1.0:
    frame = camera.grab()           # H×W×3 RGB numpy array
    if frame is not None:
        frames.append(frame)

# — 4) Save them all as PNGs — 
output_dir = "screenshots"
os.makedirs(output_dir, exist_ok=True)

for idx, frame in enumerate(frames):
    img = Image.fromarray(frame)
    img.save(os.path.join(output_dir, f"frame_{idx:04d}.png"))

print(f"Captured {len(frames)} frames in 1 second → ./{output_dir}/")
