import time
import os
import atexit
import comtypes
import dxcam
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
from PIL import Image

# ─── 1) DEFINE YOUR DQN NETWORK ───────────────────────────────────────────────
class DQNNet(nn.Module):
    def __init__(self, in_channels: int = 4, num_actions: int = 6):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        conv_out = 64 * 7 * 7
        self.fc1  = nn.Linear(conv_out, 512)
        self.out  = nn.Linear(512, num_actions)
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.flatten(start_dim=1)
        x = F.relu(self.fc1(x))
        return self.out(x)

# ─── 2) SETUP DEVICE, COM, CAMERA ─────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# region you want to capture
left, top = 0, 0
width, height = 680, 540
region = (left, top, left + width, top + height)

# init COM once
comtypes.CoInitialize()
def cleanup():
    camera.stop()
    comtypes.CoUninitialize()
atexit.register(cleanup)

# create a gray, change-driven capture
camera = dxcam.create(output_color="GRAY")
camera.start(
    target_fps=0,       # pump as fast as possible
    region=region,
    video_mode=False    # only new frames when screen changes  
)

# ─── 3) INSTANTIATE YOUR AGENT ────────────────────────────────────────────────
net = DQNNet(in_channels=4, num_actions=6).to(device).eval()
# warm up
dummy = torch.zeros(1, 4, height, width, device=device)
with torch.no_grad():
    for _ in range(5):
        _ = net(dummy)

# ─── 4) MAIN LOOP: CAPTURE → STACK → INFER → ACTION REPEAT ───────────────────
stack = deque(maxlen=4)    # holds last 4 frames
action_repeat = 4          # repeat each chosen action this many frames

for step in range(50):     # e.g. run for 50 decisions
    # 4a) collect 4 *new* frames
    while len(stack) < 4:
        frame = camera.get_latest_frame()  # blocks until desktop actually changed
        if frame is not None:
            stack.append(frame[:, :, 0])   # squeeze (h,w,1) → (h,w)

    # 4b) build state tensor (1,4,h,w)
    state = np.stack(stack, axis=0)            # shape = (4, h, w)
    tensor = torch.from_numpy(state)           # uint8 → torch.LongTensor
    tensor = tensor.unsqueeze(0).float().to(device) / 255.0

    # 4c) forward pass to get Q-values
    with torch.no_grad():
        q_vals = net(tensor)                   # shape = (1, num_actions)
    action = int(q_vals.argmax(dim=1).item())
    print(f"[Decide #{step}] Action → {action}")

    # 4d) repeat that action for the next N frames
    for _ in range(action_repeat):
        # here you’d call your env.step(action) or send the input to BizHawk
        # … and *then* grab one new frame so the desktop changes for get_latest_frame()
        # e.g.: env.step(action); time.sleep(1/60)
        _ = camera.get_latest_frame()         # drain the duplicate/new buffer

    # clear stack to force 4 fresh frames next decision
    stack.clear()

# ─── 5) CLEANUP ────────────────────────────────────────────────────────────────
camera.stop()
print("Done.")
