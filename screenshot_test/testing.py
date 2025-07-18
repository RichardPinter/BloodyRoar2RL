#!/usr/bin/env python3
import os
import time
import atexit

import comtypes
import dxcam
import numpy as np
import cv2

# ─── CONFIG ────────────────────────────────────────────────────────────────
REGION      = (0, 0, 512, 338)            # BizHawk window at “size 2”
Y_HEALTH    = 116                          # Y-row of the health bars
X1_P1, X2_P1 = 73, 293                     # P1: 220 px wide
X1_P2, X2_P2 = 355, 57                    # P2: also 220 px wide
LEN_P1      = X2_P1 - X1_P1
LEN_P2      = X2_P2 - X1_P2

LOWER_BGR   = np.array([0,150,180], dtype=np.uint8)
UPPER_BGR   = np.array([30,175,220], dtype=np.uint8)

# ─── SETUP SCREEN CAPTURE ───────────────────────────────────────────────────
comtypes.CoInitialize()
camera = dxcam.create(output_color="BGR")
camera.start(target_fps=30, region=REGION, video_mode=True)
atexit.register(lambda: (camera.stop(), comtypes.CoUninitialize()))

# ─── MAIN LOOP ──────────────────────────────────────────────────────────────
printed_once = False

try:
    while True:
        frame = camera.get_latest_frame()
        if frame is None:
            time.sleep(0.01)
            continue

        # Crop the 1-pixel-high strip for each player
        strip1 = frame[Y_HEALTH:Y_HEALTH+1, X1_P1:X2_P1]
        strip2 = frame[Y_HEALTH:Y_HEALTH+1, X1_P2:X2_P2]

        # Mask & percentage
        m1 = cv2.inRange(strip1, LOWER_BGR, UPPER_BGR)
        m2 = cv2.inRange(strip2, LOWER_BGR, UPPER_BGR)
        pct1 = (cv2.countNonZero(m1) / LEN_P1) * 100.0
        pct2 = (cv2.countNonZero(m2) / LEN_P2) * 100.0

        # Print to console
        print(f"P1 health: {pct1:5.1f}%   P2 health: {pct2:5.1f}%")

        # Once—draw boxes and save a debug image
        if not printed_once:
            debug = frame.copy()
            # red box around P1 bar
            cv2.rectangle(debug,
                          (X1_P1, Y_HEALTH-2),
                          (X2_P1, Y_HEALTH+2),
                          (0,0,255), 2)
            # blue box around P2 bar
            cv2.rectangle(debug,
                          (X1_P2, Y_HEALTH-2),
                          (X2_P2, Y_HEALTH+2),
                          (255,0,0), 2)
            os.makedirs("debug", exist_ok=True)
            cv2.imwrite("debug/health_debug.png", debug)
            print("Wrote debug/health_debug.png with overlays")
            printed_once = True

        # slow down so you can read
        time.sleep(0.1)

except KeyboardInterrupt:
    print("Exiting…")
    camera.stop()
