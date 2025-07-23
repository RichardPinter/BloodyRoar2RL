#!/usr/bin/env python3
import os
import time
import atexit

import comtypes
import dxcam
import numpy as np
import cv2

# ─── CONFIG ────────────────────────────────────────────────────────────────
REGION = (0, 0, 624, 548)   # (left, top, width, height)

# Rectangles: (label, x1, y1, x2, y2)
# Here each ROI is 1×1, so x2 = x1+1 and y2 = y1+1
RECTS = [
    ("P1_R1", 71, 475, 72, 476),
    ("P2_R2", 520, 475, 521, 476),
]

TARGET_FPS = 1
DISPLAY_INTERVAL = 0.1  # seconds between prints

# Known single‑pixel BGR states
STATE_MAP = {
    (200, 200, 200): "can transform",
    ( 48,  48, 248): "transformed",
    (240, 128,   0): "cannot transform",
}

# ─── SETUP SCREEN CAPTURE ───────────────────────────────────────────────────
comtypes.CoInitialize()
camera = dxcam.create(output_color="BGR")
camera.start(target_fps=TARGET_FPS, region=REGION, video_mode=True)
atexit.register(lambda: (camera.stop(), comtypes.CoUninitialize()))

# ─── MAIN LOOP ──────────────────────────────────────────────────────────────
printed_debug = False
BOX_COLORS = [(0, 0, 255), (255, 0, 0)]

try:
    while True:
        frame = camera.get_latest_frame()
        if frame is None:
            time.sleep(0.01)
            continue

        for label, x1, y1, x2, y2 in RECTS:
            # validate rectangle coords
            if x2 != x1 + 1 or y2 != y1 + 1:
                print(f"[WARN] {label} ROI is not 1×1: {(x1, y1, x2, y2)}")
                continue

            h, w = frame.shape[:2]
            if not (0 <= x1 < w and 0 <= y1 < h):
                print(f"[WARN] {label} out of bounds: {(x1, y1)}")
                continue

            # read the single pixel BGR
            b, g, r = frame[y1, x1]
            pixel = (int(b), int(g), int(r))
            state = STATE_MAP.get(pixel, "unknown")

            print(f"{label} pixel BGR = {pixel} → {state}")

        # draw debug rectangles once
        if not printed_debug:
            debug = frame.copy()
            for i, (lbl, x1, y1, x2, y2) in enumerate(RECTS):
                color = BOX_COLORS[i % len(BOX_COLORS)]
                cv2.rectangle(debug, (x1, y1), (x2, y2), color, 1)
                cv2.putText(debug, lbl, (x1, y1 - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
            os.makedirs("debug", exist_ok=True)
            cv2.imwrite("debug/rects_debug.png", debug)
            print("Wrote debug/rects_debug.png with overlays")
            printed_debug = True

        time.sleep(DISPLAY_INTERVAL)

except KeyboardInterrupt:
    print("Exiting…")
    camera.stop()
