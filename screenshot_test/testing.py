#!/usr/bin/env python3
import os
import time
import atexit

import comtypes
import dxcam
import numpy as np
import cv2

# ─── CONFIG ────────────────────────────────────────────────────────────────
REGION = (0, 0, 512, 338)   # (left, top, width, height) for dxcam

# Rectangles: (label, x1, y1, x2, y2)
# Make sure x2 > x1 and y2 > y1 for each.
RECTS = [
    ("P1_R1", 270, 135, 278, 140),
    ("P1_R2", 245, 135, 253, 140),
    ("P2_R1", 373, 135, 381, 140),
    ("P2_R2", 396, 135, 404, 140),
]

# Color range (BGR). Tweak as needed for your target color.
LOWER_BGR = np.array([20, 20, 150], dtype=np.uint8)
UPPER_BGR = np.array([30, 30, 220], dtype=np.uint8)

TARGET_FPS = 30
DISPLAY_INTERVAL = 0.1  # seconds between prints

# ─── SETUP SCREEN CAPTURE ───────────────────────────────────────────────────
comtypes.CoInitialize()
camera = dxcam.create(output_color="BGR")
camera.start(target_fps=TARGET_FPS, region=REGION, video_mode=True)
atexit.register(lambda: (camera.stop(), comtypes.CoUninitialize()))

# ─── UTILITIES ─────────────────────────────────────────────────────────────-
def percent_in_range(frame, rect):
    label, x1, y1, x2, y2 = rect
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"{label} invalid rectangle (non-positive size): {rect[1:]}")
    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        raise ValueError(f"{label} ROI empty: {rect[1:]}")
    # Optional denoise to reduce flicker:
    # roi = cv2.GaussianBlur(roi, (3,3), 0)
    mask = cv2.inRange(roi, LOWER_BGR, UPPER_BGR)

    pct = (cv2.countNonZero(mask) / mask.size) * 100.0
    # print(mask, pct)
    return pct

BOX_COLORS = [
    (0, 0, 255),
    (255, 0, 0),
    (0, 255, 0),
    (0, 255, 255),
]

printed_debug = False

# ─── MAIN LOOP ──────────────────────────────────────────────────────────────
try:
    while True:
        frame = camera.get_latest_frame()
        if frame is None:
            time.sleep(0.01)
            continue

        readings = []
        for rect in RECTS:
            print(rect)
            try:
                pct = percent_in_range(frame, rect)
                print(pct)
            except ValueError as e:
                print("[WARN]", e)
                pct = float("nan")
            readings.append((rect[0], pct))

        # Create labeled debug image once
        if not printed_debug:
            debug = frame.copy()
            for (i, rect) in enumerate(RECTS):
                label, x1, y1, x2, y2 = rect
                color = BOX_COLORS[i % len(BOX_COLORS)]
                cv2.rectangle(debug, (x1, y1), (x2, y2), color, 1)
                cv2.putText(debug, label, (x1, y1 - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
            os.makedirs("debug", exist_ok=True)
            cv2.imwrite("debug/rects_debug.png", debug)
            print("Wrote debug/rects_debug.png with overlays")
            printed_debug = True

        time.sleep(DISPLAY_INTERVAL)

except KeyboardInterrupt:
    print("Exiting…")
    camera.stop()
