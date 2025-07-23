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

# Wider regions (106×1 pixels each) for black‐pixel proportion + ranges
AREA_RECTS = [
    ("P1_R1_area", 71, 480, 177, 481),
    ("P2_R2_area", 469, 480, 575, 481),
]

# Single‐pixel ROIs for exact BGR → state classification
PIXEL_RECTS = [
    ("P1_R1_pixel", 71, 475, 72, 476),
    ("P2_R2_pixel", 520, 475, 521, 476),
]

# Single‐pixel states
STATE_MAP = {
    (200, 200, 200): "can transform",
    ( 48,  48, 248): "transformed",
    (240, 128,   0): "cannot transform",
}

# How we define “black” in the wider regions
BLACK_BGR = np.array([0, 0, 8], dtype=np.uint8)

TARGET_FPS       = 1
DISPLAY_INTERVAL = 0.1  # seconds

# ─── SETUP SCREEN CAPTURE ───────────────────────────────────────────────────
comtypes.CoInitialize()
camera = dxcam.create(output_color="BGR")
camera.start(target_fps=TARGET_FPS, region=REGION, video_mode=True)
atexit.register(lambda: (camera.stop(), comtypes.CoUninitialize()))

printed_debug = False
AREA_COLOR  = (0, 255, 0)
PIXEL_COLOR = (255, 0, 0)

try:
    while True:
        frame = camera.get_latest_frame()
        if frame is None:
            time.sleep(0.01)
            continue

        # 1) Analyze each wider region
        for label, x1, y1, x2, y2 in AREA_RECTS:
            roi = frame[y1:y2, x1:x2]
            if roi.size == 0:
                print(f"[WARN] {label} empty ROI")
                continue

            # mask exact “black” pixels
            mask        = cv2.inRange(roi, BLACK_BGR, BLACK_BGR)
            black_count = int(cv2.countNonZero(mask))
            total_px    = roi.shape[0] * roi.shape[1]
            pct_black   = black_count / total_px * 100.0

            # channel ranges
            B, G, R     = roi[:,:,0], roi[:,:,1], roi[:,:,2]
            min_b, max_b = int(B.min()), int(B.max())
            min_g, max_g = int(G.min()), int(G.max())
            min_r, max_r = int(R.min()), int(R.max())

            print(f"{label}:")
            print(f"  Black pixels: {black_count}/{total_px} → {pct_black:.1f}%")
            print(f"  B range = {min_b}–{max_b}")
            print(f"  G range = {min_g}–{max_g}")
            print(f"  R range = {min_r}–{max_r}\n")

        # 2) Classify each single‐pixel ROI
        for label, x1, y1, x2, y2 in PIXEL_RECTS:
            # we know x2=x1+1, y2=y1+1
            b, g, r = frame[y1, x1]
            pixel   = (int(b), int(g), int(r))
            state   = STATE_MAP.get(pixel, "unknown")
            print(f"{label}: pixel BGR = {pixel} → {state}")

        # draw debug overlays once
        if not printed_debug:
            dbg = frame.copy()
            for _, x1, y1, x2, y2 in AREA_RECTS:
                cv2.rectangle(dbg, (x1,y1), (x2,y2), AREA_COLOR, 1)
            for _, x1, y1, x2, y2 in PIXEL_RECTS:
                cv2.rectangle(dbg, (x1,y1), (x2,y2), PIXEL_COLOR, 1)
            os.makedirs("debug", exist_ok=True)
            cv2.imwrite("debug/combined_debug.png", dbg)
            print("Wrote debug/combined_debug.png with overlays")
            printed_debug = True

        time.sleep(DISPLAY_INTERVAL)

except KeyboardInterrupt:
    print("Exiting…")
    camera.stop()
