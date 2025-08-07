#!/usr/bin/env python3
import os
import time
import comtypes
import dxcam
import numpy as np
import cv2

# ─── CONFIG ────────────────────────────────────────────────────────────────
# Screen region to capture (x, y, width, height)
REGION = (0, 0, 680, 540)
# Round indicator ROIs (using your values)
ROUND_INDICATORS = {
    'p1_r1': (270, 135, 278, 140),
    'p1_r2': (245, 135, 253, 140),
    'p2_r1': (373, 135, 381, 140),
    'p2_r2': (396, 135, 404, 140),
}
# Fraction of red pixels to consider "on"
RED_THRESHOLD = 0.25  # tune between 0.1–0.5 as needed

# HSV ranges for red (two hue segments)
LOW1 = np.array([0, 70, 50], dtype=np.uint8)
HIGH1 = np.array([10, 255, 255], dtype=np.uint8)
LOW2 = np.array([170, 70, 50], dtype=np.uint8)
HIGH2 = np.array([180, 255, 255], dtype=np.uint8)


def is_indicator_red(frame: np.ndarray, coord: tuple) -> tuple:
    """Return (red_fraction, is_active) for one ROI."""
    x1, y1, x2, y2 = coord
    roi_bgr = frame[y1:y2, x1:x2]
    if roi_bgr.size == 0:
        return 0.0, False

    roi_hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    mask1 = cv2.inRange(roi_hsv, LOW1, HIGH1)
    mask2 = cv2.inRange(roi_hsv, LOW2, HIGH2)
    mask = cv2.bitwise_or(mask1, mask2)

    red_frac = mask.mean() / 255.0
    return red_frac, red_frac > RED_THRESHOLD


def notify(name: str):
    """Beep and print a notification when an indicator turns red."""
    print(f"\a>> {name} just turned RED! <<")


def clear_console():
    """Clear terminal output for a single-line status update."""
    os.system('cls' if os.name == 'nt' else 'clear')


if __name__ == '__main__':
    comtypes.CoInitialize()
    camera = dxcam.create(output_color='BGR')
    camera.start(target_fps=60, region=REGION, video_mode=True)

    prev_states = {name: False for name in ROUND_INDICATORS}
    print("Press Ctrl+C to stop")

    try:
        while True:
            frame = camera.get_latest_frame()
            if frame is None:
                time.sleep(0.01)
                continue

            statuses = []
            clear_console()
            for name, coord in ROUND_INDICATORS.items():
                frac, active = is_indicator_red(frame, coord)
                if active and not prev_states[name]:
                    notify(name)
                prev_states[name] = active
                statuses.append(f"{name}: {frac*100:5.1f}% {'ON ' if active else 'off'}")

            # Print all statuses on one line
            print('   '.join(statuses))
            time.sleep(0.1)

    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        camera.stop()
        comtypes.CoUninitialize()
        print("Done.")