#!/usr/bin/env python3
import os
import time
import comtypes
import dxcam
import numpy as np
import cv2
from datetime import datetime
from collections import deque

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

# HSV range for blue
BLUE_LOW = np.array([100, 50, 50], dtype=np.uint8)
BLUE_HIGH = np.array([130, 255, 255], dtype=np.uint8)

# Brightness threshold to detect if anything is there
BRIGHTNESS_THRESHOLD = 20  # Below this = truly absent/black (tune this if blue is detected as absent)

# If blue indicators are still showing as ABSENT, try:
# 1. Lower BRIGHTNESS_THRESHOLD to 10-20
# 2. Adjust BLUE_LOW/BLUE_HIGH HSV ranges
# 3. Check if indicators have a different color when empty

# State definitions
class IndicatorState:
    ABSENT = "absent"      # Dark/black (not visible)
    BLUE = "blue"          # Blue indicator (empty)
    RED = "red"            # Red indicator (won)

def is_indicator_red(frame: np.ndarray, coord: tuple) -> tuple:
    """Return (red_fraction, state, debug_info) for one ROI."""
    x1, y1, x2, y2 = coord
    roi_bgr = frame[y1:y2, x1:x2]
    if roi_bgr.size == 0:
        return 0.0, IndicatorState.ABSENT, {}
    
    # Check overall brightness first
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    avg_brightness = gray.mean()
    
    if avg_brightness < BRIGHTNESS_THRESHOLD:
        return 0.0, IndicatorState.ABSENT, {"brightness": avg_brightness}
    
    # Convert to HSV for color detection
    roi_hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    
    # Check for red
    mask1 = cv2.inRange(roi_hsv, LOW1, HIGH1)
    mask2 = cv2.inRange(roi_hsv, LOW2, HIGH2)
    red_mask = cv2.bitwise_or(mask1, mask2)
    red_frac = red_mask.mean() / 255.0
    
    # Check for blue
    blue_mask = cv2.inRange(roi_hsv, BLUE_LOW, BLUE_HIGH)
    blue_frac = blue_mask.mean() / 255.0
    
    debug_info = {
        "brightness": avg_brightness,
        "blue_pct": blue_frac * 100,
        "red_pct": red_frac * 100
    }
    
    if red_frac > RED_THRESHOLD:
        return red_frac, IndicatorState.RED, debug_info
    
    if blue_frac > 0.1:  # If more than 10% blue pixels
        return red_frac, IndicatorState.BLUE, debug_info
    
    # If bright but not red or blue, still consider it blue (empty indicator)
    return red_frac, IndicatorState.BLUE, debug_info

def notify(message: str, beep: bool = True):
    """Print a notification with timestamp."""
    timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    if beep:
        print(f"\a[{timestamp}] >> {message} <<")
    else:
        print(f"[{timestamp}] {message}")

def clear_console():
    """Clear terminal output for a single-line status update."""
    os.system('cls' if os.name == 'nt' else 'clear')

def check_new_match_pattern(history: dict) -> bool:
    """Check if we've seen the pattern: RED → ABSENT → BLUE for all indicators."""
    # Need at least 3 states in history
    for name, states in history.items():
        if len(states) < 3:
            return False
    
    # Check if all indicators went through RED → ABSENT → BLUE
    all_match_pattern = True
    for name, states in history.items():
        # Look for the pattern in recent history
        pattern_found = False
        for i in range(len(states) - 2):
            if (states[i] == IndicatorState.RED and 
                states[i+1] == IndicatorState.ABSENT and 
                states[i+2] == IndicatorState.BLUE):
                pattern_found = True
                break
        if not pattern_found:
            all_match_pattern = False
            break
    
    return all_match_pattern

if __name__ == '__main__':
    comtypes.CoInitialize()
    camera = dxcam.create(output_color='BGR')
    camera.start(target_fps=60, region=REGION, video_mode=True)
    
    prev_states = {name: IndicatorState.ABSENT for name in ROUND_INDICATORS}
    state_history = {name: deque(maxlen=10) for name in ROUND_INDICATORS}  # Keep last 10 states
    
    # Track match detection
    match_pattern_detected = False
    last_pattern_time = 0
    
    print("Press Ctrl+C to stop")
    print("="*80)
    
    try:
        while True:
            frame = camera.get_latest_frame()
            if frame is None:
                time.sleep(0.01)
                continue
            
            statuses = []
            state_changes = []
            
            # Check each indicator
            for name, coord in ROUND_INDICATORS.items():
                frac, state, debug_info = is_indicator_red(frame, coord)
                
                # Track state changes
                if state != prev_states[name]:
                    change_msg = f"{name}: {prev_states[name]} → {state}"
                    if debug_info:
                        change_msg += f" (bright:{debug_info.get('brightness', 0):.1f}, blue:{debug_info.get('blue_pct', 0):.1f}%, red:{debug_info.get('red_pct', 0):.1f}%)"
                    state_changes.append(change_msg)
                    state_history[name].append(state)
                
                prev_states[name] = state
                
                # Format status string
                state_str = state.upper().ljust(6)
                statuses.append(f"{name}: {frac*100:5.1f}% {state_str}")
            
            # Print current status on one line
            clear_console()
            print('   '.join(statuses))
            
            # Print state changes below (these persist)
            for change in state_changes:
                notify(change, beep=False)
            
            # Check for new match pattern
            if check_new_match_pattern(state_history):
                current_time = time.time()
                # Only notify if we haven't detected this pattern in the last 5 seconds
                if current_time - last_pattern_time > 5:
                    notify("NEW MATCH PATTERN DETECTED! All indicators: RED → ABSENT → BLUE", beep=True)
                    last_pattern_time = current_time
                    # Clear history to avoid repeated detections
                    for hist in state_history.values():
                        hist.clear()
            
            # Also check for impossible states
            p1_reds = sum(1 for name, state in prev_states.items() if 'p1' in name and state == IndicatorState.RED)
            p2_reds = sum(1 for name, state in prev_states.items() if 'p2' in name and state == IndicatorState.RED)
            
            if p1_reds >= 2 and p2_reds >= 2:
                notify("⚠️ IMPOSSIBLE STATE: Both players have 2 rounds!", beep=True)
            
            time.sleep(0.1)
            
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        camera.stop()
        comtypes.CoUninitialize()
        print("Done.")