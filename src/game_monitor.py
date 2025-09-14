#!/usr/bin/env python3
"""
Game State Monitor - Real-time game state detection and display
Watches the game and reports health, rounds, state transitions, etc.
"""

import os
import time
import sys
import numpy as np
import cv2
import dxcam
from datetime import datetime
from collections import deque

# ─── CONFIGURATION ─────────────────────────────────────────────────────────
REGION = (0, 0, 624, 548)  # x, y, width, height
Y_HEALTH = 116
X1_P1, X2_P1 = 69, 288
X1_P2, X2_P2 = 350, 569
LEN_P1 = X2_P1 - X1_P1
LEN_P2 = X2_P2 - X1_P2

# Health bar color detection (specific yellow/orange color)
LOWER_BGR = np.array([0, 150, 180], dtype=np.uint8)
UPPER_BGR = np.array([30, 175, 220], dtype=np.uint8)

# Round indicator positions
ROUND_INDICATORS = {
    'p1_round1': (270, 135, 278, 140),
    'p1_round2': (245, 135, 253, 140),
    'p2_round1': (373, 135, 381, 140),
    'p2_round2': (396, 135, 404, 140),
}

# HSV ranges for red detection (exact same as rounds.py)
LOW1 = np.array([0, 70, 50], dtype=np.uint8)     # Lower red range
HIGH1 = np.array([10, 255, 255], dtype=np.uint8)
LOW2 = np.array([170, 70, 50], dtype=np.uint8)   # Upper red range  
HIGH2 = np.array([180, 255, 255], dtype=np.uint8)

# HSV range for blue indicators
BLUE_LOW = np.array([100, 50, 50], dtype=np.uint8)
BLUE_HIGH = np.array([130, 255, 255], dtype=np.uint8)

# Transform state detection
PIXEL_RECTS = [
    ("P1", 71, 475, 72, 476),
    ("P2", 520, 475, 521, 476),
]
STATE_MAP = {
    (200, 200, 200): "can transform",
    (48, 48, 248): "transformed",
    (240, 128, 0): "cannot transform",
}

# Black pixel detection for special moves
AREA_RECTS = [
    ("P1", 71, 480, 177, 481),
    ("P2", 469, 480, 575, 481),
]
BLACK_BGR = np.array([0, 0, 8], dtype=np.uint8)

# Thresholds
DEATH_THRESHOLD = 3.0
ZERO_THRESHOLD = 0.5
PRECISE_ZERO_THRESHOLD = 0.000001
ZERO_CONFIRMATION_TIME = 1.2

# ─── HELPER CLASSES ────────────────────────────────────────────────────────
class RoundState:
    """Simplified round state tracking for monitoring"""
    def __init__(self):
        self.confirmed = {'p1': 0, 'p2': 0}
        self.pending_health_winner = None
        # Removed indicator confirmation system - indicators now only used for red validation
    
    def count_won_rounds(self, states=None):
        """Count rounds won from confirmed counts (not indicators)"""
        # Now use the confirmed counts instead of indicator tracking
        return self.confirmed['p1'], self.confirmed['p2']
    
    def get_match_status(self, p1_rounds, p2_rounds):
        """Get match status message"""
        if p1_rounds == 2:
            return "P1 WINS MATCH 2-0!" if p2_rounds == 0 else "P1 WINS MATCH 2-1!"
        elif p2_rounds == 2:
            return "P2 WINS MATCH 2-0!" if p1_rounds == 0 else "P2 WINS MATCH 2-1!"
        elif p1_rounds == 1 and p2_rounds == 1:
            return "TIED 1-1! Next round wins match!"
        elif p1_rounds == 1:
            return "P1 leads 1-0, needs 1 more round"
        elif p2_rounds == 1:
            return "P2 leads 1-0, needs 1 more round"
        else:
            return "Match not started (0-0)"
    
    def reset(self):
        """Reset for new match"""
        self.confirmed = {'p1': 0, 'p2': 0}
        self.pending_health_winner = None
    
# Removed update_indicator_tracking method - no longer using standalone indicator confirmation

# ─── DETECTION FUNCTIONS ───────────────────────────────────────────────────
def detect_round_indicators(frame):
    """Detect round indicator states using exact same logic as rounds.py"""
    states = {}
    
    for name, (x1, y1, x2, y2) in ROUND_INDICATORS.items():
        region = frame[y1:y2, x1:x2]
        hsv_region = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
        
        # Red detection: wraps around in HSV, so we need two masks (same as rounds.py)
        red_mask1 = cv2.inRange(hsv_region, LOW1, HIGH1)
        red_mask2 = cv2.inRange(hsv_region, LOW2, HIGH2)
        red_mask = cv2.bitwise_or(red_mask1, red_mask2)
        
        # Blue detection
        blue_mask = cv2.inRange(hsv_region, BLUE_LOW, BLUE_HIGH)
        
        # Count pixels and calculate percentages
        total_pixels = region.shape[0] * region.shape[1]
        if total_pixels > 0:
            red_pixels = cv2.countNonZero(red_mask)
            red_pct = red_pixels / total_pixels * 100.0
            
            blue_pixels = cv2.countNonZero(blue_mask)
            blue_pct = blue_pixels / total_pixels * 100.0
        else:
            red_pct = 0.0
            blue_pct = 0.0
        
        # Determine state: prioritize red over blue with same thresholds as rounds.py
        if red_pct > 50.0:  # Same 50% threshold as rounds.py
            states[name] = 'red'
        elif blue_pct > 30.0:  # Blue detection threshold
            states[name] = 'blue'
        else:
            # FALLBACK: If not clearly blue and has some red, assume it's red (same as rounds.py)
            if blue_pct < 10.0 and red_pct > 20.0:
                states[name] = 'red'  # Likely red but weak signal
            else:
                states[name] = 'unknown'
    
    return states

def classify_transform_state(frame):
    """Detect transform states"""
    out = {}
    for player, x1, y1, x2, y2 in PIXEL_RECTS:
        b, g, r = frame[y1, x1]
        out[player] = STATE_MAP.get((int(b), int(g), int(r)), "unknown")
    return out

def compute_black_stats(frame):
    """Compute black pixel percentages"""
    pct_out = {}
    for player, x1, y1, x2, y2 in AREA_RECTS:
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            pct_out[player] = 0.0
            continue
        
        mask = cv2.inRange(roi, BLACK_BGR, BLACK_BGR)
        cnt = int(cv2.countNonZero(mask))
        total = roi.shape[0] * roi.shape[1]
        pct = cnt / total * 100.0
        pct_out[player] = pct
    
    return pct_out

def clear_console():
    """Clear console screen"""
    os.system('cls' if os.name == 'nt' else 'clear')

def format_health_bar(pct, width=20):
    """Create visual health bar"""
    filled = int(pct * width / 100)
    bar = '█' * filled + '░' * (width - filled)
    return bar

def format_indicator(state):
    """Format indicator state with symbol"""
    if state == 'blue':
        return '[ ]'
    elif state == 'red':
        return '[✓]'
    else:
        return '[?]'

# ─── MAIN MONITOR ──────────────────────────────────────────────────────────
def main():
    print("Initializing Game Monitor...")
    
    # Initialize camera
    camera = dxcam.create(region=REGION, output_idx=0, output_color="BGR")
    if not camera:
        print("Failed to initialize camera!")
        return
    
    camera.start(target_fps=60, video_mode=True)
    print("Camera started. Monitoring game state...")
    
    # Initialize tracking
    round_state = RoundState()
    game_state = "waiting_for_match"
    player_at_zero_since = {'p1': None, 'p2': None}
    last_update_time = time.time()
    frame_count = 0
    
    # Total wins tracking (across all matches)
    total_rounds_won = {'p1': 0, 'p2': 0}
    total_matches_won = {'p1': 0, 'p2': 0}
    
    # New "first to zero" detection system
    first_to_zero_flag = None     # 'p1' or 'p2' - who went to zero first
    both_at_zero = False          # Are both players currently at zero?
    round_winner_decided = False  # Has this round been decided?
    
    # History tracking
    health_history = deque(maxlen=60)  # Last 1 second at 60fps
    event_log = deque(maxlen=15)  # Keep last 15 events
    
    # Previous state tracking for transitions
    prev_game_state = game_state
    prev_p1_rounds = 0
    prev_p2_rounds = 0
    prev_confirmed_p1 = 0
    prev_confirmed_p2 = 0
    prev_indicator_states = {}
    prev_health = {'p1': 0.0, 'p2': 0.0}
    prev_pending_winner = None
    in_death_period = False
    waiting_for_next_round = False
    waiting_for_next_match = False
    
    # Debug window toggle
    show_debug = False
    print("\nPress 'D' in the debug window to toggle display, 'Q' to quit\n")
    time.sleep(2)
    
    try:
        while True:
            # Capture frame
            frame = camera.get_latest_frame()
            if frame is None:
                continue
            
            frame_count += 1
            current_time = time.time()
            
            # Extract health bars (using exact method from main_2.py)
            strip = frame[Y_HEALTH:Y_HEALTH+1]
            m1 = cv2.inRange(strip[:, X1_P1:X2_P1], LOWER_BGR, UPPER_BGR)
            m2 = cv2.inRange(strip[:, X1_P2:X2_P2], LOWER_BGR, UPPER_BGR)
            
            # Calculate health with high precision
            pct1 = float(cv2.countNonZero(m1)) / float(LEN_P1) * 100.0
            pct2 = float(cv2.countNonZero(m2)) / float(LEN_P2) * 100.0
            
            # DEBUG: Periodic health status (every 3 seconds)
            if frame_count % 180 == 0:
                print(f"DEBUG: P1={pct1:.1f}% P2={pct2:.1f}% | Flag={first_to_zero_flag} | Both_zero={both_at_zero} | Round_decided={round_winner_decided}")
            
            # NEW: Simple "First to Zero" Round Detection System
            # Round start detection (when both players have high health)
            if pct1 >= 95.0 and pct2 >= 95.0 and not round_winner_decided:
                if frame_count % 180 == 0:  # Only print occasionally to avoid spam
                    print(f"ROUND ACTIVE: P1={pct1:.1f}% P2={pct2:.1f}%")
            
            if not round_winner_decided:
                # Check who goes to zero first (and flag them) - MORE PERMISSIVE
                if pct1 <= 2.0 and pct2 > 2.0 and first_to_zero_flag is None:
                    first_to_zero_flag = 'p1'  # P1 went to zero first
                    print(f"FLAG SET: P1 went to zero first! (P1={pct1:.1f}% P2={pct2:.1f}%)")
                elif pct2 <= 2.0 and pct1 > 2.0 and first_to_zero_flag is None:
                    first_to_zero_flag = 'p2'  # P2 went to zero first  
                    print(f"FLAG SET: P2 went to zero first! (P1={pct1:.1f}% P2={pct2:.1f}%)")
                
                # Reset flag only if flagged player recovers while OTHER player is still high
                # (not when both go to 100% which means round ended)
                if first_to_zero_flag == 'p1' and pct1 > 50.0 and pct2 > 50.0 and not both_at_zero:
                    print(f"FLAG RESET: P1 recovered before both went to zero! (P1={pct1:.1f}% P2={pct2:.1f}%)")
                    first_to_zero_flag = None  # P1 recovered, reset flag
                elif first_to_zero_flag == 'p2' and pct2 > 50.0 and pct1 > 50.0 and not both_at_zero:
                    print(f"FLAG RESET: P2 recovered before both went to zero! (P1={pct1:.1f}% P2={pct2:.1f}%)")
                    first_to_zero_flag = None  # P2 recovered, reset flag
                
                # Track when both players are at zero - MORE PERMISSIVE
                if pct1 <= 2.0 and pct2 <= 2.0:
                    if not both_at_zero:
                        both_at_zero = True
                        print(f"BOTH AT ZERO: P1={pct1:.1f}% P2={pct2:.1f}% | Flag={first_to_zero_flag}")
                
                # Round end: Both restore to high health after being at zero - MORE PERMISSIVE
                if both_at_zero and pct1 >= 95.0 and pct2 >= 95.0 and first_to_zero_flag is not None:
                    print(f"HEALTH RESTORED: P1={pct1:.1f}% P2={pct2:.1f}% | Flag={first_to_zero_flag}")
                    # Winner = whoever was NOT flagged as first to zero
                    winner = 'p2' if first_to_zero_flag == 'p1' else 'p1'
                    
                    # Update totals
                    total_rounds_won[winner] += 1
                    round_state.confirmed[winner] += 1
                    
                    print(f"ROUND END: {winner.upper()} WINS! ({first_to_zero_flag.upper()} went to zero first)")
                    print(f"Score: P1={round_state.confirmed['p1']} P2={round_state.confirmed['p2']} | Total rounds: P1={total_rounds_won['p1']} P2={total_rounds_won['p2']}")
                    
                    # Check for match end
                    if round_state.confirmed[winner] >= 2:
                        total_matches_won[winner] += 1
                        print(f"MATCH END: {winner.upper()} wins match! Total matches: P1={total_matches_won['p1']} P2={total_matches_won['p2']}")
                        round_state.reset()  # Reset for new match
                    
                    # Reset for next round
                    first_to_zero_flag = None
                    both_at_zero = False
                    round_winner_decided = True
            
            # Reset round decision when new round starts - MORE PERMISSIVE
            if round_winner_decided and pct1 >= 95.0 and pct2 >= 95.0:
                print(f"ROUND RESET: Ready for next round (P1={pct1:.1f}% P2={pct2:.1f}%)")
                round_winner_decided = False
            
            # Old detection system completely removed - using simple "first to zero" logic above
            
            # No continuous terminal updates - only print round announcements above
            
            # Optional debug window
            if show_debug or cv2.getWindowProperty('Game Monitor Debug', cv2.WND_PROP_VISIBLE) >= 0:
                debug_frame = frame.copy()
                
                # Draw health bar regions
                cv2.rectangle(debug_frame, (X1_P1, Y_HEALTH-2), (X2_P1, Y_HEALTH+2), (0, 255, 0), 1)
                cv2.rectangle(debug_frame, (X1_P2, Y_HEALTH-2), (X2_P2, Y_HEALTH+2), (0, 255, 0), 1)
                
                # Add text overlays with health and flags
                cv2.putText(debug_frame, f"P1: {pct1:.1f}%", (X1_P1, Y_HEALTH-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                cv2.putText(debug_frame, f"P2: {pct2:.1f}%", (X1_P2, Y_HEALTH-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
                
                # Show current flags
                flag_text = f"First to zero: {first_to_zero_flag or 'None'}"
                cv2.putText(debug_frame, flag_text, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                cv2.putText(debug_frame, f"Both at zero: {both_at_zero}", (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                
                cv2.imshow('Game Monitor Debug', debug_frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('d'):
                    show_debug = not show_debug
    
    except KeyboardInterrupt:
        print("\n\nMonitoring stopped by user.")
    except Exception as e:
        print(f"\nError: {e}")
    finally:
        camera.stop()
        cv2.destroyAllWindows()
        print("Game Monitor shut down.")

if __name__ == "__main__":
    main()