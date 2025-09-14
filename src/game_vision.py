#!/usr/bin/env python3
"""
Computer vision functions for game state detection.
Handles health bars, round indicators, and transform states.
"""
import cv2
import numpy as np
from src.config import (
    Y_HEALTH, X1_P1, X2_P1, X1_P2, X2_P2, LEN_P1, LEN_P2,
    LOWER_BGR, UPPER_BGR, ROUND_INDICATORS,
    LOW1, HIGH1, LOW2, HIGH2, BLUE_LOW, BLUE_HIGH,
    PIXEL_RECTS, STATE_MAP, AREA_RECTS, BLACK_BGR,
    TRANSFORM_IDX, NUM_ACTIONS, EXTRA_DIM
)
from src.logging_utils import log_debug
import torch

def detect_health(frame):
    """
    Detect health percentages for both players.
    Returns: (p1_health_pct, p2_health_pct)
    """
    strip = frame[Y_HEALTH:Y_HEALTH+1]
    m1 = cv2.inRange(strip[:, X1_P1:X2_P1], LOWER_BGR, UPPER_BGR)
    m2 = cv2.inRange(strip[:, X1_P2:X2_P2], LOWER_BGR, UPPER_BGR)
    pct1 = cv2.countNonZero(m1) / LEN_P1 * 100.0
    pct2 = cv2.countNonZero(m2) / LEN_P2 * 100.0
    return pct1, pct2

def detect_round_indicators(frame):
    """
    Round detection using HSV color space.
    Returns: (results_dict, color_states_dict)
    - results_dict: {indicator_name: bool} for backward compatibility
    - color_states_dict: {indicator_name: 'red'/'blue'/'unknown'}
    """
    results = {}
    states = {}
    debug_info = []
    
    for name, (x1, y1, x2, y2) in ROUND_INDICATORS.items():
        # Extract region
        region = frame[y1:y2, x1:x2]
        
        # Convert to HSV for more reliable color detection
        hsv_region = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
        
        # Red detection: wraps around in HSV, so we need two masks
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
        
        # Determine state: prioritize red over blue
        if red_pct > 85.0:
            states[name] = 'red'
            results[name] = True  # Keep backward compatibility
        elif blue_pct > 30.0:  # Lower threshold for blue
            states[name] = 'blue'
            results[name] = False
        else:
            states[name] = 'unknown'
            results[name] = False
        
        # Debug logging for calibration
        if red_pct > 5 or blue_pct > 5:
            debug_info.append(f"{name}: R:{red_pct:.1f}% B:{blue_pct:.1f}% -> {states[name]}")
    
    # Log all debug info at once to reduce file writes
    if debug_info:
        log_debug(f"Round indicators: {', '.join(debug_info)}")
    
    return results, states

def classify_transform_state(frame):
    """
    Classify transform state for both players based on pixel colors.
    Returns: {'P1_R1_pixel': state, 'P2_R2_pixel': state}
    where state is one of: 'can transform', 'transformed', 'cannot transform', 'unknown'
    """
    out = {}
    for player, x1, y1, x2, y2 in PIXEL_RECTS:
        b, g, r = frame[y1, x1]
        out[player] = STATE_MAP.get((int(b), int(g), int(r)), "unknown")
    return out

def compute_black_stats(frame):
    """
    Compute percentage of black pixels in transform areas.
    Returns: (pct_dict, range_dict)
    - pct_dict: {'P1_R1_area': percentage, 'P2_R2_area': percentage}
    - range_dict: {'P1_R1_area': channel_ranges, 'P2_R2_area': channel_ranges}
    """
    pct_out = {}
    range_out = {}
    for player, x1, y1, x2, y2 in AREA_RECTS:
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            pct_out[player] = None
            range_out[player] = None
            continue
        
        mask = cv2.inRange(roi, BLACK_BGR, BLACK_BGR)
        cnt = int(cv2.countNonZero(mask))
        total = roi.shape[0] * roi.shape[1]
        pct = cnt / total * 100.0
        pct_out[player] = pct
        
        B, G, R = roi[:,:,0], roi[:,:,1], roi[:,:,2]
        range_out[player] = {
            'B': (int(B.min()), int(B.max())),
            'G': (int(G.min()), int(G.max())),
            'R': (int(R.min()), int(R.max())),
        }
    return pct_out, range_out

def legal_mask_from_ts(ts):
    """
    Build a boolean mask [NUM_ACTIONS] from the on-screen transform state.
    Allow 'transform' only if P1 == 'can transform' and not already 'transformed'.
    """
    p1 = ts.get("P1_R1_pixel", "unknown")
    can_t = (p1 == "can transform")
    is_t = (p1 == "transformed")
    allow_transform = can_t and (not is_t)
    
    mask = np.ones(NUM_ACTIONS, dtype=bool)
    if not allow_transform:
        mask[TRANSFORM_IDX] = False
    return mask

def legal_mask_from_extras(extras_tensor):
    """
    Mask for a batch of next-states (used in learner targets).
    Returns BoolTensor [B, NUM_ACTIONS].
    """
    device = extras_tensor.device
    mask = torch.ones((extras_tensor.size(0), NUM_ACTIONS), dtype=torch.bool, device=device)
    
    if EXTRA_DIM == 10:
        # extras: [P1 onehot(4), P2 onehot(4), p1_black, p2_black]
        p1_onehot = extras_tensor[:, 0:4]             # (B,4)
        can_t = p1_onehot[:, 0] > 0.5                 # 'can transform'
        transformed = p1_onehot[:, 1] > 0.5           # 'transformed'
        allow_transform = can_t & (~transformed)
    else:
        # extras[:,0] is code scaled to [0,1] by /3.0 -> recover 0..3
        p1_code = (extras_tensor[:, 0] * 3.0).round().long().clamp(0, 3)
        can_t = p1_code.eq(0)
        transformed = p1_code.eq(1)
        allow_transform = can_t & (~transformed)
    
    mask[:, TRANSFORM_IDX] = allow_transform
    return mask

def compute_extra_features(frame):
    """
    Compute extra features for neural network input.
    Returns: numpy array of shape (EXTRA_DIM,)
    """
    ts = classify_transform_state(frame)
    bp, _ = compute_black_stats(frame)
    
    code = {'can transform': 0, 'transformed': 1, 'cannot transform': 2, 'unknown': 3}
    c1 = code[ts['P1_R1_pixel']]
    c2 = code[ts['P2_R2_pixel']]
    
    onehot1 = np.eye(4, dtype=np.float32)[c1]
    onehot2 = np.eye(4, dtype=np.float32)[c2]
    
    p1_black = bp.get('P1_R1_area', 0.0) / 100.0 if bp.get('P1_R1_area') is not None else 0.0
    p2_black = bp.get('P2_R2_area', 0.0) / 100.0 if bp.get('P2_R2_area') is not None else 0.0
    
    return np.concatenate([onehot1, onehot2, [p1_black, p2_black]]).astype(np.float32)