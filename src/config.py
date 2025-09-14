"""
Configuration module for the RL game agent.
All constants, hyperparameters, and settings in one place.
"""
import os
import torch
import numpy as np
from datetime import datetime

# ─── DIRECTORIES ─────────────────────────────────────────────────────────
LOG_DIR = "logs"
MODEL_DIR = "models"
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# ─── LOGGING CONFIG ─────────────────────────────────────────────────────
ENABLE_LOGGING = True
LOG_FILENAME = os.path.join(LOG_DIR, f"game_debug_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
LOG_CSV = "health_results.csv"
ACTIONS_FILE = "lua_emulator/actions.txt"
ROUND_VALIDATION_LOG = "round_validation.csv"

# ─── CAPTURE REGION ─────────────────────────────────────────────────────
REGION = (0, 0, 624, 548)  # x, y, width, height

# ─── HEALTH BAR COORDINATES ─────────────────────────────────────────────
Y_HEALTH = 116
X1_P1, X2_P1 = 69, 288
X1_P2, X2_P2 = 350, 569
LEN_P1 = X2_P1 - X1_P1
LEN_P2 = X2_P2 - X1_P2

# ─── COLOR RANGES (BGR) ─────────────────────────────────────────────────
LOWER_BGR = np.array([0, 150, 180], dtype=np.uint8)
UPPER_BGR = np.array([30, 175, 220], dtype=np.uint8)

# Red indicators (HSV)
LOW1 = np.array([0, 70, 50], dtype=np.uint8)
HIGH1 = np.array([10, 255, 255], dtype=np.uint8)
LOW2 = np.array([170, 70, 50], dtype=np.uint8)
HIGH2 = np.array([180, 255, 255], dtype=np.uint8)

# Blue indicators (HSV)
BLUE_LOW = np.array([100, 50, 50], dtype=np.uint8)
BLUE_HIGH = np.array([130, 255, 255], dtype=np.uint8)

# Red BGR for additional detection
RED_BGR_LOWER = np.array([0, 0, 150], dtype=np.uint8)
RED_BGR_UPPER = np.array([60, 60, 255], dtype=np.uint8)

# ─── ROUND INDICATORS ─────────────────────────────────────────────────────
ROUND_INDICATORS = {
    'p1_round1': (270, 135, 278, 140),
    'p1_round2': (245, 135, 253, 140),
    'p2_round1': (373, 135, 381, 140),
    'p2_round2': (396, 135, 404, 140),
}

# ─── TRANSFORM STATE DETECTION ─────────────────────────────────────────
PIXEL_RECTS = [
    ("P1_R1_pixel", 71, 475, 72, 476),
    ("P2_R2_pixel", 520, 475, 521, 476),
]

STATE_MAP = {
    (200, 200, 200): "can transform",
    (48, 48, 248): "transformed",
    (240, 128, 0): "cannot transform",
}

AREA_RECTS = [
    ("P1_R1_area", 71, 480, 177, 481),
    ("P2_R2_area", 469, 480, 575, 481),
]

BLACK_BGR = np.array([0, 0, 8], dtype=np.uint8)

# ─── RL HYPERPARAMETERS ─────────────────────────────────────────────────
FRAME_STACK = 4          # DQN-style 4-frame decision stack
HOLD_FRAMES = 4          # action repeat (frame-skip)
CNN_SIZE = (84, 84)

ACTIONS = ['special', "kick", "transform", "jump", "squat", "left", "right"]
NUM_ACTIONS = len(ACTIONS)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_FRAMES = 1000
GAMMA = 0.95
BATCH_SIZE = 32
TARGET_SYNC = 1000

# Replay + training schedule
REPLAY_SIZE = 10000
LEARNING_STARTS = 50_00   # warm-up before learning
TRAIN_FREQ = 4              # train every N learner ticks

LEARNING_RATE = 5e-5
REWARD_CLIP = 1.0
FINAL_REWARD = 1.0
TAU = 0.005                 # Soft update coefficient

# ─── HEALTH THRESHOLDS ─────────────────────────────────────────────────
HEALTH_LIMIT = 98
DEATH_THRESHOLD = 2.0       # Both must be <= 2%
ALIVE_THRESHOLD = 95.0      # Both must be >= 95% to confirm round start
ZERO_HEALTH_DURATION = 1.0  # Must be at low health for 1 second

# ─── ROUND DETECTION TIMINGS ─────────────────────────────────────────────
FULL_HOLD_SEC = 0.30        # Require 300ms at full health
CONFIRMATION_TIME = 0.3     # Legacy timer-based confirmation
HEALTH_LOST_TIMEOUT = 5.0   # 5 seconds before entering fallback mode

# ─── MODEL SETTINGS ─────────────────────────────────────────────────────
MODEL_NUMBER = 126
LOAD_CHECKPOINT = f"{MODEL_DIR}/model_match_{MODEL_NUMBER}.pth"
TEST_MODE = False

# ─── EXTRA FEATURES DIMENSION ─────────────────────────────────────────
EXTRA_DIM = 10  # [P1 onehot(4), P2 onehot(4), p1_black_pct, p2_black_pct]

# ─── TRANSFORM ACTION INDEX ─────────────────────────────────────────────
TRANSFORM_IDX = ACTIONS.index("transform")