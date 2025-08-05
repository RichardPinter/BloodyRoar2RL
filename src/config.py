import numpy as np

# ─── CAPTURE REGION ───────────────────────────────────────────────
REGION       = (0, 0, 624, 548)
Y_HEALTH     = 116
X1_P1, X2_P1 = 73, 292
X1_P2, X2_P2 = 355, 574

# ─── IMAGE PROCESSING ────────────────────────────────────────────
LOWER_BGR = [0, 150, 180]
UPPER_BGR = [30, 175, 220]
CNN_SIZE  = (84, 84)
FRAME_STACK = 10

# ─── ACTIONS ──────────────────────────────────────────────────────
ACTIONS = ['special','kick','transform','jump','squat','left','right']

# ─── RL HYPERPARAMETERS ──────────────────────────────────────────
GAMMA           = 0.99
BATCH_SIZE      = 32
TARGET_SYNC     = 1000
REPLAY_SIZE     = 10000
MIN_BUFFER_SIZE = 100
LEARNING_RATE   = 1e-4

# ─── FILE PATHS ───────────────────────────────────────────────────
MODEL_DIR      = '../models'
LOG_CSV        = 'health_results.csv'
ACTIONS_FILE   = '../actions.txt'
LOG_DIR        = '../logs'

# ─── MISC ─────────────────────────────────────────────────────────
MAX_FRAMES = 1000
HEALTH_LIMIT = 99.0
TEST_MODE  = False

ROUND_INDICATORS = {
    'p1_round1': (270, 135, 278, 140),
    'p1_round2': (245, 135, 253, 140),
    'p2_round1': (373, 135, 381, 140),
    'p2_round2': (396, 135, 404, 140),
}
RED_BGR_LOWER   = np.array([0, 0, 150], dtype=np.uint8)
RED_BGR_UPPER   = np.array([60, 60, 255], dtype=np.uint8)

# ─── TRANSFORM STATE & BLACK‐PIXEL HELPERS ─────────────────────────────────
PIXEL_RECTS = [
    ("P1_R1_pixel", 71, 475, 72, 476),
    ("P2_R2_pixel", 520, 475, 521, 476),
]
STATE_MAP = {
    (200, 200, 200): "can transform",
    ( 48,  48, 248): "transformed",
    (240, 128,   0): "cannot transform",
}

AREA_RECTS = [
    ("P1_R1_area", 71, 480, 177, 481),
    ("P2_R2_area", 469, 480, 575, 481),
]
BLACK_BGR = np.array([0, 0, 8], dtype=np.uint8)