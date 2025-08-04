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