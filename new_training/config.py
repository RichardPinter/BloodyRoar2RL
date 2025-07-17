# config.py
import numpy as np
import torch

REGION       = (0, 0, 680, 540)
Y_HEALTH     = 115
X1_P1, X2_P1 = 78, 298
X1_P2, X2_P2 = 358, 578

slice_p1    = (slice(Y_HEALTH, Y_HEALTH+1), slice(X1_P1, X2_P1), slice(None))
slice_p2    = (slice(Y_HEALTH, Y_HEALTH+1), slice(X1_P2, X2_P2), slice(None))

LOWER_BGR    = np.array([15, 205, 230], dtype=np.uint8)
UPPER_BGR    = np.array([30, 220, 245], dtype=np.uint8)

FRAME_STACK  = 4
CNN_SIZE     = (84, 84)
ACTIONS      = ["jump", "kick", "transform", "squat", "left", "right"]

DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LR           = 1e-4
GAMMA        = 0.99
BATCH_SIZE   = 32
TARGET_SYNC  = 1000
REPLAY_SIZE  = 10000

ACTIONS_FILE = "../actions.txt"
LOG_CSV      = "health_results.csv"
MAX_FRAMES   = 100

DURATION      = 60.0
