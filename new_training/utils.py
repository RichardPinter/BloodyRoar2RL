import cv2
import numpy as np
from config import LOWER_BGR, UPPER_BGR, FRAME_STACK, CNN_SIZE, ACTIONS_FILE, slice_p1, slice_p2


def extract_health(frame: np.ndarray) -> tuple[float, float]:
    """
    Extracts player health percentages from a frame.

    Args:
        frame: BGR image array of shape (H, W, 3).
    Returns:
        (pct1, pct2): health percentages for player 1 and player 2.
    """
    # Threshold health bars
    mask1 = cv2.inRange(frame[slice_p1], LOWER_BGR, UPPER_BGR)
    pct1 = cv2.countNonZero(mask1) / mask1.size * 100.0
    mask2 = cv2.inRange(frame[slice_p2], LOWER_BGR, UPPER_BGR)
    pct2 = cv2.countNonZero(mask2) / mask2.size * 100.0
    return pct1, pct2


def preprocess_frame(frame: np.ndarray) -> np.ndarray:
    """
    Convert BGR frame to normalized grayscale and resize for model input.

    Args:
        frame: BGR image array of shape (H, W, 3).
    Returns:
        processed: float32 array of shape CNN_SIZE, values in [0,1].
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, CNN_SIZE, interpolation=cv2.INTER_NEAREST)
    return resized.astype(np.float32) / 255.0


def write_action(action: str, path: str = ACTIONS_FILE) -> None:
    """
    Atomically write the chosen action to file for Lua polling.

    Args:
        action: action string to write.
        path: path to actions file.
    """
    try:
        with open(path, "w") as f:
            f.write(action)
    except IOError as e:
        raise RuntimeError(f"Failed to write action '{action}' to {path}: {e}")
