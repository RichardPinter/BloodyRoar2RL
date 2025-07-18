import time
import atexit
import comtypes
import dxcam
import numpy as np
import cv2

# ─── CONFIG ────────────────────────────────────────────────────────────────
REGION      = (0, 0, 624, 548)      # x, y, width, height

# Health bar detection (existing working setup)
Y_HEALTH    = 116
X1_P1, X2_P1 = 73, 292
X1_P2, X2_P2 = 355, 574
LEN_P1        = X2_P1 - X1_P1
LEN_P2        = X2_P2 - X1_P2
LOWER_BGR   = np.array([0,150,180], dtype=np.uint8)
UPPER_BGR   = np.array([30,175,220], dtype=np.uint8)

# Round indicator detection (ADJUST THESE COORDINATES!)
ROUND_INDICATORS = {
    'p1_round1': ( 270, 135, 278, 140),   # Player 1, first round indicator
    'p1_round2': ( 245, 135, 253, 140),   # Player 1, second round indicator
    'p2_round1': ( 373, 135, 381, 140),   # Player 2, first round indicator
    'p2_round2': ( 396, 135, 404, 140),  # Player 2, second round indicator
}

# Red color range for round indicators
RED_BGR_LOWER = np.array([0, 0, 150], dtype=np.uint8)
RED_BGR_UPPER = np.array([60, 60, 255], dtype=np.uint8)

# ─── DXCAM SETUP ────────────────────────────────────────────────────────────
comtypes.CoInitialize()
camera = dxcam.create(output_color="BGR")
camera.start(target_fps=60, region=REGION, video_mode=True)
atexit.register(lambda: (camera.stop(), comtypes.CoUninitialize()))

def test_health_detection(frame):
    """Test health bar detection (existing working code)"""
    strip = frame[Y_HEALTH:Y_HEALTH+1]

    # Player 1 health
    m1 = cv2.inRange(strip[:, X1_P1:X2_P1], LOWER_BGR, UPPER_BGR)
    pct1 = cv2.countNonZero(m1) / LEN_P1 * 100.0

    # Player 2 health
    m2 = cv2.inRange(strip[:, X1_P2:X2_P2], LOWER_BGR, UPPER_BGR)
    pct2 = cv2.countNonZero(m2) / LEN_P2 * 100.0

    return pct1, pct2

def test_round_detection(frame):
    """Test round indicator detection using SAME PATTERN as health bars"""
    results = {}

    for name, (x1, y1, x2, y2) in ROUND_INDICATORS.items():
        # Extract region (same as health bar)
        region = frame[y1:y2, x1:x2]

        # Create mask for red pixels (same pattern as health bar)
        mask = cv2.inRange(region, RED_BGR_LOWER, RED_BGR_UPPER)

        # Count red pixels and calculate percentage (same as health bar)
        total_pixels = region.shape[0] * region.shape[1]
        if total_pixels > 0:
            red_pixels = cv2.countNonZero(mask)
            red_pct = red_pixels / total_pixels * 100.0
        else:
            red_pct = 0.0

        results[name] = red_pct

    return results

def main():
    print("=" * 60)
    print("TESTING HEALTH AND ROUND DETECTION")
    print("=" * 60)
    print("Health bars:")
    print(f"  P1: Y={Y_HEALTH}, X={X1_P1}-{X2_P1} (len={LEN_P1})")
    print(f"  P2: Y={Y_HEALTH}, X={X1_P2}-{X2_P2} (len={LEN_P2})")
    print(f"  Color range: {LOWER_BGR} - {UPPER_BGR}")
    print()
    print("Round indicators:")
    for name, coords in ROUND_INDICATORS.items():
        print(f"  {name}: {coords}")
    print(f"  Red range: {RED_BGR_LOWER} - {RED_BGR_UPPER}")
    print()
    print("Press Ctrl+C to stop")
    print("=" * 60)

    last_print_time = 0

    try:
        while True:
            frame = camera.get_latest_frame()
            if frame is None:
                time.sleep(0.01)
                continue

            # Test both detection methods
            p1_health, p2_health = test_health_detection(frame)
            round_results = test_round_detection(frame)

            # Print results every 0.5 seconds
            current_time = time.time()
            if current_time - last_print_time > 0.5:
                print(f"\n[{time.strftime('%H:%M:%S')}]")
                print(f"HEALTH:  P1={p1_health:5.1f}%  P2={p2_health:5.1f}%")
                print("ROUNDS:")
                for name, pct in round_results.items():
                    status = "RED" if pct > 50 else "---"
                    print(f"  {name:10}: {pct:5.1f}% ({status})")

                # Simple round detection logic
                p1_rounds = sum([round_results['p1_round1'] > 50, round_results['p1_round2'] > 50])
                p2_rounds = sum([round_results['p2_round1'] > 50, round_results['p2_round2'] > 50])
                print(f"SCORE:   P1={p1_rounds} rounds  P2={p2_rounds} rounds")

                # Round start detection
                if p1_health >= 99 and p2_health >= 99:
                    print(">>> ROUND START DETECTED! (both at 99%+)")

                # Round end detection
                if p1_rounds > 0 or p2_rounds > 0:
                    print(">>> ROUND WON DETECTED!")

                last_print_time = current_time

            time.sleep(0.01)

    except KeyboardInterrupt:
        print("\n" + "=" * 60)
        print("Test stopped by user")
        print("=" * 60)

    camera.stop()

if __name__ == "__main__":
    main()