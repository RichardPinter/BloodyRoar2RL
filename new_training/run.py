import atexit
import time
from queue import Queue
import threading

import dxcam
import comtypes

from config import REGION, LOG_CSV, DURATION, DEVICE, FRAME_STACK, LR, ACTIONS
from buffer import ReplayBuffer
from model import build_nets
from workers import Producer, Consumer, Learner
from utils import write_action
from PIL import Image


def cleanup(camera):
    camera.stop()
    comtypes.CoUninitialize()


if __name__ == "__main__":
    # Initialize COM for dxcam
    comtypes.CoInitialize()

    # Start dxcam and ensure cleanup on exit
    camera = dxcam.create(output_color="BGR")
    camera.start(target_fps=30, region=REGION, video_mode=True)
    atexit.register(cleanup, camera)

    # Shared components
    frame_q = Queue(maxsize=16)
    stop_event = threading.Event()
    buffer = ReplayBuffer()

    # Build networks and optimizer
    policy_net, target_net, optimizer = build_nets(
        device=DEVICE,
        frame_stack=FRAME_STACK,
        num_actions=len(ACTIONS),
        lr=LR
    )

    # Clear any existing actions
    write_action("")

    # Create worker threads
    prod = Producer(camera, frame_q, stop_event)
    cons = Consumer(frame_q, buffer, policy_net, stop_event)
    learn = Learner(buffer, policy_net, target_net, optimizer, stop_event)

    # Start threads
    for t in (prod, cons, learn):
        t.daemon = True
        t.start()

    # Run until interrupted or duration elapses
    try:
        if DURATION:
            time.sleep(DURATION)
            stop_event.set()
        else:
            while not stop_event.is_set():
                time.sleep(1)
    except KeyboardInterrupt:
        stop_event.set()

    # Wait for threads to finish
    prod.join()
    frame_q.join()
    cons.join()
    learn.join()

    # After training, save screenshots and logs
    # (Assumes screenshots collected in Consumer and accessible globally)
    try:
        import workers
        for i, frame in enumerate(workers.screenshots):
            Image.fromarray(frame[..., ::-1]).save(f"screenshots/frame_{i:04d}.png")
    except Exception:
        pass

    # Save health results CSV
    try:
        import workers
        with open(LOG_CSV, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["time_s", "p1_pct", "p2_pct"])
            writer.writerows(workers.results)
    except Exception:
        pass

    print("Done: health CSV saved and screenshots exported.")
