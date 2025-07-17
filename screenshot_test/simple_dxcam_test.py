import time
import gc

# Monkey-patch comtypes so its __del__ ignores COM teardown errors
import comtypes
from comtypes._post_coinit.unknwn import _compointer_base

_orig_del = _compointer_base.__del__
def _safe_del(self):
    try:
        _orig_del(self)
    except (ValueError, OSError):
        # swallow “without VTable” and access-violation errors
        pass
_compointer_base.__del__ = _safe_del

import dxcam
import mss

def benchmark_dxcam(iterations=100):
    """GPU-accelerated screen grabs via DXGI/COM."""
    comtypes.CoInitialize()
    camera = dxcam.create()
    try:
        # Warm-up
        camera.grab()
        time.sleep(0.1)

        # Timed captures
        start = time.perf_counter()
        for _ in range(iterations):
            camera.grab()
        end = time.perf_counter()

        total  = end - start
        avg_ms = total / iterations * 1000
        fps    = iterations / total

        print(f"[dxcam] Total time for {iterations} grabs: {total:.3f} s")
        print(f"[dxcam] Average per grab:           {avg_ms:.2f} ms")
        print(f"[dxcam] Approx FPS:                 {fps:.1f}")
    finally:
        camera.stop()
        camera.release()
        del camera
        gc.collect()
        comtypes.CoUninitialize()

def benchmark_mss(iterations=100, region=None):
    """CPU-based screen grabs via mss."""
    sct = mss.mss()
    # Warm-up
    _ = sct.grab(region or sct.monitors[1])
    time.sleep(0.1)

    # Timed captures
    start = time.perf_counter()
    for _ in range(iterations):
        _ = sct.grab(region or sct.monitors[1])
    end = time.perf_counter()

    total  = end - start
    avg_ms = total / iterations * 1000
    fps    = iterations / total

    print(f"[mss]  Total time for {iterations} grabs: {total:.3f} s")
    print(f"[mss]  Average per grab:           {avg_ms:.2f} ms")
    print(f"[mss]  Approx FPS:                 {fps:.1f}")

    del sct
    gc.collect()

if __name__ == "__main__":
    # DXCAM benchmark
    benchmark_dxcam(100)
    print("-" * 40)

    # MSS full-screen benchmark
    benchmark_mss(100)
    print("-" * 40)

    # (Optional) MSS smaller region
    small_region = {"top": 0, "left": 0, "width": 800, "height": 600}
    benchmark_mss(100, region=small_region)
