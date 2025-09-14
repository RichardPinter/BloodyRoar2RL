"""
Screen capture module using DXCam.
Handles high-performance game frame capture.
"""
import time
import atexit
import threading
import comtypes
import dxcam
from queue import Queue
from src.config import REGION
from src.logging_utils import log_state, log_debug

class ScreenCapture:
    """Manages DXCam screen capture in a separate thread"""
    
    def __init__(self, frame_queue, stop_event):
        self.frame_queue = frame_queue
        self.stop_event = stop_event
        self.camera = None
        self.thread = None
        self.start_time = None
        
    def initialise(self):
        """Initialise DXCam camera"""
        comtypes.CoInitialise()
        self.camera = dxcam.create(output_color="BGR")
        self.camera.start(target_fps=60, region=REGION, video_mode=True)
        
        # Register cleanup
        atexit.register(self.cleanup)
        
        log_state(f"Screen capture initialised: Region {REGION}, 60 FPS target")
        
    def cleanup(self):
        """Clean up DXCam resources"""
        if self.camera:
            self.camera.stop()
        comtypes.CoUninitialise()
        log_debug("Screen capture cleaned up")
    
    def producer_loop(self):
        """Main capture loop running in separate thread"""
        self.start_time = time.perf_counter()
        frames_captured = 0
        
        while not self.stop_event.is_set():
            frame = self.camera.get_latest_frame()
            if frame is not None:
                timestamp = time.perf_counter() - self.start_time
                try:
                    self.frame_queue.put((frame.copy(), timestamp), timeout=0.001)
                    frames_captured += 1
                    
                    # Log stats periodically
                    if frames_captured % 600 == 0:  # Every 10 seconds at 60fps
                        elapsed = time.perf_counter() - self.start_time
                        fps = frames_captured / elapsed
                        log_debug(f"Capture stats: {frames_captured} frames, {fps:.1f} FPS avg")
                except:
                    # Queue full, skip frame
                    pass
        
        log_state(f"Screen capture stopped after {frames_captured} frames")
    
    def start(self):
        """Start the capture thread"""
        self.initialise()
        self.thread = threading.Thread(
            target=self.producer_loop,
            name="ScreenCapture",
            daemon=True
        )
        self.thread.start()
        log_state("Screen capture thread started")
        
    def join(self):
        """Wait for capture thread to finish"""
        if self.thread:
            self.thread.join()