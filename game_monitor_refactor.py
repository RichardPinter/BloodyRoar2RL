# game_monitor_refactored.py (partial - showing how to use WindowCapture)
import numpy as np
from window_capture import WindowCapture
from ultralytics import YOLO

class GameMonitor:
    def __init__(self, window_title: str):
        # Use our new WindowCapture instead of manual win32gui
        self.capture = WindowCapture(window_title)
        
        if not self.capture.is_valid:
            raise RuntimeError(f"Window not found: {window_title}")
        
        # Initialize YOLO
        self.model = YOLO('yolov8n.pt')
        
        # Health bar parameters (same as before)
        self.health_params = {
            'p1_x': 505,
            'p2_x': 1421,
            'bar_len': 400,
            'y': 155,
            'lower_bgr': np.array([0, 160, 190], dtype=np.uint8),
            'upper_bgr': np.array([20, 180, 220], dtype=np.uint8),
            'drop_per_px': 0.25
        }
    
    def detect_health(self):
        """Detect health bars"""
        # Now we can use our cleaner capture_region method!
        # P1 health bar
        p1_strip = self.capture.capture_region(
            x=self.health_params['p1_x'],
            y=self.health_params['y'],
            width=self.health_params['bar_len'],
            height=1
        )
        
        if p1_strip is None:
            return None, None
        
        # P2 health bar  
        p2_strip = self.capture.capture_region(
            x=self.health_params['p2_x'] - self.health_params['bar_len'],
            y=self.health_params['y'],
            width=self.health_params['bar_len'],
            height=1
        )
        
        if p2_strip is None:
            return None, None
        
        # Rest of health detection logic stays the same...
        # (Your color masking code here)
        
    def get_frame(self):
        """Get current game frame"""
        return self.capture.capture()