# window_capture.py
import win32gui
import win32con
from mss import mss
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple

@dataclass
class WindowInfo:
    """Container for window information"""
    hwnd: int
    title: str
    x: int
    y: int
    width: int
    height: int
    
    @property
    def monitor_dict(self):
        """MSS-compatible monitor dictionary"""
        return {
            'left': self.x,
            'top': self.y,
            'width': self.width,
            'height': self.height
        }

class WindowCapture:
    """Handles window detection and screen capture for Windows applications"""
    
    def __init__(self, window_title: str):
        self.window_title = window_title
        self.sct = mss()
        self._window_info: Optional[WindowInfo] = None
        self._last_valid_hwnd: Optional[int] = None
        
        # Initialize window info
        self.refresh_window()
    
    def find_window(self) -> Optional[int]:
        """Find window by title, returns window handle or None"""
        try:
            hwnd = win32gui.FindWindow(None, self.window_title)
            if hwnd:
                return hwnd
        except Exception as e:
            print(f"Error finding window: {e}")
        return None
    
    def get_window_info(self, hwnd: int) -> Optional[WindowInfo]:
        """Get window position and size information"""
        try:
            # Check if window still exists and is visible
            if not win32gui.IsWindow(hwnd) or not win32gui.IsWindowVisible(hwnd):
                return None
            
            # Get window rectangle
            rect = win32gui.GetClientRect(hwnd)
            x, y = win32gui.ClientToScreen(hwnd, (0, 0))
            width = rect[2]
            height = rect[3]
            
            # Verify sensible dimensions
            if width <= 0 or height <= 0:
                return None
            
            return WindowInfo(
                hwnd=hwnd,
                title=self.window_title,
                x=x,
                y=y,
                width=width,
                height=height
            )
            
        except Exception as e:
            print(f"Error getting window info: {e}")
            return None
    
    def refresh_window(self) -> bool:
        """Refresh window information, returns True if successful"""
        # Try to use last valid handle first
        if self._last_valid_hwnd:
            info = self.get_window_info(self._last_valid_hwnd)
            if info:
                self._window_info = info
                return True
        
        # Otherwise search for window
        hwnd = self.find_window()
        if hwnd:
            info = self.get_window_info(hwnd)
            if info:
                self._window_info = info
                self._last_valid_hwnd = hwnd
                return True
        
        return False
    
    @property
    def is_valid(self) -> bool:
        """Check if we have valid window information"""
        return self._window_info is not None
    
    @property
    def window_info(self) -> Optional[WindowInfo]:
        """Get current window information"""
        return self._window_info
    
    def capture(self) -> Optional[np.ndarray]:
        """Capture the entire window as numpy array (BGR format)"""
        if not self.is_valid:
            if not self.refresh_window():
                return None
        
        try:
            # Capture screen
            screenshot = self.sct.grab(self._window_info.monitor_dict)
            
            # Convert to numpy array (BGRA -> BGR)
            image = np.array(screenshot)
            return image[:, :, :3]  # Remove alpha channel
            
        except Exception as e:
            print(f"Error capturing window: {e}")
            # Try to refresh window info on next attempt
            self._window_info = None
            return None
    
    def capture_region(self, x: int, y: int, width: int, height: int) -> Optional[np.ndarray]:
        """Capture a specific region relative to window (BGR format)"""
        if not self.is_valid:
            if not self.refresh_window():
                return None
        
        try:
            # Create region coordinates (relative to window)
            region = {
                'left': self._window_info.x + x,
                'top': self._window_info.y + y,
                'width': width,
                'height': height
            }
            
            # Validate region is within window bounds
            if (x < 0 or y < 0 or 
                x + width > self._window_info.width or 
                y + height > self._window_info.height):
                print(f"Warning: Region extends outside window bounds")
            
            # Capture region
            screenshot = self.sct.grab(region)
            image = np.array(screenshot)
            return image[:, :, :3]  # Remove alpha channel
            
        except Exception as e:
            print(f"Error capturing region: {e}")
            return None
    
    def get_relative_position(self, screen_x: int, screen_y: int) -> Optional[Tuple[int, int]]:
        """Convert screen coordinates to window-relative coordinates"""
        if not self.is_valid:
            return None
        
        rel_x = screen_x - self._window_info.x
        rel_y = screen_y - self._window_info.y
        return (rel_x, rel_y)
    
    def get_screen_position(self, window_x: int, window_y: int) -> Optional[Tuple[int, int]]:
        """Convert window-relative coordinates to screen coordinates"""
        if not self.is_valid:
            return None
        
        screen_x = window_x + self._window_info.x
        screen_y = window_y + self._window_info.y
        return (screen_x, screen_y)


# Example usage and testing
if __name__ == "__main__":
    # Test the window capture
    WINDOW_TITLE = "Bloody Roar II (USA) [PlayStation] - BizHawk"
    
    capture = WindowCapture(WINDOW_TITLE)
    
    if capture.is_valid:
        print(f"Window found: {capture.window_info}")
        
        # Test full capture
        image = capture.capture()
        if image is not None:
            print(f"Captured image shape: {image.shape}")
        
        # Test region capture (e.g., health bar area)
        health_region = capture.capture_region(505, 150, 400, 10)
        if health_region is not None:
            print(f"Health region shape: {health_region.shape}")
    else:
        print(f"Window '{WINDOW_TITLE}' not found!")