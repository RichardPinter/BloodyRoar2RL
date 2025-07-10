# health_detector.py
import numpy as np
import time
from dataclasses import dataclass
from typing import Tuple, Optional
from window_capture import WindowCapture

@dataclass
class HealthBarConfig:
    """Configuration for health bar detection"""
    # Position and size
    p1_x: int = 505          # P1 health bar left edge
    p2_x: int = 1421         # P2 health bar right edge  
    bar_length: int = 400    # Length of health bar
    bar_y: int = 155         # Y position of health bar
    bar_height: int = 1      # Height to sample (1 pixel strip)
    
    # Color detection (BGR format)
    lower_bgr: Tuple[int, int, int] = (0, 160, 190)    # Lower bound for yellow
    upper_bgr: Tuple[int, int, int] = (20, 180, 220)   # Upper bound for yellow
    
    # Calibration
    health_drop_per_pixel: float = 0.25  # % health lost per pixel

@dataclass 
class HealthState:
    """Container for health detection results"""
    p1_health: float  # 0-100
    p2_health: float  # 0-100
    p1_pixels: int    # Raw pixel count
    p2_pixels: int    # Raw pixel count
    
class HealthDetector:
    """Detects player health from fighting game health bars"""
    
    def __init__(self, config: Optional[HealthBarConfig] = None):
        self.config = config or HealthBarConfig()
        
        # Throttling for health bar visibility checks
        self.last_visibility_check = 0
        self.visibility_check_interval = 0.1  # Check every 100ms
        
    def detect_health_from_strip(self, pixel_strip: np.ndarray, 
                                reverse: bool = False) -> Tuple[float, int]:
        """
        Detect health from a 1-pixel high strip of health bar
        
        Args:
            pixel_strip: BGR pixel array shape (1, width, 3)
            reverse: If True, scan from right to left (for P2)
            
        Returns:
            (health_percentage, damage_pixels)
        """
        # Extract color channels
        if len(pixel_strip.shape) == 3:
            pixel_strip = pixel_strip[0]  # Remove height dimension
            
        b, g, r = pixel_strip[:, 0], pixel_strip[:, 1], pixel_strip[:, 2]
        
        # Create mask for yellow health bar color
        mask = (
            (r >= self.config.lower_bgr[2]) & (r <= self.config.upper_bgr[2]) &
            (g >= self.config.lower_bgr[1]) & (g <= self.config.upper_bgr[1]) &
            (b >= self.config.lower_bgr[0]) & (b <= self.config.upper_bgr[0])
        )
        
        # Find where health bar ends
        non_yellow = np.where(~mask)[0]
        
        if reverse:
            # P2: scan from right to left
            damage_edge = non_yellow[0] if len(non_yellow) > 0 else len(mask)
            damage_pixels = len(mask) - damage_edge
        else:
            # P1: scan from left to right  
            damage_edge = non_yellow[-1] + 1 if len(non_yellow) > 0 else 0
            damage_pixels = damage_edge
            
        # Calculate health percentage
        health_lost = damage_pixels * self.config.health_drop_per_pixel
        health_pct = max(0.0, min(100.0, 100.0 - health_lost))
        
        return health_pct, damage_pixels
    
    def detect(self, capture: WindowCapture) -> Optional[HealthState]:
        """
        Detect both players' health using window capture
        
        Args:
            capture: WindowCapture instance
            
        Returns:
            HealthState or None if detection fails
        """
        # Capture P1 health bar strip
        p1_strip = capture.capture_region(
            x=self.config.p1_x,
            y=self.config.bar_y,
            width=self.config.bar_length,
            height=self.config.bar_height
        )
        
        if p1_strip is None:
            return None
            
        # Capture P2 health bar strip  
        p2_strip = capture.capture_region(
            x=self.config.p2_x - self.config.bar_length,
            y=self.config.bar_y,
            width=self.config.bar_length,
            height=self.config.bar_height
        )
        
        if p2_strip is None:
            return None
            
        # Detect health for both players
        p1_health, p1_pixels = self.detect_health_from_strip(p1_strip, reverse=False)
        p2_health, p2_pixels = self.detect_health_from_strip(p2_strip, reverse=True)
        
        return HealthState(
            p1_health=p1_health,
            p2_health=p2_health,
            p1_pixels=p1_pixels,
            p2_pixels=p2_pixels
        )
    
    def visualize_health_bars(self, image: np.ndarray, health_state: HealthState) -> np.ndarray:
        """Add health bar visualization to image"""
        import cv2
        display = image.copy()
        
        # P1 health bar outline (green)
        cv2.rectangle(display,
                     (self.config.p1_x, self.config.bar_y - 5),
                     (self.config.p1_x + self.config.bar_length, self.config.bar_y + 5),
                     (0, 255, 0), 2)
        
        # P1 health text
        cv2.putText(display, f"P1: {health_state.p1_health:.1f}%",
                   (self.config.p1_x, self.config.bar_y - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # P2 health bar outline (red)
        cv2.rectangle(display,
                     (self.config.p2_x - self.config.bar_length, self.config.bar_y - 5),
                     (self.config.p2_x, self.config.bar_y + 5),
                     (0, 0, 255), 2)
        
        # P2 health text
        cv2.putText(display, f"P2: {health_state.p2_health:.1f}%",
                   (self.config.p2_x - 150, self.config.bar_y - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        return display
    
    def health_bars_visible(self, capture: WindowCapture) -> bool:
        """
        Check if health bars are visible (indicates we're in combat)
        
        Args:
            capture: WindowCapture instance
            
        Returns:
            True if health bars are visible and valid, False otherwise
        """
        # Throttle checks to avoid excessive computation
        current_time = time.time()
        if current_time - self.last_visibility_check < self.visibility_check_interval:
            # Use cached result for recent checks
            return hasattr(self, '_last_visibility_result') and self._last_visibility_result
        
        self.last_visibility_check = current_time
        
        try:
            # Use existing detect method to check health bars
            health_state = self.detect(capture)
            
            if health_state is not None:
                # Health bars are visible if we get reasonable health values
                p1_health = health_state.p1_health
                p2_health = health_state.p2_health
                
                # Valid health values indicate visible health bars
                if (p1_health >= 0 and p1_health <= 100 and 
                    p2_health >= 0 and p2_health <= 100):
                    
                    self._last_visibility_result = True
                    return True
            
            # If we can't detect valid health bars, they're not visible
            self._last_visibility_result = False
            return False
            
        except Exception as e:
            print(f"Error checking health bar visibility: {e}")
            self._last_visibility_result = False
            return False
    
    def wait_for_health_bars(self, capture: WindowCapture, timeout: float = 30.0) -> bool:
        """
        Wait until health bars are visible (for round transitions)
        
        Args:
            capture: WindowCapture instance
            timeout: Maximum time to wait in seconds
            
        Returns:
            True if health bars became visible, False if timeout
        """
        print("⏳ Waiting for health bars to appear...")
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            if self.health_bars_visible(capture):
                elapsed = time.time() - start_time
                print(f"✅ Health bars detected after {elapsed:.1f}s - ready to start!")
                return True
            
            # Print status every 5 seconds
            elapsed = time.time() - start_time
            if elapsed > 0 and int(elapsed) % 5 == 0:
                print(f"   Still waiting for health bars... ({elapsed:.1f}s elapsed)")
            
            time.sleep(0.5)  # Check every 500ms
        
        print(f"❌ Timeout waiting for health bars after {timeout}s")
        return False


# Test the health detector
if __name__ == "__main__":
    import cv2
    import time
    
    WINDOW_TITLE = "Bloody Roar II (USA) [PlayStation] - BizHawk"
    
    # Initialize components
    capture = WindowCapture(WINDOW_TITLE)
    health_detector = HealthDetector()
    
    if not capture.is_valid:
        print(f"Window '{WINDOW_TITLE}' not found!")
        exit()
    
    print("Health Detection Test")
    print("Press 'q' to quit")
    print("-" * 40)
    
    cv2.namedWindow('Health Detection', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Health Detection', 800, 600)
    
    while True:
        # Capture frame
        frame = capture.capture()
        if frame is None:
            print("Failed to capture frame")
            time.sleep(0.1)
            continue
        
        # Detect health
        health_state = health_detector.detect(capture)
        
        if health_state:
            # Print health values
            print(f"\rP1: {health_state.p1_health:5.1f}% | P2: {health_state.p2_health:5.1f}%", 
                  end='', flush=True)
            
            # Visualize
            display = health_detector.visualize_health_bars(frame, health_state)
            cv2.imshow('Health Detection', display)
        else:
            print("\rHealth detection failed", end='', flush=True)
            cv2.imshow('Health Detection', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cv2.destroyAllWindows()