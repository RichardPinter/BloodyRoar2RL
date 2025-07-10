import time
import numpy as np
from typing import Optional, Dict, Any
from enum import Enum

from window_capture import WindowCapture
from health_detector import HealthDetector
from game_controller import BizHawkController

class GameScreen(Enum):
    """Different screens/states in the game"""
    UNKNOWN = "unknown"
    MENU = "menu"
    CHARACTER_SELECT = "character_select"
    LOADING = "loading"
    COMBAT = "combat"
    VICTORY = "victory"
    GAME_OVER = "game_over"

class ArcadeEnvironment:
    """
    Simplified utility class for arcade mode functionality.
    Handles health bar detection and fast-forwarding through non-combat states.
    
    This is NOT a gym environment - it's just a collection of utility functions
    for the training script to use alongside ArcadeEpisode and RoundSubEpisode.
    """
    
    def __init__(self, window_title: str = "Bloody Roar II (USA) [PlayStation] - BizHawk"):
        self.window_title = window_title
        
        # Initialize pixel-based health detection
        try:
            self.capture = WindowCapture(window_title)
            self.health_detector = HealthDetector()
            self.health_detection_available = True
        except Exception as e:
            print(f"Warning: Could not initialize health detection: {e}")
            self.health_detection_available = False
        
        # Initialize controller (for fast-forwarding)
        try:
            self.controller = BizHawkController()
            self.controller_available = True
        except Exception as e:
            print(f"Warning: Could not initialize controller: {e}")
            self.controller_available = False
        
        # State tracking
        self.current_screen = GameScreen.UNKNOWN
        self.last_health_check = 0
        self.health_check_interval = 0.1  # Check every 100ms
        
        print("ArcadeEnvironment utility initialized")
    
    def health_bars_visible(self) -> bool:
        """
        Check if health bars are visible (indicates we're in combat)
        Uses pixel-based detection like RoundSubEpisode
        
        Returns:
            True if health bars are visible, False otherwise
        """
        if not self.health_detection_available:
            print("Warning: Health detection not available, assuming health bars visible")
            return True
        
        # Throttle health checks to avoid excessive computation
        current_time = time.time()
        if current_time - self.last_health_check < self.health_check_interval:
            return self.current_screen == GameScreen.COMBAT
        
        self.last_health_check = current_time
        
        try:
            # Use pixel-based health detection
            health_state = self.health_detector.detect(self.capture)
            
            if health_state is not None:
                # If we can detect valid health bars, we're in combat
                # Health bars are visible if we get reasonable health values
                p1_health = health_state.p1_health
                p2_health = health_state.p2_health
                
                if (p1_health >= 0 and p1_health <= 100 and 
                    p2_health >= 0 and p2_health <= 100):
                    
                    self.current_screen = GameScreen.COMBAT
                    return True
            
            # If we can't detect valid health bars, we're probably not in combat
            self.current_screen = GameScreen.MENU
            return False
            
        except Exception as e:
            print(f"Error checking health bars: {e}")
            # If there's an error, assume we're not in combat
            self.current_screen = GameScreen.UNKNOWN
            return False
    
    def fast_forward_frame(self):
        """
        Advance the game by one frame during non-combat states
        
        This is a placeholder implementation that can be enhanced later
        """
        if not self.controller_available:
            print("Warning: Controller not available, cannot fast-forward")
            return
        
        try:
            # Placeholder: For now, just send a neutral action to advance frame
            # In a real implementation, this could:
            # - Send specific menu navigation commands
            # - Skip loading screens
            # - Auto-select default options
            
            # Send a brief "nothing" action to advance frame
            self.controller.send_action("stop")
            time.sleep(0.016)  # ~1 frame at 60fps
            
        except Exception as e:
            print(f"Error fast-forwarding frame: {e}")
    
    def get_current_screen_state(self) -> GameScreen:
        """
        Get the current screen/game state
        
        Returns:
            Current game screen state
        """
        # Update screen state based on health bar visibility
        if self.health_bars_visible():
            self.current_screen = GameScreen.COMBAT
        else:
            # For now, assume non-combat is menu
            # This could be enhanced to detect specific screens
            self.current_screen = GameScreen.MENU
        
        return self.current_screen
    
    def wait_for_combat(self, timeout: float = 30.0) -> bool:
        """
        Wait until we're in combat state (health bars visible)
        
        Args:
            timeout: Maximum time to wait in seconds
            
        Returns:
            True if combat state reached, False if timeout
        """
        print("Waiting for combat state...")
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            if self.health_bars_visible():
                print("Combat state detected!")
                return True
            
            # Fast-forward while waiting
            self.fast_forward_frame()
            
            # Print status every 5 seconds
            elapsed = time.time() - start_time
            if elapsed > 0 and int(elapsed) % 5 == 0:
                print(f"Still waiting for combat... ({elapsed:.1f}s elapsed)")
        
        print(f"Timeout waiting for combat state after {timeout}s")
        return False
    
    def skip_to_combat(self, max_attempts: int = 1000) -> bool:
        """
        Skip through non-combat states until we reach combat
        
        Args:
            max_attempts: Maximum number of fast-forward attempts
            
        Returns:
            True if combat reached, False if max attempts exceeded
        """
        print("Skipping to combat...")
        
        for attempt in range(max_attempts):
            if self.health_bars_visible():
                print(f"Reached combat after {attempt} attempts")
                return True
            
            self.fast_forward_frame()
            
            # Print progress every 100 attempts
            if attempt > 0 and attempt % 100 == 0:
                print(f"Fast-forwarded {attempt} frames, still looking for combat...")
        
        print(f"Failed to reach combat after {max_attempts} attempts")
        return False
    
    def is_in_combat(self) -> bool:
        """
        Simple alias for health_bars_visible() for clarity
        
        Returns:
            True if currently in combat
        """
        return self.health_bars_visible()
    
    def get_debug_info(self) -> Dict[str, Any]:
        """
        Get debug information about the current state
        
        Returns:
            Dictionary with debug information
        """
        return {
            'health_detection_available': self.health_detection_available,
            'controller_available': self.controller_available,
            'current_screen': self.current_screen.value,
            'health_bars_visible': self.health_bars_visible(),
            'last_health_check': self.last_health_check,
        }
    
    def close(self):
        """Clean up resources"""
        # Nothing to clean up for now
        print("ArcadeEnvironment utility closed")

# Test function
if __name__ == "__main__":
    print("Testing ArcadeEnvironment utility...")
    
    arcade_env = ArcadeEnvironment()
    
    try:
        # Test health bar detection
        print(f"Health bars visible: {arcade_env.health_bars_visible()}")
        print(f"Current screen: {arcade_env.get_current_screen_state()}")
        
        # Test debug info
        debug_info = arcade_env.get_debug_info()
        print(f"Debug info: {debug_info}")
        
        # Test fast-forwarding (just a few frames)
        print("Testing fast-forward for 5 frames...")
        for i in range(5):
            arcade_env.fast_forward_frame()
            print(f"Frame {i+1} forwarded")
        
        # Test waiting for combat (short timeout)
        print("Testing wait for combat (5 second timeout)...")
        combat_reached = arcade_env.wait_for_combat(timeout=5.0)
        print(f"Combat reached: {combat_reached}")
        
    except Exception as e:
        print(f"Error during test: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        arcade_env.close()