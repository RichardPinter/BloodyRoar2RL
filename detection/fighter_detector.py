# fighter_detector.py
import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict
from ultralytics import YOLO
import cv2

@dataclass
class Fighter:
    """Detected fighter information"""
    center: Tuple[int, int]  # Center position
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    confidence: float
    player_id: Optional[int] = None  # 1 or 2
    
@dataclass
class FighterDetection:
    """Result of fighter detection"""
    player1: Optional[Fighter] = None
    player2: Optional[Fighter] = None
    all_detections: List[Fighter] = None
    distance: Optional[float] = None
    
    def __post_init__(self):
        if self.all_detections is None:
            self.all_detections = []
        
        # Calculate distance if both players detected
        if self.player1 and self.player2:
            self.distance = abs(self.player1.center[0] - self.player2.center[0])

class FighterDetector:
    """Detects fighter positions using YOLO"""
    
    def __init__(self, model_path: str = 'yolov8n.pt', 
                 confidence_threshold: float = 0.3,
                 min_y_position: int = 100):
        """
        Initialize fighter detector
        
        Args:
            model_path: Path to YOLO model
            confidence_threshold: Minimum confidence for detections
            min_y_position: Minimum Y position (to filter UI elements)
        """
        print(f"Loading YOLO model: {model_path}")
        self.model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold
        self.min_y_position = min_y_position
        
        # Track last known positions for stability
        self.last_p1_pos: Optional[Tuple[int, int]] = None
        self.last_p2_pos: Optional[Tuple[int, int]] = None
        
    def detect(self, image: np.ndarray) -> FighterDetection:
        """
        Detect fighters in image
        
        Args:
            image: BGR image array
            
        Returns:
            FighterDetection with player positions
        """
        # Run YOLO detection (class 0 = person)
        results = self.model(image, 
                           classes=[0], 
                           conf=self.confidence_threshold,
                           verbose=False)
        
        fighters = []
        
        # Process detections
        for result in results:
            if result.boxes is not None:
                for box in result.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    center_x = int((x1 + x2) / 2)
                    center_y = int((y1 + y2) / 2)
                    
                    # Skip if too high (likely UI element)
                    if center_y < self.min_y_position:
                        continue
                    
                    fighter = Fighter(
                        center=(center_x, center_y),
                        bbox=(int(x1), int(y1), int(x2), int(y2)),
                        confidence=box.conf[0].item()
                    )
                    fighters.append(fighter)
        
        # Sort by x position (leftmost first)
        fighters.sort(key=lambda f: f.center[0])
        
        # Assign to players
        detection = FighterDetection(all_detections=fighters)
        
        if len(fighters) >= 2:
            # Two or more detections - take leftmost as P1, next as P2
            detection.player1 = fighters[0]
            detection.player1.player_id = 1
            detection.player2 = fighters[1]
            detection.player2.player_id = 2
            
        elif len(fighters) == 1:
            # One detection - use position history to determine player
            fighter = fighters[0]
            if self._is_player1(fighter.center):
                detection.player1 = fighter
                detection.player1.player_id = 1
            else:
                detection.player2 = fighter
                detection.player2.player_id = 2
        
        # Update position history
        if detection.player1:
            self.last_p1_pos = detection.player1.center
        if detection.player2:
            self.last_p2_pos = detection.player2.center
            
        return detection
    
    def _is_player1(self, position: Tuple[int, int]) -> bool:
        """Determine if position belongs to player 1 based on history"""
        if not self.last_p1_pos or not self.last_p2_pos:
            # No history - use screen position (left half = P1)
            return position[0] < 960  # Assuming 1920 width
            
        # Compare distances to last known positions
        dist_to_p1 = abs(position[0] - self.last_p1_pos[0])
        dist_to_p2 = abs(position[0] - self.last_p2_pos[0])
        
        return dist_to_p1 < dist_to_p2
    
    def visualize_fighters(self, image: np.ndarray, 
                          detection: FighterDetection) -> np.ndarray:
        """Add fighter visualization to image"""
        display = image.copy()
        
        # Draw all detections in gray
        for fighter in detection.all_detections:
            x1, y1, x2, y2 = fighter.bbox
            cv2.rectangle(display, (x1, y1), (x2, y2), (128, 128, 128), 1)
            cv2.putText(display, f"{fighter.confidence:.2f}",
                       (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX,
                       0.4, (128, 128, 128), 1)
        
        # Draw P1 in green
        if detection.player1:
            x, y = detection.player1.center
            cv2.circle(display, (x, y), 10, (0, 255, 0), -1)
            cv2.putText(display, "P1", (x - 20, y - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Draw P2 in red  
        if detection.player2:
            x, y = detection.player2.center
            cv2.circle(display, (x, y), 10, (0, 0, 255), -1)
            cv2.putText(display, "P2", (x - 20, y - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Draw distance
        if detection.distance:
            cv2.putText(display, f"Distance: {detection.distance:.0f}px",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                       1, (255, 255, 0), 2)
        
        return display


# Test the fighter detector
if __name__ == "__main__":
    WINDOW_TITLE = "Bloody Roar II (USA) [PlayStation] - BizHawk"
    
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from detection.window_capture import WindowCapture
    
    capture = WindowCapture(WINDOW_TITLE)
    fighter_detector = FighterDetector()
    
    if not capture.is_valid:
        print(f"Window '{WINDOW_TITLE}' not found!")
        exit()
        
    print("Fighter Detection Test")
    print("Press 'q' to quit")
    print("-" * 40)
    
    cv2.namedWindow('Fighter Detection', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Fighter Detection', 800, 600)
    
    while True:
        # Capture frame
        frame = capture.capture()
        if frame is None:
            continue
            
        # Detect fighters
        detection = fighter_detector.detect(frame)
        
        # Print info
        p1_str = f"P1: {detection.player1.center}" if detection.player1 else "P1: ---"
        p2_str = f"P2: {detection.player2.center}" if detection.player2 else "P2: ---"
        dist_str = f"Dist: {detection.distance:.0f}px" if detection.distance else "Dist: ---"
        
        print(f"\r{p1_str} | {p2_str} | {dist_str}", end='', flush=True)
        
        # Visualize
        display = fighter_detector.visualize_fighters(frame, detection)
        cv2.imshow('Fighter Detection', display)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cv2.destroyAllWindows()