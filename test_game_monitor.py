# test_game_monitor.py
import cv2
import numpy as np
from game_state_monitor import GameStateMonitor

WINDOW_TITLE = "Bloody Roar II (USA) [PlayStation] - BizHawk"

def main():
    # Initialize monitor
    try:
        monitor = GameStateMonitor(WINDOW_TITLE)
    except RuntimeError as e:
        print(f"Error: {e}")
        return
        
    print("Game State Monitor Test")
    print("Press 'q' to quit")
    print("-" * 50)
    
    # Create visualization windows
    cv2.namedWindow('Game View', cv2.WINDOW_NORMAL)
    cv2.namedWindow('Observation Vector', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Game View', 800, 600)
    cv2.resizeWindow('Observation Vector', 600, 200)
    
    # FPS tracking
    fps_time = 0
    fps_counter = 0
    fps = 0
    
    while True:
        # Capture state
        game_state = monitor.capture_state()
        
        if game_state:
            # Get normalized observation
            observation = monitor.get_normalized_observation(player_perspective=1)
            
            # Print state info
            print(f"\rP1: ({game_state.player1.x:.0f}, {game_state.player1.y:.0f}) "
                  f"HP:{game_state.player1.health:.1f}% | "
                  f"P2: ({game_state.player2.x:.0f}, {game_state.player2.y:.0f}) "
                  f"HP:{game_state.player2.health:.1f}% | "
                  f"Dist:{game_state.distance:.0f}", end='', flush=True)
            
            # Get frame for visualization
            frame = monitor.get_raw_frame()
            if frame is not None:
                # Add visualizations
                display = monitor.visualize_state(frame, game_state)
                
                # Add FPS counter
                fps_counter += 1
                current_time = cv2.getTickCount() / cv2.getTickFrequency()
                if current_time - fps_time > 1.0:
                    fps = fps_counter
                    fps_counter = 0
                    fps_time = current_time
                cv2.putText(display, f"FPS: {fps}", (10, 90), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                cv2.imshow('Game View', display)
            
            # Visualize observation vector
            if observation is not None:
                obs_viz = visualize_observation(observation)
                cv2.imshow('Observation Vector', obs_viz)
        else:
            print("\rNo valid game state detected", end='', flush=True)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cv2.destroyAllWindows()

def visualize_observation(observation: np.ndarray) -> np.ndarray:
    """Create visual representation of observation vector"""
    # Create canvas
    height = 200
    width = 600
    viz = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Labels for each observation value
    labels = [
        "Agent HP",      # 0
        "Agent X",       # 1
        "Agent Y",       # 2
        "Agent VelX",    # 3
        "Agent VelY",    # 4
        "Opp HP",        # 5
        "Opp RelX",      # 6
        "Opp RelY",      # 7
        "Opp VelX",      # 8
        "Opp VelY",      # 9
        "Distance",      # 10
        "Facing"         # 11
    ]
    
    # Draw each observation value as a bar
    bar_width = width // len(observation)
    bar_spacing = 5
    
    for i, (val, label) in enumerate(zip(observation, labels)):
        x = i * bar_width + bar_spacing
        
        # Normalize value to pixel height (center at height/2)
        center_y = height // 2
        bar_height = int(val * (height // 3))
        
        # Choose color based on value
        if val > 0:
            color = (0, 255, 0)  # Green for positive
            y1 = center_y
            y2 = center_y - bar_height
        else:
            color = (0, 0, 255)  # Red for negative
            y1 = center_y
            y2 = center_y - bar_height
        
        # Draw bar
        cv2.rectangle(viz, (x, y1), (x + bar_width - 2*bar_spacing, y2), color, -1)
        
        # Draw zero line
        cv2.line(viz, (0, center_y), (width, center_y), (128, 128, 128), 1)
        
        # Add label
        cv2.putText(viz, label[:6], (x, height - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        
        # Add value
        cv2.putText(viz, f"{val:.2f}", (x, 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
    
    return viz

if __name__ == "__main__":
    main()