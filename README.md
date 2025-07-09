# Bloody Roar 2 Reinforcement Learning System

A real-time game state capture and reinforcement learning system for Bloody Roar 2 running on BizHawk emulator.

## 🎮 Overview

This project implements a complete pipeline for training RL agents to play Bloody Roar 2:
- Real-time game state capture from BizHawk emulator
- Health bar detection using computer vision
- Fighter position tracking using YOLO
- Normalized state representation for neural networks
- Action execution through BizHawk's Lua scripting

## 📁 Project Structure

```
bro2_rl/
├── Core Systems
│   ├── window_capture.py      # Efficient window/region capture from BizHawk
│   ├── health_detector.py      # Yellow health bar detection (color masking)
│   ├── fighter_detector.py     # YOLO-based fighter position tracking
│   └── game_controller.py      # File-based action execution system
│
├── State Management
│   ├── game_state.py          # Unified game state representation
│   ├── state_normalizer.py    # Converts state to RL-ready observations
│   ├── state_history.py       # Tracks history, calculates velocities
│   └── game_state_monitor.py  # Main orchestrator combining all systems
│
├── Testing
│   ├── test_game_monitor.py   # Full system test with visualization
│   ├── test_state_structure.py # State representation tests
│   └── test_window_capture.py  # Window capture tests
│
└── notebooks/                 # Jupyter notebooks for analysis
```

## 🚀 Current Implementation

### 1. Window Capture System
- Captures game frames from BizHawk emulator window
- Supports full window and region capture
- Handles window finding and error cases
- ~30+ FPS capture rate

### 2. Health Detection
- Detects yellow health bar pixels using BGR color masking
- Range: BGR(0,160,190) to BGR(20,180,220)
- Calculates health percentage (0-100%)
- Positions: P1 at x=505, P2 at x=1421, y=155

### 3. Fighter Detection
- Uses YOLOv8n for person detection
- Filters UI elements (y > 100)
- Tracks P1/P2 based on screen position
- Calculates distance between fighters

### 4. State Representation
**12-value observation vector** normalized to [-1, 1]:
```
[0-4]  Agent (P1):     health, x, y, velocity_x, velocity_y
[5-9]  Opponent (P2):  health, relative_x, relative_y, velocity_x, velocity_y
[10]   Distance:       Normalized distance between fighters
[11]   Facing:         Direction (1 or -1)
```

### 5. Action System
File-based communication with BizHawk Lua script:
- Actions: "punch", "kick", "left", "right", "up", "down", etc.
- 300ms delay between actions
- Writes to: `C:\Users\richa\Desktop\Personal\Uni\ShenLong\actions.txt`

## 🎯 Setup Instructions

### Prerequisites
1. **BizHawk Emulator** with Bloody Roar 2 ROM
2. **Python 3.8+** with packages:
   ```bash
   pip install numpy opencv-python ultralytics
   ```
3. **YOLO Model**: Download yolov8n.pt to project directory

### BizHawk Lua Script
Create a Lua script in BizHawk that reads from `actions.txt`:
```lua
local actions_file = "C:\\Users\\richa\\Desktop\\Personal\\Uni\\ShenLong\\actions.txt"

while true do
    local file = io.open(actions_file, "r")
    if file then
        local action = file:read("*all")
        file:close()
        
        -- Map actions to button presses
        if action == "punch" then
            joypad.set({["P1 Square"] = true}, 1)
        elseif action == "kick" then
            joypad.set({["P1 X"] = true}, 1)
        -- Add more mappings...
        end
        
        -- Clear the file
        file = io.open(actions_file, "w")
        file:write("")
        file:close()
    end
    emu.frameadvance()
end
```

### Running the System
1. Start BizHawk with Bloody Roar 2
2. Load the Lua script in BizHawk
3. Run the test:
   ```bash
   python test_game_monitor.py
   ```

## 📊 Testing the System

The test script shows:
- **Game View**: Live game with health bars and fighter detection overlays
- **Observation Vector**: Bar graph of the 12 normalized values
- **Console Output**: Real-time positions, health, and distance

Expected output:
```
P1: (500, 400) HP:85.0% | P2: (900, 420) HP:70.0% | Dist:400
```

## 🔧 Configuration

### Health Detection (`health_detector.py`)
```python
HealthBarConfig:
    p1_x: 505           # P1 health bar X position
    p2_x: 1421          # P2 health bar X position  
    bar_length: 400     # Health bar width
    bar_y: 155          # Y position
    health_drop_per_pixel: 0.25  # % health per pixel
```

### Fighter Detection (`fighter_detector.py`)
```python
FighterDetector:
    confidence_threshold: 0.3  # YOLO confidence
    min_y_position: 100       # Filter UI elements
```

## 🚧 Next Steps

### 1. Create Gym Environment (`br2_env.py`)
```python
class BR2Environment(gym.Env):
    def __init__(self):
        self.observation_space = spaces.Box(-1, 1, shape=(12,))
        self.action_space = spaces.Discrete(10)  # punch, kick, move, etc.
        
    def step(self, action):
        # Execute action
        # Capture new state
        # Calculate reward
        # Return obs, reward, done, info
        
    def reset(self):
        # Reset match
        # Return initial observation
```

### 2. Implement Reward Function
```python
def calculate_reward(prev_state, curr_state):
    # Damage dealt (+10 per % damage)
    damage_dealt = prev_state.player2.health - curr_state.player2.health
    
    # Damage taken (-10 per % damage)
    damage_taken = prev_state.player1.health - curr_state.player1.health
    
    # Time penalty (-0.1 per frame)
    # Win bonus (+100)
    # Distance reward (optional)
    
    return reward
```

### 3. Train RL Agent
Using Stable Baselines3:
```python
from stable_baselines3 import PPO

env = BR2Environment()
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=1000000)
```

### 4. Advanced Features
- **Combo Detection**: Track action sequences for combo rewards
- **Frame Data Collection**: Learn optimal move timings
- **Character-Specific Models**: Different models per character
- **Self-Play**: Train against previous versions
- **Curriculum Learning**: Start with stationary opponent

### 5. Evaluation Metrics
- Win rate over time
- Average damage dealt/taken
- Action distribution analysis
- State visitation heatmaps

## 🐛 Troubleshooting

### "Window not found" Error
- Ensure BizHawk window title exactly matches: `"Bloody Roar II (USA) [PlayStation] - BizHawk"`
- Window must be visible (not minimized)

### Low FPS
- Reduce YOLO confidence threshold
- Skip frames (process every 2-3 frames)
- Use smaller YOLO model

### Detection Issues
- Adjust color ranges for health bars
- Modify YOLO confidence threshold
- Check Y position filter for fighters

## 📝 Contributing

1. Fork the repository
2. Create your feature branch
3. Test thoroughly with `test_game_monitor.py`
4. Submit pull request with clear description

## 📄 License

This project is for educational and research purposes only.