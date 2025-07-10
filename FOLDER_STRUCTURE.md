# Bloody Roar 2 RL Project - Folder Structure

## Overview
This project has been reorganized into a clean folder structure for better maintainability.

## Folder Structure

```
bro2_rl/
├── core/                    # Main training components
│   ├── slow_rl_environment.py    # RL environment with slow sampling
│   ├── rl_training_simple.py     # PPO training script
│   └── round_sub_episode.py      # Round state monitoring
│
├── detection/               # Computer vision & detection
│   ├── fighter_detector.py       # YOLO-based character detection
│   ├── health_detector.py        # Health bar detection
│   ├── window_capture.py         # Screen capture from emulator
│   ├── game_state.py            # Game state data structures
│   └── state_history.py         # State history management
│
├── control/                 # Game control interfaces
│   ├── game_controller.py        # Python → Lua controller
│   ├── controller_config.py      # Button mappings
│   └── bizhawk_controller.lua    # Lua script for BizHawk
│
├── utils/                   # Utilities & testing
│   ├── measure_detection_performance.py  # YOLO performance profiler
│   ├── test_frame_grouping.py           # Frame grouping test
│   └── test_window_capture.py           # Window capture test
│
├── old/                     # Obsolete files (can be deleted)
│   └── [21 old files]
│
├── models/                  # Saved RL model weights
├── notebooks/               # Jupyter notebooks for analysis
├── yolov8n.pt              # YOLO model file
├── actions.txt             # Controller communication file
└── README.md               # Main project documentation
```

## Running the Training

To start RL training:
```bash
cd core
python rl_training_simple.py
```

## Key Components

- **Environment**: `SlowRLEnvironment` - Samples game state every second, makes decisions every 4-10 seconds
- **Algorithm**: PPO (Proximal Policy Optimization) with PyTorch
- **Actions**: kick, punch, forward, back, jump (5 actions)
- **State**: 11 features including health, positions, movement

## Dependencies
- PyTorch
- NumPy
- OpenCV (cv2)
- MSS (screen capture)
- win32gui (Windows only)
- ultralytics (YOLO)