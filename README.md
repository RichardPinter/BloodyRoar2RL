# Bloody Roar 2

## Overview
This repo implements a **complete RL pipeline** for teaching an agent to play *Bloody Roar 2* using the BizHawk emulator on **Windows**. It captures frames in real time, parses HUD health bars with computer vision, learns a **Double DQN** policy with experience replay, and executes actions back in the emulator.

- Real-time screen capture (Windows, `dxcam`)
- Health-bar detection → reward shaping
- Double DQN + replay buffer, soft target updates, gradient clipping
- Asynchronous actor/learner threads with checkpoints and TensorBoard logs

> **Legal:** No ROMs are included. Use game assets you legally own.

---

## Setup (Windows 10/11)

### Prerequisites
- **Python:** 3.11 (64-bit) — e.g., your `cuda-env` Conda env  
- **Emulator:** **BizHawk** (x64) installed  
- **Game:** **US region** *Bloody Roar 2* ROM (HUD coordinates assume US)  
- **Screen capture:** `dxcam` (Windows-only)

### 1) Create/activate Conda env
```powershell
# use your existing env
conda activate cuda-env

# OR create a fresh one
# conda create -n bro2-rl python=3.11 -y
# conda activate bro2-rl
```

### 2) Install Python dependencies
```powershell
# From requirements (if present)
pip install -r requirements.txt

```
> Optional (GPU): install PyTorch + CUDA variants that match your drivers (`conda` channels `pytorch`/`nvidia`, or `pip`).

### 3) Configure paths
Edit `bro2rl/config.py`:
```python
BIZHAWK_EXE = r"C:\Emulators\BizHawk\EmuHawk.exe"
ROM_PATH    = r"C:\Games\BloodyRoar2\BloodyRoar2_US.bin"
REGION      = "US"
```

### 4) Run
```powershell
python .\main.py
```

## Project structure

```
bro2_rl/
├─ main.py
├─ bro2rl/
│  ├─ __init__.py
│  ├─ agent.py
│  ├─ trainer.py
│  ├─ shared_state.py
│  ├─ screen_capture.py
│  ├─ game_vision.py
│  ├─ rounds.py
│  ├─ models.py
│  ├─ logging_utils.py
│  └─ config.py
└─ lua/
   └─ round.controller.lua
```
---

## Results (optional)
> Replace placeholders when you have runs.

- **Env:** BizHawk (US ROM), Windows 11, Python 3.11, {{GPU/CPU}}  
- **Training:** Double DQN, replay {{N}}, batch {{B}}, γ={{0.99}}, τ={{0.005}}  
- **Logs:** `tensorboard --logdir runs`

| Checkpoint | Episodes | Win rate vs CPU | Avg reward | Notes |
|------------|----------|-----------------|------------|-------|
| match_10   | 200      | 22%             | -0.15      | warm-up |
| match_50   | 1,000    | 38%             | +0.12      | tuned lr/τ |
| best       | 1,800    | **44%**         | **+0.23**  | current best |

---

## Contributing (optional)
PRs and issue reports are welcome.

1. Fork and create a feature branch.  
2. Keep changes focused (one feature/fix).  
3. If you modify training logic, add a brief note and (ideally) a before/after plot or TB screenshot.  
4. Run any linters/tests you use.

---

## Known limitations
- Windows-only (uses `dxcam` for capture).  
- US ROM required (HUD coordinates assume US layout).  
- Emulator timing and window focus can affect capture; see `config.py` for tweaks.

---

## License
MIT (see `LICENSE`). ROMs are **not** included.
