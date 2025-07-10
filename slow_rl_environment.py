#!/usr/bin/env python3
"""
Slow RL Environment for Performance-Constrained Systems

Uses 1-second sampling intervals and 4-second observation windows to work
with slow YOLO detection. Designed for 10x slowed game speed.

Timeline:
- Sample data every 1 real second (10 game seconds)
- Collect 4 samples per action decision
- Make 1 action every 4 real seconds
- Calculate rewards based on health deltas and movement patterns
"""

import time
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime

from round_sub_episode import RoundStateMonitor, GameState, RoundOutcome
from game_controller import BizHawkController

@dataclass
class SlowObservation:
    """Single observation at 1-second interval"""
    timestamp: float
    p1_health: float
    p2_health: float
    p1_position: Optional[Tuple[int, int]]
    p2_position: Optional[Tuple[int, int]]
    distance: Optional[float]
    frame_count: int

@dataclass
class SlowState:
    """4-second observation window for RL decision"""
    observations: List[SlowObservation]  # 4 observations
    
    # Calculated features
    p1_health_start: float
    p1_health_end: float
    p1_health_delta: float
    
    p2_health_start: float
    p2_health_end: float
    p2_health_delta: float
    
    p1_movement_distance: float
    p2_movement_distance: float
    p1_net_displacement: float
    p2_net_displacement: float
    
    average_distance: float
    observation_window_duration: float

class SlowRLEnvironment:
    """
    RL Environment with 1-second sampling and 4-second decision intervals
    Designed to work with slow YOLO detection on slowed game
    """
    
    def __init__(self):
        print("Initializing Slow RL Environment...")
        
        # Initialize components
        self.round_monitor = RoundStateMonitor()
        self.controller = BizHawkController()
        
        # Actions for the environment
        self.actions = ['kick', 'punch', 'forward', 'back', 'jump']
        self.action_space_size = len(self.actions)
        
        # Timing configuration
        self.sampling_interval = 1.0  # 1 second between samples
        self.observation_window = 4   # 4 samples per action
        
        # State tracking
        self.current_observations: List[SlowObservation] = []
        self.episode_step = 0
        self.total_reward = 0
        
        print(f"Action space: {self.action_space_size} actions")
        print(f"Actions: {self.actions}")
        print(f"Sampling: Every {self.sampling_interval}s")
        print(f"Decision: Every {self.observation_window}s")
    
    def reset(self) -> np.ndarray:
        """Reset environment for new episode"""
        print("🔄 Resetting Slow RL environment...")
        
        # Reset monitor
        self.round_monitor.reset()
        time.sleep(0.5)  # Let game stabilize
        
        # Clear observation history
        self.current_observations = []
        self.episode_step = 0
        self.total_reward = 0
        
        # Collect initial 4-second observation window
        initial_state = self._collect_observation_window()
        
        print(f"Reset complete. Initial P1: {initial_state.p1_health_start:.1f}% P2: {initial_state.p2_health_start:.1f}%")
        
        return self._state_to_array(initial_state)
    
    def step(self, action_index: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        Take an RL step:
        1. Execute action
        2. Collect 4 seconds of observations (4 samples)
        3. Calculate reward from state changes
        4. Return new state
        """
        self.episode_step += 1
        step_start_time = time.time()
        
        # Get state before action
        state_before = self._get_latest_slow_state() if self.current_observations else None
        
        # Execute action
        action_name = self.actions[action_index]
        print(f"\n🎮 Step {self.episode_step}: Action '{action_name}' (index {action_index})")
        self._execute_action(action_name)
        
        # Collect 4-second observation window
        state_after = self._collect_observation_window()
        
        # Calculate timing
        step_duration = time.time() - step_start_time
        
        # Calculate reward
        reward = self._calculate_reward(state_before, state_after)
        self.total_reward += reward
        
        # Check if episode is done
        done = self._is_episode_done()
        
        # Create info dict
        info = {
            'episode_step': self.episode_step,
            'total_reward': self.total_reward,
            'action_name': action_name,
            'step_duration': step_duration,
            'p1_health_delta': state_after.p1_health_delta,
            'p2_health_delta': state_after.p2_health_delta,
            'p1_movement': state_after.p1_movement_distance,
            'p2_movement': state_after.p2_movement_distance,
        }
        
        # Print step summary
        print(f"  ⏱️  Step completed in {step_duration:.1f}s")
        print(f"  💔 Health changes: P1={state_after.p1_health_delta:+.1f}% P2={state_after.p2_health_delta:+.1f}%")
        print(f"  🏃 Movement: P1={state_after.p1_movement_distance:.1f}px P2={state_after.p2_movement_distance:.1f}px")
        print(f"  🎯 Reward: {reward:.3f} | Total: {self.total_reward:.3f} | Done: {done}")
        
        return self._state_to_array(state_after), reward, done, info
    
    def _collect_observation_window(self) -> SlowState:
        """Collect observations over the configured observation window"""
        print(f"📊 Collecting {self.observation_window}-second observation window...")
        
        observations = []
        
        for i in range(self.observation_window):
            # Wait for sampling interval
            if i > 0:  # Don't wait before first sample
                time.sleep(self.sampling_interval)
            
            # Collect observation
            obs = self._collect_single_observation()
            observations.append(obs)
            
            print(f"  Sample {i+1}/{self.observation_window}: P1={obs.p1_health:.1f}% P2={obs.p2_health:.1f}% "
                  f"Pos=({obs.p1_position}, {obs.p2_position})")
        
        # Store observations for next iteration
        self.current_observations = observations
        
        # Create SlowState with calculated features
        return self._create_slow_state(observations)
    
    def _collect_single_observation(self) -> SlowObservation:
        """Collect a single observation from game state"""
        game_state = self.round_monitor.get_current_state()
        
        return SlowObservation(
            timestamp=time.time(),
            p1_health=game_state.p1_health,
            p2_health=game_state.p2_health,
            p1_position=game_state.p1_position,
            p2_position=game_state.p2_position,
            distance=game_state.fighter_distance,
            frame_count=game_state.frame_count
        )
    
    def _create_slow_state(self, observations: List[SlowObservation]) -> SlowState:
        """Create SlowState from variable number of observations with calculated features"""
        if len(observations) < 2:
            raise ValueError(f"Need at least 2 observations, got {len(observations)}")
        
        first_obs = observations[0]
        last_obs = observations[-1]
        
        # Health deltas
        p1_health_delta = last_obs.p1_health - first_obs.p1_health
        p2_health_delta = last_obs.p2_health - first_obs.p2_health
        
        # Movement calculations
        p1_movement_dist, p1_net_disp = self._calculate_movement(
            [obs.p1_position for obs in observations]
        )
        p2_movement_dist, p2_net_disp = self._calculate_movement(
            [obs.p2_position for obs in observations]
        )
        
        # Average distance
        distances = [obs.distance for obs in observations if obs.distance is not None]
        avg_distance = np.mean(distances) if distances else 0.0
        
        # Time span
        time_span = last_obs.timestamp - first_obs.timestamp
        
        return SlowState(
            observations=observations,
            p1_health_start=first_obs.p1_health,
            p1_health_end=last_obs.p1_health,
            p1_health_delta=p1_health_delta,
            p2_health_start=first_obs.p2_health,
            p2_health_end=last_obs.p2_health,
            p2_health_delta=p2_health_delta,
            p1_movement_distance=p1_movement_dist,
            p2_movement_distance=p2_movement_dist,
            p1_net_displacement=p1_net_disp,
            p2_net_displacement=p2_net_disp,
            average_distance=avg_distance,
            observation_window_duration=time_span
        )
    
    def _calculate_movement(self, positions: List[Optional[Tuple[int, int]]]) -> Tuple[float, float]:
        """Calculate total movement distance and net displacement"""
        valid_positions = [pos for pos in positions if pos is not None]
        
        if len(valid_positions) < 2:
            return 0.0, 0.0
        
        # Total movement distance (sum of distances between consecutive points)
        total_distance = 0.0
        for i in range(1, len(valid_positions)):
            dx = valid_positions[i][0] - valid_positions[i-1][0]
            dy = valid_positions[i][1] - valid_positions[i-1][1]
            total_distance += np.sqrt(dx**2 + dy**2)
        
        # Net displacement (straight-line distance from start to end)
        start_pos = valid_positions[0]
        end_pos = valid_positions[-1]
        dx = end_pos[0] - start_pos[0]
        dy = end_pos[1] - start_pos[1]
        net_displacement = np.sqrt(dx**2 + dy**2)
        
        return total_distance, net_displacement
    
    def _get_latest_slow_state(self) -> Optional[SlowState]:
        """Get SlowState from current observations"""
        if len(self.current_observations) != 4:
            return None
        return self._create_slow_state(self.current_observations)
    
    def _execute_action(self, action_name: str):
        """Execute the specified action"""
        try:
            method = getattr(self.controller, action_name)
            method()
        except AttributeError:
            print(f"Warning: Action '{action_name}' not found, sending as raw command")
            self.controller.send_action(action_name)
    
    def _state_to_array(self, state: SlowState) -> np.ndarray:
        """Convert SlowState to numpy array for RL agent"""
        return np.array([
            state.p1_health_start,
            state.p1_health_end,
            state.p1_health_delta,
            state.p2_health_start,
            state.p2_health_end,
            state.p2_health_delta,
            state.p1_movement_distance,
            state.p2_movement_distance,
            state.p1_net_displacement,
            state.p2_net_displacement,
            state.average_distance
        ]).astype(np.float32)
    
    def _calculate_reward(self, state_before: Optional[SlowState], state_after: SlowState) -> float:
        """Calculate reward from state changes"""
        
        # Health-based rewards
        health_reward = state_after.p1_health_delta * 0.01 - state_after.p2_health_delta * 0.01
        
        # Movement reward (encourage activity but not excessive movement)
        movement_reward = min(state_after.p1_movement_distance * 0.001, 0.1)
        
        # Distance reward (encourage staying in fighting range)
        optimal_distance = 200.0
        distance_penalty = abs(state_after.average_distance - optimal_distance) * -0.001
        
        # Round outcome bonus
        outcome_reward = 0.0
        if self.round_monitor.current_state.round_outcome == RoundOutcome.PLAYER_WIN:
            outcome_reward = 10.0
        elif self.round_monitor.current_state.round_outcome == RoundOutcome.PLAYER_LOSS:
            outcome_reward = -10.0
        
        total_reward = health_reward + movement_reward + distance_penalty + outcome_reward
        return total_reward
    
    def _is_episode_done(self) -> bool:
        """Check if episode should end"""
        # End if round is finished
        if self.round_monitor.is_round_finished():
            return True
        
        # End if too many steps (prevent infinite episodes)
        if self.episode_step >= 50:  # 50 * 4 seconds = 200 seconds max
            return True
        
        return False
    
    def get_observation_space_size(self) -> int:
        """Get size of observation space for RL agent"""
        return 11  # Features in _state_to_array
    
    def get_action_space_size(self) -> int:
        """Get size of action space for RL agent"""
        return self.action_space_size
    
    def close(self):
        """Clean up environment"""
        self.round_monitor.close()
        print("Slow RL Environment closed")


def test_slow_environment():
    """Test the slow RL environment"""
    print("🧪 Testing Slow RL Environment")
    print("=" * 60)
    
    env = SlowRLEnvironment()
    
    try:
        # Reset environment
        obs = env.reset()
        print(f"Initial observation shape: {obs.shape}")
        print(f"Initial observation: {obs}")
        
        # Take a few actions
        for step_i in range(3):
            print(f"\n{'='*20} STEP {step_i + 1} {'='*20}")
            
            # Random action
            action = np.random.randint(0, env.get_action_space_size())
            
            # Take step (this will take ~4 seconds)
            obs, reward, done, info = env.step(action)
            
            print(f"\nStep Results:")
            print(f"  Action: {action} ({env.actions[action]})")
            print(f"  Reward: {reward:.3f}")
            print(f"  Done: {done}")
            print(f"  Info: {info}")
            
            if done:
                print("Episode finished!")
                break
        
        print(f"\nObservation space size: {env.get_observation_space_size()}")
        print(f"Action space size: {env.get_action_space_size()}")
        
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
    except Exception as e:
        print(f"Error during test: {e}")
        import traceback
        traceback.print_exc()
    finally:
        env.close()


if __name__ == "__main__":
    test_slow_environment()