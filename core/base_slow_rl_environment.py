#!/usr/bin/env python3
"""
Base Slow RL Environment (Abstract)

Abstract base class for algorithm-specific slow RL environments.
Contains shared game logic, round detection, action execution, and reward calculation.
Algorithm-specific observation collection is implemented in concrete classes.

Architecture:
- BaseSlowRLEnvironment (abstract) - Game logic, timing, rewards
  ├── PPOSlowRLEnvironment - Returns flat observation vectors
  └── DQNSlowRLEnvironment - Returns (screenshots, health_history) tuples
"""

import time
import numpy as np
from typing import Dict, Any, List, Tuple, Optional, Union
from dataclasses import dataclass
from datetime import datetime
from abc import ABC, abstractmethod
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.round_sub_episode import RoundStateMonitor, GameState, RoundOutcome
from control.game_controller import BizHawkController


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
    observations: List[SlowObservation]  # Variable number of observations
    
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


class BaseSlowRLEnvironment(ABC):
    """
    Abstract base class for slow RL environments.
    
    Contains shared game logic:
    - Timing and sampling control
    - Game state monitoring and round detection
    - Action execution
    - Reward calculation
    - Episode management
    
    Subclasses implement:
    - _collect_observation() - How to collect state data
    - get_observation_space_size() - Size of observation space
    - _observation_to_output() - Convert internal state to agent output format
    """
    
    def __init__(self):
        print("Initializing Base Slow RL Environment...")
        
        # Timing configuration
        self.sampling_interval = 1.0  # 1 second between samples
        self.observation_window = 8   # 8 samples per action
        self.death_detection_ratio = 0.95  # 95% of observation window for death detection
        
        # Calculate death detection threshold based on observation window
        # Use 95% of observation window so round can end near end of collection
        death_detection_frames = max(2, int(self.observation_window * self.death_detection_ratio))
        
        # Initialize shared components
        self.round_monitor = RoundStateMonitor(zero_threshold=death_detection_frames)
        self.controller = BizHawkController()
        
        # Actions for the environment
        self.actions = [
            'left', 
            'right', 
            'jump', 
            'squat',
            'transform', 
            'kick',       
            'punch',         
            'special', 
            'block',        
            'throw',   
        ]
        self.action_space_size = len(self.actions)
        
        # State tracking
        self.current_observations: List[SlowObservation] = []
        self.episode_step = 0
        self.total_reward = 0
        
        print(f"Action space: {self.action_space_size} actions")
        print(f"Actions: {self.actions}")
        print(f"Sampling: Every {self.sampling_interval}s")
        print(f"Decision: Every {self.observation_window}s")
        print(f"Death detection: {death_detection_frames} frames ({death_detection_frames}s) of 0% health")
        print(f"Death detection ratio: {self.death_detection_ratio*100:.0f}% of observation window")
    
    def reset(self) -> Union[np.ndarray, Tuple]:
        """Reset environment for new episode"""
        print("🔄 Resetting Slow RL environment...")
        
        # Reset monitor
        self.round_monitor.reset()
        time.sleep(0.5)  # Let game stabilize
        
        # Wait for round to be ready (both players at full health)
        print("⏳ Waiting for round to start...")
        if self.round_monitor.wait_for_round_ready(timeout=30.0):
            print("🎬 Round started! Beginning training...")
        else:
            print("⚠️  Round not detected, starting anyway...")
        
        # Clear observation history
        self.current_observations = []
        self.episode_step = 0
        self.total_reward = 0
        
        # Collect initial observation window
        initial_state = self._collect_observation_window()
        
        print(f"Reset complete. Initial P1: {initial_state.p1_health_start:.1f}% P2: {initial_state.p2_health_start:.1f}%")
        
        return self._observation_to_output(initial_state)
    
    def step(self, action_index: int) -> Tuple[Union[np.ndarray, Tuple], float, bool, Dict[str, Any]]:
        """
        Take an RL step:
        1. Execute action
        2. Collect observation window
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
        
        # Collect observation window
        state_after = self._collect_observation_window()
        
        # Calculate timing
        step_duration = time.time() - step_start_time
        
        # Calculate reward
        reward = self._calculate_reward(state_before, state_after)
        self.total_reward += reward
        
        # Check if episode is done
        done = self._is_episode_done()
        
        # If episode is done, show winner analysis (for actual deaths)
        if done:
            self.round_monitor.print_winner_analysis()
        
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
        
        return self._observation_to_output(state_after), reward, done, info
    
    def _collect_observation_window(self) -> SlowState:
        """Collect observations over the configured observation window"""
        print(f"📊 Collecting {self.observation_window}-second observation window...")
        
        observations = []
        
        for i in range(self.observation_window):
            # Wait for sampling interval
            if i > 0:  # Don't wait before first sample
                time.sleep(self.sampling_interval)
            
            # Collect observation (algorithm-specific)
            obs = self._collect_single_observation()
            observations.append(obs)
            
            print(f"  Sample {i+1}/{self.observation_window}: P1={obs.p1_health:.1f}% P2={obs.p2_health:.1f}% "
                  f"Pos=({obs.p1_position}, {obs.p2_position})")
            
            # Only check for round end after we have minimum required observations (≥2)
            if len(observations) >= 2 and self.round_monitor.is_round_finished():
                winner = self.round_monitor.get_winner()
                print(f"  🏁 Round ended during collection! Winner: {winner}")
                print(f"  📊 Collected {len(observations)}/{self.observation_window} samples before round end")
                
                # Show detailed winner analysis for any round end
                self.round_monitor.print_winner_analysis()
                
                break
        
        # Store observations for next iteration
        self.current_observations = observations
        
        # Create SlowState with calculated features
        return self._create_slow_state(observations)
    
    def _collect_single_observation(self) -> SlowObservation:
        """Collect a single observation from game state (shared logic)"""
        game_state = self.round_monitor.get_current_state()
        
        # Let subclasses collect additional data if needed
        self._on_observation_collected(game_state)
        
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
        if len(self.current_observations) < 2:
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
    
    def _calculate_reward(self, state_before: Optional[SlowState], state_after: SlowState) -> float:
        """Calculate reward from state changes - HEALTH ONLY"""
        
        # P1 health change: positive if health increases, negative if decreases
        p1_health_reward = state_after.p1_health_delta * 0.1
        
        # P2 health change: negative if health increases, positive if decreases
        p2_health_reward = -state_after.p2_health_delta * 0.1
        
        # Total health-based reward
        health_reward = p1_health_reward + p2_health_reward
        
        # Round outcome bonus
        if self.round_monitor.current_state.round_outcome == RoundOutcome.PLAYER_WIN:
            health_reward += 10.0
        elif self.round_monitor.current_state.round_outcome == RoundOutcome.PLAYER_LOSS:
            health_reward -= 10.0
        
        return health_reward
    
    def _is_episode_done(self) -> bool:
        """Check if episode should end"""
        # Only end if round is finished (actual player death)
        if self.round_monitor.is_round_finished():
            return True
        
        # No artificial step or time limits - round continues until actual death
        return False
    
    def get_action_space_size(self) -> int:
        """Get size of action space for RL agent"""
        return self.action_space_size
    
    def get_actions(self) -> List[str]:
        """Get list of available actions"""
        return self.actions.copy()
    
    def close(self):
        """Clean up environment"""
        self.round_monitor.close()
        print("Base Slow RL Environment closed")
    
    # Abstract methods that subclasses must implement
    
    @abstractmethod
    def _observation_to_output(self, state: SlowState) -> Union[np.ndarray, Tuple]:
        """
        Convert internal SlowState to output format for RL agent.
        
        Returns:
            For PPO: np.ndarray (flat observation vector)
            For DQN: Tuple[np.ndarray, np.ndarray] (screenshots, health_history)
        """
        pass
    
    @abstractmethod
    def get_observation_space_size(self) -> int:
        """
        Get size of observation space for RL agent.
        
        Returns:
            For PPO: Size of flat vector
            For DQN: Not used (tuple output)
        """
        pass
    
    @abstractmethod
    def _on_observation_collected(self, game_state: GameState):
        """
        Hook called when a game state observation is collected.
        Allows subclasses to collect additional data (e.g., screenshots for DQN).
        
        Args:
            game_state: The current game state from round monitor
        """
        pass


def test_base_environment():
    """Test that the base environment contains shared logic correctly"""
    print("🧪 Testing Base Environment (Abstract)")
    print("=" * 60)
    print("Note: Cannot instantiate abstract class directly.")
    print("Testing that all shared components are properly defined...")
    
    # Test that the abstract class is properly defined
    try:
        # Check that it's abstract
        from abc import ABC
        assert issubclass(BaseSlowRLEnvironment, ABC), "Should be abstract"
        
        # Check that abstract methods exist
        abstract_methods = BaseSlowRLEnvironment.__abstractmethods__
        expected_methods = {'_observation_to_output', 'get_observation_space_size', '_on_observation_collected'}
        assert abstract_methods == expected_methods, f"Abstract methods mismatch: {abstract_methods}"
        
        # Test that we can't instantiate it
        try:
            BaseSlowRLEnvironment()
            assert False, "Should not be able to instantiate abstract class"
        except TypeError:
            pass  # Expected
        
        print("✅ Base environment abstract class properly defined!")
        print(f"   Abstract methods: {list(abstract_methods)}")
        print("   Shared logic: timing, actions, rewards, round monitoring")
        
        return True
        
    except Exception as e:
        print(f"❌ Base environment test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    test_base_environment()