#!/usr/bin/env python3
"""
20-Frame RL Environment

Takes actions every 20 frames (~333ms), measures rewards from state changes.
Solves the action-reward timing issue with proper action execution time.
"""

import time
import numpy as np
from typing import Dict, Any, List, Tuple
from dataclasses import dataclass

from round_sub_episode import RoundStateMonitor, GameState, RoundOutcome
from game_controller import BizHawkController
from controller_config import get_all_actions, get_action_index

@dataclass
class RLState:
    """RL state representation"""
    # Health (2 values: P1, P2)
    health: np.ndarray  # [p1_health, p2_health] 0-100
    
    # Positions (4 values: P1_x, P1_y, P2_x, P2_y)  
    positions: np.ndarray  # [p1_x, p1_y, p2_x, p2_y]
    
    # Distance between fighters
    distance: float
    
    # Frame count
    frame: int

class RL20FrameEnvironment:
    """
    RL Environment that acts every 20 frames and measures cumulative rewards
    """
    
    def __init__(self):
        print("Initializing 20-Frame RL Environment...")
        
        # Initialize components
        self.round_monitor = RoundStateMonitor()
        self.controller = BizHawkController()
        
        # RL configuration - simplified to 2 actions
        self.actions = ['kick', 'punch']  # Only 2 actions for simplicity
        self.action_space_size = len(self.actions)
        self.frame_skip = 20  # Act every 20 frames (~333ms)
        
        # State tracking
        self.current_state = None
        self.last_state = None
        self.episode_step = 0
        self.total_reward = 0
        
        # Timing tracking
        self.step_durations = []
        self.expected_frame_time = 1/60  # 16.67ms per frame
        self.expected_step_time = self.frame_skip * self.expected_frame_time
        
        print(f"Action space: {self.action_space_size} actions")
        print(f"Actions: {self.actions}")
        print(f"Frame skip: {self.frame_skip} (~{self.frame_skip/60*1000:.0f}ms between actions)")
    
    def reset(self) -> np.ndarray:
        """Reset environment for new episode"""
        print("🔄 Resetting RL environment...")
        
        # Reset monitor
        self.round_monitor.reset()
        
        # Wait a moment for game to stabilize
        time.sleep(0.5)
        
        # Get initial state
        self.current_state = self._get_rl_state()
        self.last_state = self.current_state
        self.episode_step = 0
        self.total_reward = 0
        
        print(f"Reset complete. Initial state: P1={self.current_state.health[0]:.1f}% P2={self.current_state.health[1]:.1f}%")
        
        return self._state_to_array(self.current_state)
    
    def step(self, action_index: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        Take an RL step:
        1. Execute action
        2. Wait 20 frames (~333ms)
        3. Measure state change
        4. Calculate reward
        """
        self.episode_step += 1
        
        # Record start time
        step_start_time = time.time()
        
        # Store state before action
        state_before = self.current_state
        
        # Execute action
        action_name = self.actions[action_index]
        print(f"Step {self.episode_step}: Taking action '{action_name}' (index {action_index}) [t={0:.3f}s]")
        
        self._execute_action(action_name)
        
        # Wait 20 frames and collect states
        states_during = self._collect_states_for_frames(self.frame_skip, step_start_time)
        
        # Get final state after action
        state_after = self._get_rl_state()
        self.current_state = state_after
        
        # Record total step duration
        step_end_time = time.time()
        actual_duration = step_end_time - step_start_time
        self.step_durations.append(actual_duration)
        
        # Calculate reward from state change
        reward = self._calculate_reward(state_before, state_after, states_during)
        self.total_reward += reward
        
        # Check if episode is done
        done = self._is_episode_done()
        
        # Create info dict
        info = {
            'episode_step': self.episode_step,
            'total_reward': self.total_reward,
            'action_name': action_name,
            'p1_health': state_after.health[0],
            'p2_health': state_after.health[1],
            'distance': state_after.distance,
            'round_outcome': self.round_monitor.current_state.round_outcome.value,
        }
        
        # Show timing info
        expected_ms = self.expected_step_time * 1000
        actual_ms = actual_duration * 1000
        avg_duration = sum(self.step_durations) / len(self.step_durations)
        avg_ms = avg_duration * 1000
        
        print(f"  Reward: {reward:.3f} | P1: {state_after.health[0]:.1f}% | P2: {state_after.health[1]:.1f}% | Done: {done}")
        print(f"  Timing: {actual_ms:.0f}ms (expected {expected_ms:.0f}ms) | Avg: {avg_ms:.0f}ms")
        
        # Update for next step
        self.last_state = state_before
        
        return self._state_to_array(state_after), reward, done, info
    
    def _execute_action(self, action_name: str):
        """Execute the specified action"""
        try:
            method = getattr(self.controller, action_name)
            method()
        except AttributeError:
            print(f"Warning: Action '{action_name}' not found, sending as raw command")
            self.controller.send_action(action_name)
    
    def _collect_states_for_frames(self, num_frames: int, start_time: float) -> List[RLState]:
        """Collect states for the specified number of frames"""
        states = []
        
        for frame_i in range(num_frames):
            # Wait one frame (~16.7ms at 60fps)
            time.sleep(self.expected_frame_time)
            
            # Get current state
            rl_state = self._get_rl_state()
            states.append(rl_state)
            
            # Debug output for key frames only (reduce spam)
            elapsed = time.time() - start_time
            if frame_i == 0:
                print(f"    Frame +{frame_i+1}: P1={rl_state.health[0]:.1f}% P2={rl_state.health[1]:.1f}% [t={elapsed:.3f}s]")
            elif frame_i == num_frames // 2:  # Middle frame
                print(f"    Frame +{frame_i+1}: P1={rl_state.health[0]:.1f}% P2={rl_state.health[1]:.1f}% [t={elapsed:.3f}s] (mid)")
            elif frame_i == num_frames - 1:
                print(f"    Frame +{frame_i+1}: P1={rl_state.health[0]:.1f}% P2={rl_state.health[1]:.1f}% [t={elapsed:.3f}s] (final)")
        
        return states
    
    def _get_rl_state(self) -> RLState:
        """Get current RL state from round monitor"""
        game_state = self.round_monitor.get_current_state()
        
        # Extract health
        health = np.array([game_state.p1_health, game_state.p2_health], dtype=np.float32)
        
        # Extract positions (handle missing positions)
        if game_state.p1_position and game_state.p2_position:
            positions = np.array([
                game_state.p1_position[0], game_state.p1_position[1],  # P1 x, y
                game_state.p2_position[0], game_state.p2_position[1]   # P2 x, y  
            ], dtype=np.float32)
        else:
            # Default positions if detection fails
            positions = np.array([400.0, 300.0, 800.0, 300.0], dtype=np.float32)
        
        # Distance
        distance = game_state.fighter_distance if game_state.fighter_distance else 400.0
        
        return RLState(
            health=health,
            positions=positions,
            distance=distance,
            frame=game_state.frame_count
        )
    
    def _state_to_array(self, state: RLState) -> np.ndarray:
        """Convert RLState to numpy array for RL agent"""
        # Combine all state features into single array
        # [p1_health, p2_health, p1_x, p1_y, p2_x, p2_y, distance]
        return np.concatenate([
            state.health,           # 2 values
            state.positions,        # 4 values  
            [state.distance]        # 1 value
        ]).astype(np.float32)      # Total: 7 values
    
    def _calculate_reward(self, state_before: RLState, state_after: RLState, 
                         states_during: List[RLState]) -> float:
        """Calculate reward from state changes over 4 frames"""
        
        # Health change rewards
        p1_health_change = state_after.health[0] - state_before.health[0]
        p2_health_change = state_after.health[1] - state_before.health[1]
        
        # Reward for damaging opponent, penalty for taking damage
        health_reward = p1_health_change * 0.01 - p2_health_change * 0.01
        
        # Distance reward (encourage staying in fighting range)
        optimal_distance = 200.0  # Good fighting distance
        distance_penalty = abs(state_after.distance - optimal_distance) * -0.001
        
        # Bonus for winning/losing
        outcome_reward = 0.0
        if self.round_monitor.current_state.round_outcome == RoundOutcome.PLAYER_WIN:
            outcome_reward = 10.0
        elif self.round_monitor.current_state.round_outcome == RoundOutcome.PLAYER_LOSS:
            outcome_reward = -10.0
        
        total_reward = health_reward + distance_penalty + outcome_reward
        
        return total_reward
    
    def _is_episode_done(self) -> bool:
        """Check if episode should end"""
        # End if round is finished
        if self.round_monitor.is_round_finished():
            return True
        
        # End if too many steps (prevent infinite episodes)
        if self.episode_step >= 200:  # ~67 seconds at 20-frame steps
            return True
        
        return False
    
    def get_observation_space_size(self) -> int:
        """Get size of observation space for RL agent"""
        return 7  # [p1_health, p2_health, p1_x, p1_y, p2_x, p2_y, distance]
    
    def get_action_space_size(self) -> int:
        """Get size of action space for RL agent"""
        return self.action_space_size
    
    def close(self):
        """Clean up environment"""
        self.round_monitor.close()
        print("RL Environment closed")


def test_rl_environment():
    """Test the 20-frame RL environment"""
    print("🧪 Testing 20-Frame RL Environment")
    print("=" * 60)
    
    env = RL20FrameEnvironment()
    
    try:
        # Reset environment
        obs = env.reset()
        print(f"Initial observation shape: {obs.shape}")
        print(f"Initial observation: {obs}")
        
        # Take a few random actions
        for step_i in range(5):
            print(f"\n--- Step {step_i + 1} ---")
            
            # Random action
            action = np.random.randint(0, env.get_action_space_size())
            
            # Take step
            obs, reward, done, info = env.step(action)
            
            print(f"Action: {action} ({env.actions[action]})")
            print(f"Reward: {reward:.3f}")
            print(f"Done: {done}")
            print(f"Info: {info}")
            
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
    test_rl_environment()