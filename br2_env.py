# br2_env.py
import gym
from gym import spaces
import numpy as np
import time
from typing import Tuple, Dict, Any, Optional

from game_state_monitor import GameStateMonitor
from game_controller import BizHawkController
from game_state import GameState

class BR2Environment(gym.Env):
    """
    Gym Environment for Bloody Roar 2
    
    Observation: 12-value normalized vector [-1, 1]
    Action: 10 discrete actions (move, attack, etc.)
    Reward: Simple health-based (damage dealt +, damage taken -)
    """
    
    def __init__(self, window_title: str = "Bloody Roar II (USA) [PlayStation] - BizHawk"):
        super(BR2Environment, self).__init__()
        
        # Initialize game systems
        self.monitor = GameStateMonitor(window_title)
        self.controller = BizHawkController()
        
        # Define spaces
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(12,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(16)
        
        # Action mapping - More fighting moves!
        self.action_map = {
            0: None,           # No action
            1: "left",         # Move left
            2: "right",        # Move right
            3: "up",           # Jump/up
            4: "down",         # Crouch/down
            5: "punch",        # Light punch
            6: "kick",         # Light kick
            7: "heavy_punch",  # Heavy punch
            8: "heavy_kick",   # Heavy kick
            9: "block",        # Block/defend
            10: "grab",        # Grab/throw
            11: "jump_punch",  # Jump + punch
            12: "jump_kick",   # Jump + kick
            13: "crouch_punch", # Down + punch
            14: "crouch_kick", # Down + kick
            15: "beast"        # Beast transformation
        }
        
        # State tracking
        self.previous_state: Optional[GameState] = None
        self.current_state: Optional[GameState] = None
        self.episode_steps = 0
        self.max_episode_steps = 1800  # ~30 seconds at 60fps
        
        print("BR2 Environment initialized")
        print(f"Observation space: {self.observation_space}")
        print(f"Action space: {self.action_space}")
        
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        Execute one step in the environment
        
        Args:
            action: Integer action (0-9)
            
        Returns:
            observation, reward, done, info
        """
        # Execute action using appropriate controller method
        action_name = self.action_map.get(action)
        if action_name is not None:
            if action_name == "punch":
                self.controller.punch()
            elif action_name == "kick":
                self.controller.kick()  
            elif action_name == "heavy_punch":
                self.controller.heavy_punch()
            elif action_name == "heavy_kick":
                self.controller.heavy_kick()
            elif action_name == "grab":
                self.controller.grab()
            elif action_name == "jump_punch":
                self.controller.jump_punch()
            elif action_name == "jump_kick":
                self.controller.jump_kick()
            elif action_name == "crouch_punch":
                self.controller.crouch_punch()
            elif action_name == "crouch_kick":
                self.controller.crouch_kick()
            elif action_name == "beast":
                self.controller.beast()
            elif action_name == "block":
                self.controller.send_action("l2")  # Block button
            else:
                # Basic movements
                self.controller.send_action(action_name)
        
        # Small delay to let action take effect
        time.sleep(0.05)
        
        # Capture new state
        self.previous_state = self.current_state
        self.current_state = self.monitor.capture_state()
        
        # Get normalized observation
        observation = self.monitor.get_normalized_observation(player_perspective=1)
        if observation is None:
            observation = np.zeros(12, dtype=np.float32)
        
        # Calculate reward
        reward = 0.0
        if self.previous_state is not None and self.current_state is not None:
            reward = self.calculate_reward(self.previous_state, self.current_state)
        
        # Check if episode is done
        done = self.is_done()
        
        # Increment step counter
        self.episode_steps += 1
        
        # Create info dict
        info = {
            'episode_steps': self.episode_steps,
            'current_state': self.current_state,
            'action_executed': self.action_map.get(action, 'none'),
        }
        
        if self.current_state:
            info.update({
                'p1_health': self.current_state.player1.health,
                'p2_health': self.current_state.player2.health,
                'distance': self.current_state.distance,
            })
        
        return observation, reward, done, info
    
    def reset(self) -> np.ndarray:
        """
        Reset the environment to start a new episode
        
        Returns:
            Initial observation
        """
        print("Resetting environment...")
        
        # Reset step counter
        self.episode_steps = 0
        
        # Clear action queue
        self.controller.clear_file()
        
        # Wait a moment for game to stabilize
        time.sleep(0.5)
        
        # Capture initial state
        attempts = 0
        while attempts < 10:
            self.current_state = self.monitor.capture_state()
            if self.current_state is not None:
                break
            time.sleep(0.1)
            attempts += 1
        
        self.previous_state = None
        
        # Get initial observation
        observation = self.monitor.get_normalized_observation(player_perspective=1)
        if observation is None:
            observation = np.zeros(12, dtype=np.float32)
        
        print(f"Environment reset. Initial health - P1: {self.current_state.player1.health:.1f}%, P2: {self.current_state.player2.health:.1f}%")
        
        return observation
    
    def calculate_reward(self, prev_state: GameState, curr_state: GameState) -> float:
        """
        Calculate reward based on health changes
        
        Simple reward function:
        - Own health increases: good (+)
        - Own health decreases: bad (-)
        - Enemy health increases: bad (-)
        - Enemy health decreases: good (+)
        
        Args:
            prev_state: Previous game state
            curr_state: Current game state
            
        Returns:
            Reward value
        """
        # Calculate health changes
        own_health_change = curr_state.player1.health - prev_state.player1.health
        enemy_health_change = curr_state.player2.health - prev_state.player2.health
        
        # Simple reward: own_health_change - enemy_health_change
        # If I lose 10 health: -10
        # If enemy loses 10 health: +10 
        # If I gain 5 health: +5
        # If enemy gains 5 health: -5
        reward = own_health_change - enemy_health_change
        
        return reward
    
    def is_done(self) -> bool:
        """
        Check if episode should end
        
        Returns:
            True if episode is over
        """
        # End if max steps reached
        if self.episode_steps >= self.max_episode_steps:
            return True
        
        # End if no valid state
        if self.current_state is None:
            return True
        
        # End if either player's health is very low (round is over)
        if (self.current_state.player1.health <= 5 or 
            self.current_state.player2.health <= 5):
            return True
        
        return False
    
    def render(self, mode='human'):
        """
        Render the environment
        For now, just print current state info
        """
        if self.current_state:
            print(f"Step {self.episode_steps}: "
                  f"P1 HP: {self.current_state.player1.health:.1f}% "
                  f"P2 HP: {self.current_state.player2.health:.1f}% "
                  f"Distance: {self.current_state.distance:.0f}")
    
    def close(self):
        """Clean up resources"""
        self.controller.clear_file()
        print("Environment closed")
    
    def get_action_meanings(self):
        """Return the meaning of each action"""
        return [self.action_map[i] for i in range(self.action_space.n)]