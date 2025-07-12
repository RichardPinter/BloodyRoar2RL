#!/usr/bin/env python3
"""
PPO Slow RL Environment

PPO-specific implementation of the slow RL environment.
Returns flat observation vectors for traditional RL agents (PPO, A2C, etc.).

This is essentially the original SlowRLEnvironment moved into the new architecture.
"""

import numpy as np
from typing import Union
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.base_slow_rl_environment import BaseSlowRLEnvironment, SlowState, GameState


class PPOSlowRLEnvironment(BaseSlowRLEnvironment):
    """
    PPO-specific slow RL environment.
    
    Returns flat observation vectors for traditional RL agents.
    Maintains backward compatibility with existing PPO training code.
    """
    
    def __init__(self):
        super().__init__()
        print("🎯 PPO Slow RL Environment initialized (flat observation vectors)")
    
    def _observation_to_output(self, state: SlowState) -> np.ndarray:
        """Convert SlowState to flat numpy array for PPO agent"""
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
    
    def get_observation_space_size(self) -> int:
        """Get size of observation space for PPO agent"""
        return 11  # Features in _observation_to_output
    
    def _on_observation_collected(self, game_state: GameState):
        """
        Hook called when observation is collected.
        PPO doesn't need additional data collection (no screenshots).
        """
        pass  # PPO only needs the basic game state data


def test_ppo_environment():
    """Test the PPO-specific environment"""
    print("🧪 Testing PPO Slow RL Environment")
    print("=" * 60)
    print("Testing flat observation vector output for PPO agents...")
    
    env = PPOSlowRLEnvironment()
    
    try:
        print(f"✅ PPO Environment created successfully")
        print(f"   Observation space size: {env.get_observation_space_size()}")
        print(f"   Action space size: {env.get_action_space_size()}")
        print(f"   Actions: {env.get_actions()}")
        
        print("\n📊 Testing without game connection (dry run)...")
        
        # Test observation output format
        # Create dummy SlowState to test output format
        from core.base_slow_rl_environment import SlowState, SlowObservation
        import time
        
        dummy_obs = SlowObservation(
            timestamp=time.time(),
            p1_health=100.0,
            p2_health=95.0,
            p1_position=(100, 200),
            p2_position=(300, 200),
            distance=200.0,
            frame_count=1000
        )
        
        dummy_state = SlowState(
            observations=[dummy_obs],
            p1_health_start=100.0,
            p1_health_end=95.0,
            p1_health_delta=-5.0,
            p2_health_start=95.0,
            p2_health_end=90.0,
            p2_health_delta=-5.0,
            p1_movement_distance=50.0,
            p2_movement_distance=30.0,
            p1_net_displacement=40.0,
            p2_net_displacement=25.0,
            average_distance=200.0,
            observation_window_duration=4.0
        )
        
        # Test output format
        output = env._observation_to_output(dummy_state)
        print(f"   Output type: {type(output)}")
        print(f"   Output shape: {output.shape}")
        print(f"   Output dtype: {output.dtype}")
        print(f"   Output values: {output}")
        
        # Verify output format
        assert isinstance(output, np.ndarray), "Output should be numpy array"
        assert output.shape == (11,), f"Output shape should be (11,), got {output.shape}"
        assert output.dtype == np.float32, f"Output dtype should be float32, got {output.dtype}"
        
        print("\n✅ PPO Environment test passed!")
        print("   - Correct flat vector output")
        print("   - Compatible with existing PPO agents")
        print("   - Maintains backward compatibility")
        
        return True
        
    except Exception as e:
        print(f"❌ PPO Environment test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        env.close()


if __name__ == "__main__":
    test_ppo_environment()