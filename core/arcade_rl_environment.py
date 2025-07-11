#!/usr/bin/env python3
"""
Arcade RL Environment

Wraps MatchRLEnvironment to provide arcade/tournament mode training.
Agent must win 8 matches in a row to complete the arcade.
If they lose any match, the arcade run ends.

Key Features:
- Tracks progress through 8-match arcade
- Handles transitions between matches
- Provides arcade-level context and rewards
- Clear success/failure conditions
"""

import time
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.match_rl_environment import MatchRLEnvironment

@dataclass
class ArcadeState:
    """Arcade progress tracking"""
    arcade_match_number: int  # Current match (1-8)
    total_wins: int          # Matches won so far
    total_losses: int        # Should be 0 or arcade ends
    is_final_match: bool     # True if this is match 8
    arcade_active: bool      # False after 8 wins or 1 loss
    arcade_completed: bool   # True only if all 8 matches won

class ArcadeRLEnvironment:
    """
    Arcade mode environment for 8-match tournament training.
    Each episode is still one round, but with arcade context.
    """
    
    def __init__(self, matches_to_win: int = 8, match_transition_delay: float = 3.0):
        print("Initializing Arcade RL Environment...")
        
        # Arcade configuration
        self.matches_to_win = matches_to_win
        self.match_transition_delay = match_transition_delay
        
        # Initialize match environment
        self.match_env = MatchRLEnvironment()
        
        # Arcade state
        self.arcade_state = ArcadeState(
            arcade_match_number=0,
            total_wins=0,
            total_losses=0,
            is_final_match=False,
            arcade_active=False,
            arcade_completed=False
        )
        
        # Statistics
        self.arcade_attempts = 0
        self.successful_arcades = 0
        self.arcade_history: List[Dict[str, Any]] = []
        self.current_arcade_start_time = None
        
        print(f"Arcade RL Environment initialized:")
        print(f"  Arcade format: Win {matches_to_win} matches in a row")
        print(f"  Match transition delay: {match_transition_delay}s")
        print(f"  Failure condition: Lose any match")
    
    def reset(self) -> np.ndarray:
        """
        Reset environment for new round
        Starts new arcade if previous one ended
        """
        # Check if we need to start a new arcade
        if not self.arcade_state.arcade_active:
            self._start_new_arcade()
        
        # Check if we need to transition to next match
        if self.match_env.match_manager.is_match_finished():
            self._handle_match_completion()
            
            # If arcade ended, start new one
            if not self.arcade_state.arcade_active:
                self._start_new_arcade()
        
        # Reset for next round
        round_state = self.match_env.reset()
        
        # Create extended state with arcade context
        arcade_extended_state = self._extend_state_with_arcade(round_state)
        
        return arcade_extended_state
    
    def step(self, action_index: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        Take a step in the current round
        """
        # Execute step in match environment
        match_state, match_reward, round_done, match_info = self.match_env.step(action_index)
        
        # Extend state with arcade context
        arcade_state = self._extend_state_with_arcade(match_state)
        
        # Start with match reward
        total_reward = match_reward
        
        # Check for match completion
        if match_info.get('match_completed', False):
            match_winner = match_info.get('match_winner')
            
            if match_winner == 'PLAYER 1':
                # Player won the match
                self.arcade_state.total_wins += 1
                print(f"\n🏆 ARCADE: Match {self.arcade_state.arcade_match_number} WON!")
                print(f"   Progress: {self.arcade_state.total_wins}/{self.matches_to_win} wins")
                
                # Check for arcade completion
                if self.arcade_state.total_wins >= self.matches_to_win:
                    self.arcade_state.arcade_completed = True
                    self.arcade_state.arcade_active = False
                    total_reward += 100.0  # Arcade completion bonus
                    print(f"\n🎉 ARCADE COMPLETED! Won all {self.matches_to_win} matches!")
                    self._record_arcade_result(success=True)
            else:
                # Player lost the match
                self.arcade_state.total_losses += 1
                self.arcade_state.arcade_active = False
                total_reward -= 50.0  # Arcade failure penalty
                print(f"\n💀 ARCADE OVER! Lost match {self.arcade_state.arcade_match_number}")
                print(f"   Final record: {self.arcade_state.total_wins}-{self.arcade_state.total_losses}")
                self._record_arcade_result(success=False)
        
        # Add arcade info
        arcade_info = match_info.copy()
        arcade_info.update({
            'arcade_match_number': self.arcade_state.arcade_match_number,
            'arcade_wins': self.arcade_state.total_wins,
            'arcade_losses': self.arcade_state.total_losses,
            'arcade_active': self.arcade_state.arcade_active,
            'arcade_completed': self.arcade_state.arcade_completed,
            'is_final_match': self.arcade_state.is_final_match
        })
        
        return arcade_state, total_reward, round_done, arcade_info
    
    def _start_new_arcade(self):
        """Start a new arcade run"""
        self.arcade_attempts += 1
        self.current_arcade_start_time = time.time()
        
        print(f"\n{'='*60}")
        print(f"🕹️  STARTING ARCADE ATTEMPT #{self.arcade_attempts}")
        print(f"{'='*60}")
        print(f"Goal: Win {self.matches_to_win} matches in a row")
        print(f"Current success rate: {self._get_success_rate():.1f}%")
        
        # Reset arcade state
        self.arcade_state = ArcadeState(
            arcade_match_number=1,
            total_wins=0,
            total_losses=0,
            is_final_match=False,
            arcade_active=True,
            arcade_completed=False
        )
    
    def _handle_match_completion(self):
        """Handle transition between matches in arcade"""
        if not self.arcade_state.arcade_active:
            return
        
        # Get match result
        last_match = self.match_env.match_results[-1] if self.match_env.match_results else None
        
        if last_match and last_match['winner'] == 'PLAYER 1':
            # Prepare for next match
            if self.arcade_state.total_wins < self.matches_to_win - 1:
                print(f"\n⏳ Waiting {self.match_transition_delay}s for next opponent...")
                time.sleep(self.match_transition_delay)
                
                self.arcade_state.arcade_match_number += 1
                self.arcade_state.is_final_match = (
                    self.arcade_state.arcade_match_number == self.matches_to_win
                )
                
                print(f"\n🎮 ARCADE: Starting Match {self.arcade_state.arcade_match_number}")
                if self.arcade_state.is_final_match:
                    print("   🔥 FINAL MATCH! Win this to complete the arcade!")
    
    def _extend_state_with_arcade(self, match_state: np.ndarray) -> np.ndarray:
        """Add arcade context to state vector"""
        arcade_features = [
            float(self.arcade_state.arcade_match_number),  # Current match (1-8)
            float(self.arcade_state.total_wins),            # Wins so far
            float(self.arcade_state.is_final_match)         # 1.0 if final match
        ]
        
        return np.concatenate([match_state, arcade_features]).astype(np.float32)
    
    def _record_arcade_result(self, success: bool):
        """Record arcade attempt results"""
        duration = time.time() - self.current_arcade_start_time
        
        result = {
            'attempt': self.arcade_attempts,
            'success': success,
            'matches_won': self.arcade_state.total_wins,
            'duration': duration,
            'timestamp': datetime.now().isoformat()
        }
        
        self.arcade_history.append(result)
        
        if success:
            self.successful_arcades += 1
    
    def _get_success_rate(self) -> float:
        """Get arcade completion success rate"""
        if self.arcade_attempts == 0:
            return 0.0
        return (self.successful_arcades / self.arcade_attempts) * 100
    
    def get_observation_space_size(self) -> int:
        """Get size of observation space"""
        return self.match_env.get_observation_space_size() + 3  # +3 arcade features
    
    def get_action_space_size(self) -> int:
        """Get size of action space"""
        return self.match_env.get_action_space_size()
    
    def get_actions(self) -> List[str]:
        """Get list of available actions"""
        return self.match_env.get_actions()
    
    def print_arcade_summary(self):
        """Print comprehensive arcade statistics"""
        print("\n" + "="*80)
        print("🕹️  ARCADE TRAINING SUMMARY")
        print("="*80)
        
        print(f"\nOverall Statistics:")
        print(f"  Total Attempts: {self.arcade_attempts}")
        print(f"  Successful Completions: {self.successful_arcades}")
        print(f"  Success Rate: {self._get_success_rate():.1f}%")
        
        if self.arcade_history:
            # Recent performance
            recent = self.arcade_history[-10:]
            recent_successes = sum(1 for r in recent if r['success'])
            print(f"\nRecent Performance (last {len(recent)} attempts):")
            print(f"  Success Rate: {recent_successes/len(recent)*100:.1f}%")
            
            # Best run
            best_run = max(self.arcade_history, key=lambda x: x['matches_won'])
            print(f"\nBest Run:")
            print(f"  Matches Won: {best_run['matches_won']}/{self.matches_to_win}")
            
            # Average matches per attempt
            avg_matches = np.mean([r['matches_won'] for r in self.arcade_history])
            print(f"\nAverage Matches Won: {avg_matches:.1f}")
        
        print("="*80)
        
        # Also print match-level summary
        self.match_env.print_match_summary()
    
    def close(self):
        """Clean up environment"""
        self.match_env.close()
        print("Arcade RL Environment closed")


def test_arcade_environment():
    """Test the arcade environment"""
    print("🧪 Testing Arcade RL Environment")
    print("="*60)
    
    env = ArcadeRLEnvironment(matches_to_win=3, match_transition_delay=1.0)  # Shorter for testing
    
    try:
        # Test a few rounds
        for episode in range(10):
            print(f"\n{'='*20} EPISODE {episode + 1} {'='*20}")
            
            # Reset
            obs = env.reset()
            print(f"Observation shape: {obs.shape}")
            print(f"Arcade state: Match {int(obs[19])}, Wins {int(obs[20])}")
            
            # Take some actions
            for step in range(3):
                action = np.random.randint(0, env.get_action_space_size())
                obs, reward, done, info = env.step(action)
                
                print(f"  Step {step+1}: Reward {reward:.3f}, Done {done}")
                
                if info.get('match_completed'):
                    print(f"  Match result: {info.get('match_winner')}")
                    print(f"  Arcade status: {info.get('arcade_wins')}-{info.get('arcade_losses')}")
                
                if done:
                    break
        
        # Print summary
        env.print_arcade_summary()
        
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
    except Exception as e:
        print(f"Error during test: {e}")
        import traceback
        traceback.print_exc()
    finally:
        env.close()


if __name__ == "__main__":
    test_arcade_environment()