#!/usr/bin/env python3
"""
Match RL Environment

Wraps SlowRLEnvironment with MatchManager to provide complete match training.
Handles best-of-3 matches with automatic round transitions while maintaining
RL training interface for individual rounds with match context.

Key Features:
- Each RL episode = one round (like SlowRLEnvironment)
- Automatic round transitions between episodes
- Match context added to observations
- Match-level statistics and progress tracking
- No draws possible (one player always wins)
"""

import time
import numpy as np
from typing import Dict, Any, List, Tuple, Optional, Union
from dataclasses import dataclass
from datetime import datetime
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.ppo_slow_rl_environment import PPOSlowRLEnvironment
from core.dqn_slow_rl_environment import DQNSlowRLEnvironment
from core.match_manager import MatchManager, MatchOutcome, RoundResult

@dataclass
class MatchRLState:
    """Extended state information including match context"""
    # Round-level state (from base environment)
    round_state: Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]
    
    # Match context
    current_round: int
    p1_rounds_won: int
    p2_rounds_won: int
    rounds_to_win: int
    max_rounds: int
    match_duration: float
    
    # Round context
    is_final_round_possible: bool  # Could this round end the match?
    match_point_p1: bool  # P1 can win match this round
    match_point_p2: bool  # P2 can win match this round

class MatchRLEnvironment:
    """
    RL Environment that manages complete matches using MatchManager.
    Trains on individual rounds but maintains match-level context and statistics.
    
    Supports both PPO and DQN agents through factory pattern.
    """
    
    def __init__(self, max_rounds: int = 3, rounds_to_win: int = 2, env_type: str = "ppo", img_size: Tuple[int, int] = (84, 84)):
        """
        Initialize match environment.
        
        Args:
            max_rounds: Maximum rounds per match (usually 3)
            rounds_to_win: Rounds needed to win match (usually 2)
            env_type: Type of environment ("ppo" or "dqn")
            img_size: Target size for screenshots (height, width) - only used for DQN
        """
        print("Initializing Match RL Environment...")
        
        # Store environment type and configuration
        self.env_type = env_type
        self.img_size = img_size
        
        # Initialize match manager
        self.match_manager = MatchManager(max_rounds=max_rounds, rounds_to_win=rounds_to_win)
        
        # Current round environment (created fresh for each round)
        self.current_round_env: Optional[Union[PPOSlowRLEnvironment, DQNSlowRLEnvironment]] = None
        
        # Match configuration
        self.max_rounds = max_rounds
        self.rounds_to_win = rounds_to_win
        
        # State tracking
        self.match_episode = 0
        self.total_rounds_played = 0
        self.current_match_active = False
        
        # Statistics
        self.match_results: List[Dict[str, Any]] = []
        
        print(f"Match RL Environment initialized:")
        print(f"  Environment type: {env_type.upper()}")
        print(f"  Match format: Best-of-{max_rounds} (first to {rounds_to_win})")
        print(f"  Episode scope: One round per episode with match context")
    
    def reset(self) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Reset environment for new round (or new match if previous match finished)
        
        Returns:
            For PPO: np.ndarray (flat observation + match features)
            For DQN: Tuple[np.ndarray, np.ndarray] (screenshots, health_history + match features)
        """
        # Check if we need to start a new match
        if not self.current_match_active or self.match_manager.is_match_finished():
            self._start_new_match()
        
        # Start next round
        self._start_next_round()
        
        # Get initial round state
        round_state = self.current_round_env.reset()
        
        # Create match state with context
        match_state = self._create_match_state(round_state)
        
        print(f"🎬 Round {match_state.current_round} ready!")
        print(f"   Match score: P1={match_state.p1_rounds_won} P2={match_state.p2_rounds_won}")
        if match_state.match_point_p1:
            print(f"   🔥 MATCH POINT for P1!")
        elif match_state.match_point_p2:
            print(f"   🔥 MATCH POINT for P2!")
        
        return self._match_state_to_output(match_state)
    
    def step(self, action_index: int) -> Tuple[Union[np.ndarray, Tuple[np.ndarray, np.ndarray]], float, bool, Dict[str, Any]]:
        """
        Take an RL step in the current round
        
        Returns:
            (state, reward, done, info) where done=True means round ended
        """
        if self.current_round_env is None:
            raise RuntimeError("Environment not reset. Call reset() first.")
        
        step_start_time = time.time()
        
        # Execute action in round environment
        round_state, round_reward, round_done, round_info = self.current_round_env.step(action_index)
        
        # Create match state with context
        match_state = self._create_match_state(round_state)
        
        # Initialize return values
        total_reward = round_reward  # Keep round-based rewards for now
        episode_done = round_done
        match_info = round_info.copy()
        
        # Add match context to info
        match_info.update({
            'match_episode': self.match_episode,
            'current_round': match_state.current_round,
            'match_score': f"{match_state.p1_rounds_won}-{match_state.p2_rounds_won}",
            'is_match_point': match_state.match_point_p1 or match_state.match_point_p2,
            'rounds_to_win': self.rounds_to_win,
            'max_rounds': self.max_rounds
        })
        
        # Handle round completion
        if round_done:
            round_result = self._complete_current_round()
            self.total_rounds_played += 1
            
            if round_result:
                print(f"\n🏁 Round {round_result.round_number} completed!")
                print(f"   Winner: {round_result.winner}")
                print(f"   Duration: {round_result.duration:.1f}s")
                
                match_info.update({
                    'round_completed': True,
                    'round_winner': round_result.winner,
                    'round_duration': round_result.duration,
                    'total_rounds_played': self.total_rounds_played
                })
                
                # Check if match is finished
                if self.match_manager.is_match_finished():
                    match_winner = self.match_manager.get_match_winner()
                    
                    print(f"\n🎉 MATCH FINISHED!")
                    print(f"   Winner: {match_winner}")
                    print(f"   Final Score: P1={self.match_manager.stats.p1_rounds_won} P2={self.match_manager.stats.p2_rounds_won}")
                    
                    # Store match result
                    self._store_match_result(match_winner)
                    self.current_match_active = False
                    
                    match_info.update({
                        'match_completed': True,
                        'match_winner': match_winner,
                        'match_duration': self.match_manager.stats.duration,
                        'final_score': {
                            'p1_rounds': self.match_manager.stats.p1_rounds_won,
                            'p2_rounds': self.match_manager.stats.p2_rounds_won
                        }
                    })
                else:
                    print(f"   Match continues... Score: P1={self.match_manager.stats.p1_rounds_won} P2={self.match_manager.stats.p2_rounds_won}")
        
        return self._match_state_to_output(match_state), total_reward, episode_done, match_info
    
    def _start_new_match(self):
        """Start a new match"""
        self.match_episode += 1
        print(f"\n🥊 Starting Match {self.match_episode}")
        print("=" * 60)
        
        # Start new match
        self.match_manager.start_match()
        self.current_match_active = True
    
    def _start_next_round(self):
        """Start the next round in the current match"""
        # Create environment using factory pattern
        if self.env_type == "ppo":
            self.current_round_env = PPOSlowRLEnvironment()
        elif self.env_type == "dqn":
            self.current_round_env = DQNSlowRLEnvironment(img_size=self.img_size)
        else:
            raise ValueError(f"Unsupported environment type: {self.env_type}")
        
        # Get death detection threshold from environment config
        death_threshold = self.current_round_env.observation_window * self.current_round_env.death_detection_ratio
        death_threshold = max(2, int(death_threshold))
        
        # Start next round
        if not self.match_manager.start_next_round(zero_threshold=death_threshold):
            raise RuntimeError("Failed to start round")
        
        # Environment already created above using factory pattern
        print(f"   Round {self.match_manager.current_round_number} environment ready ({self.env_type.upper()})")
    
    def _complete_current_round(self) -> Optional[RoundResult]:
        """Complete the current round using MatchManager"""
        # Get final state from the ACTUAL round monitor (in SlowRLEnvironment)
        final_state = self.current_round_env.round_monitor.current_state
        
        # Let MatchManager handle round completion
        return self.match_manager._complete_current_round(final_state)
    
    def _create_match_state(self, round_state: Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]) -> MatchRLState:
        """Create MatchRLState from current round state and match context"""
        stats = self.match_manager.stats
        current_round = self.match_manager.current_round_number
        
        # Determine match point situations
        match_point_p1 = (stats.p1_rounds_won == self.rounds_to_win - 1)
        match_point_p2 = (stats.p2_rounds_won == self.rounds_to_win - 1)
        
        # Check if this could be the final round
        is_final_round_possible = (
            match_point_p1 or match_point_p2 or 
            current_round >= self.max_rounds
        )
        
        return MatchRLState(
            round_state=round_state,
            current_round=current_round,
            p1_rounds_won=stats.p1_rounds_won,
            p2_rounds_won=stats.p2_rounds_won,
            rounds_to_win=self.rounds_to_win,
            max_rounds=self.max_rounds,
            match_duration=stats.duration,
            is_final_round_possible=is_final_round_possible,
            match_point_p1=match_point_p1,
            match_point_p2=match_point_p2
        )
    
    def _store_match_result(self, winner: Optional[str]):
        """Store completed match results"""
        match_result = {
            'match_episode': self.match_episode,
            'winner': winner,
            'duration': self.match_manager.stats.duration,
            'total_rounds': self.match_manager.stats.total_rounds,
            'p1_rounds_won': self.match_manager.stats.p1_rounds_won,
            'p2_rounds_won': self.match_manager.stats.p2_rounds_won,
            'outcome': self.match_manager.stats.match_outcome.value,
            'round_results': [
                {
                    'round': r.round_number,
                    'winner': r.winner,
                    'duration': r.duration,
                    'p1_health': r.final_p1_health,
                    'p2_health': r.final_p2_health
                }
                for r in self.match_manager.stats.round_results
            ]
        }
        
        self.match_results.append(match_result)
    
    def _match_state_to_output(self, match_state: MatchRLState) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Convert MatchRLState to output format for RL agent.
        
        Returns:
            For PPO: np.ndarray (flat features + match context)
            For DQN: Tuple[np.ndarray, np.ndarray] (screenshots, health_history + match context)
        """
        # Create match context features
        match_features = np.array([
            float(match_state.current_round),           # Current round number
            float(match_state.p1_rounds_won),           # P1 rounds won
            float(match_state.p2_rounds_won),           # P2 rounds won
            float(match_state.rounds_to_win),           # Rounds needed to win
            float(match_state.match_duration) / 60.0,   # Match duration in minutes
            float(match_state.is_final_round_possible), # 1.0 if final round possible
            float(match_state.match_point_p1),          # 1.0 if P1 match point
            float(match_state.match_point_p2),          # 1.0 if P2 match point
        ], dtype=np.float32)
        
        if self.env_type == "ppo":
            # PPO: Flat vector input, concatenate match features
            return np.concatenate([match_state.round_state, match_features]).astype(np.float32)
        else:
            # DQN: Tuple input (screenshots, health_history)
            screenshots, health_history = match_state.round_state
            
            # Extend health history with match features
            # Shape: (health_history_length, N) → (health_history_length, N + 8)
            extended_health = np.zeros((health_history.shape[0], health_history.shape[1] + 8), dtype=np.float32)
            extended_health[:, :health_history.shape[1]] = health_history  # Original health data
            
            # Add match features to each frame (broadcast)
            extended_health[:, health_history.shape[1]:] = match_features[np.newaxis, :]  # Broadcast to all frames
            
            return screenshots, extended_health
    
    def get_observation_space_size(self) -> int:
        """Get size of observation space for RL agent"""
        if self.env_type == "ppo":
            base_size = 11  # Round features from PPO environment
            return base_size + 8  # + 8 match features = 19 total
        else:
            # DQN uses tuple output, return base size for compatibility
            return self.current_round_env.get_observation_space_size() if self.current_round_env else 0
    
    def get_action_space_size(self) -> int:
        """Get size of action space for RL agent"""
        return self.current_round_env.get_action_space_size() if self.current_round_env else 10
    
    def get_actions(self) -> List[str]:
        """Get list of available actions"""
        # Return default action list if no round environment exists yet
        default_actions = [
            'left', 'right', 'jump', 'squat', 'transform', 
            'kick', 'punch', 'special', 'block', 'throw'
        ]
        return self.current_round_env.actions if self.current_round_env else default_actions
    
    def print_match_summary(self):
        """Print summary of all completed matches"""
        if not self.match_results:
            print("No completed matches yet")
            return
        
        print("\n" + "=" * 80)
        print("🏆 MATCH TRAINING SUMMARY")
        print("=" * 80)
        
        # Overall statistics
        total_matches = len(self.match_results)
        p1_wins = sum(1 for m in self.match_results if m['winner'] == 'PLAYER 1')
        p2_wins = sum(1 for m in self.match_results if m['winner'] == 'PLAYER 2')
        
        print(f"Total Matches: {total_matches}")
        print(f"P1 Wins: {p1_wins} ({p1_wins/total_matches*100:.1f}%)")
        print(f"P2 Wins: {p2_wins} ({p2_wins/total_matches*100:.1f}%)")
        print(f"Total Rounds Played: {self.total_rounds_played}")
        
        # Recent performance
        if total_matches >= 3:
            recent_matches = self.match_results[-3:]
            recent_p1_wins = sum(1 for m in recent_matches if m['winner'] == 'PLAYER 1')
            print(f"Recent P1 Win Rate (last 3): {recent_p1_wins}/3 ({recent_p1_wins/3*100:.1f}%)")
        
        # Average match duration
        avg_duration = np.mean([m['duration'] for m in self.match_results])
        print(f"Average Match Duration: {avg_duration:.1f}s")
        
        print("=" * 80)
    
    def close(self):
        """Clean up environment"""
        if self.current_round_env:
            self.current_round_env.close()
        self.match_manager.close()
        print("Match RL Environment closed")


def test_match_rl_environment():
    """Test the match RL environment"""
    print("🧪 Testing Match RL Environment")
    print("=" * 60)
    
    env = MatchRLEnvironment()
    
    try:
        # Test multiple rounds across matches
        print("Testing complete match with multiple rounds...")
        
        for episode in range(5):  # Test 5 episodes (rounds)
            print(f"\n{'='*20} EPISODE {episode + 1} {'='*20}")
            
            # Reset environment (starts new round or new match)
            obs = env.reset()
            print(f"Observation shape: {obs.shape}")
            print(f"Match context: Round {obs[11]:.0f}, Score {obs[12]:.0f}-{obs[13]:.0f}")
            
            # Take some actions in this round
            for step in range(3):
                action = np.random.randint(0, env.get_action_space_size())
                obs, reward, done, info = env.step(action)
                
                print(f"  Step {step+1}: Action {action}, Reward {reward:.3f}, Done {done}")
                print(f"    Match info: {info.get('match_score', 'N/A')}")
                
                if done:
                    print(f"  Round finished! {info.get('round_winner', 'Unknown')} wins")
                    if info.get('match_completed', False):
                        print(f"  🎉 MATCH COMPLETED! Winner: {info.get('match_winner', 'Unknown')}")
                    break
        
        # Print summary
        env.print_match_summary()
        
        print(f"\nObservation space size: {env.get_observation_space_size()}")
        print(f"Action space size: {env.get_action_space_size()}")
        print(f"Available actions: {env.get_actions()}")
        
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
    except Exception as e:
        print(f"Error during test: {e}")
        import traceback
        traceback.print_exc()
    finally:
        env.close()


if __name__ == "__main__":
    test_match_rl_environment()