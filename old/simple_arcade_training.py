#!/usr/bin/env python3
"""
Simple Arcade Training Script

This script demonstrates how all the arcade components work together:
- ArcadeEpisode: Manages opponent progression and matches
- RoundSubEpisode: Handles individual rounds with pixel-based win detection
- ArcadeEnvironment: Utility for health detection and fast-forwarding

This is the simplified approach requested by the user - no TAS restart initially.
"""

import time
import numpy as np
from typing import Dict, Any

from arcade_episode import ArcadeEpisode, ArcadeOutcome
from round_sub_episode import RoundSubEpisode, RoundOutcome
from arcade_environment import ArcadeEnvironment


class SimpleArcadeTrainer:
    """
    Simple trainer that uses random actions to demonstrate arcade flow
    In a real implementation, this would be replaced with an RL algorithm
    """
    
    def __init__(self, max_opponents: int = 3):
        self.max_opponents = max_opponents
        self.arcade_episode = None
        self.arcade_env = None
        
        # Training stats
        self.total_rounds = 0
        self.total_matches = 0
        self.p1_wins = 0
        self.p2_wins = 0
        
    def run_arcade_training(self, max_rounds_per_match: int = 10) -> Dict[str, Any]:
        """
        Run a single arcade training episode
        
        Args:
            max_rounds_per_match: Maximum rounds per match (prevents infinite loops)
            
        Returns:
            Training results and statistics
        """
        print("🎮 Starting Simple Arcade Training")
        print("=" * 60)
        
        # Initialize arcade components
        self.arcade_episode = ArcadeEpisode(max_opponents=self.max_opponents)
        self.arcade_env = ArcadeEnvironment()
        
        try:
            # Start arcade episode
            arcade_info = self.arcade_episode.reset()
            print(f"Arcade started: {arcade_info}")
            
            total_training_rounds = 0
            
            # Main training loop
            while not self.arcade_episode.is_done():
                print(f"\n🆚 FACING OPPONENT {self.arcade_episode.current_opponent}")
                print("-" * 40)
                
                # Train against current opponent until match is complete
                match_rounds = 0
                
                while (not self.arcade_episode.is_done() and 
                       not self.arcade_episode._is_match_complete() and
                       match_rounds < max_rounds_per_match):
                    
                    match_rounds += 1
                    total_training_rounds += 1
                    
                    print(f"  Round {match_rounds} vs Opponent {self.arcade_episode.current_opponent}")
                    
                    # Train on a single round
                    round_outcome = self._train_single_round()
                    
                    # Update arcade state
                    arcade_info = self.arcade_episode.complete_round(round_outcome)
                    
                    # Print round result
                    self._print_round_result(round_outcome, arcade_info)
                    
                    # Update training stats
                    self.total_rounds += 1
                    if round_outcome == RoundOutcome.PLAYER_WIN:
                        self.p1_wins += 1
                    elif round_outcome == RoundOutcome.PLAYER_LOSS:
                        self.p2_wins += 1
                
                # Check if match completed
                if self.arcade_episode._is_match_complete():
                    self.total_matches += 1
                    print(f"  ✅ Match {self.total_matches} completed!")
                
                # Prevent infinite loops
                if match_rounds >= max_rounds_per_match:
                    print(f"  ⚠️  Reached max rounds per match ({max_rounds_per_match})")
                    break
            
            # Final results
            final_stats = self._get_training_results()
            self._print_final_results(final_stats)
            
            return final_stats
            
        except KeyboardInterrupt:
            print("\n🛑 Training interrupted by user")
            return self._get_training_results()
            
        except Exception as e:
            print(f"\n❌ Training error: {e}")
            import traceback
            traceback.print_exc()
            return self._get_training_results()
            
        finally:
            self._cleanup()
    
    def _train_single_round(self) -> RoundOutcome:
        """
        Train on a single round using random actions
        
        In a real implementation, this would:
        1. Reset the round environment
        2. Run RL training loop (observation -> action -> reward -> repeat)
        3. Return when round is complete
        
        For this demo, we simulate the training with random outcomes.
        """
        
        # Get the current round for training
        current_round = self.arcade_episode.get_current_round()
        
        # Reset round
        try:
            observation = current_round.reset()
            print(f"    Round reset. Observation shape: {observation.shape}")
        except Exception as e:
            print(f"    Warning: Could not reset round: {e}")
            # Return a random outcome if reset fails
            return np.random.choice([RoundOutcome.PLAYER_WIN, RoundOutcome.PLAYER_LOSS])
        
        # Training loop with random actions
        done = False
        step_count = 0
        max_steps = 100  # Prevent infinite loops
        
        print(f"    Training with random actions...")
        
        while not done and step_count < max_steps:
            # Random action (in real training, this would come from RL agent)
            action = np.random.randint(0, current_round.env.action_space.n)
            
            # Take step
            try:
                observation, reward, done, info = current_round.step(action)
                step_count += 1
                
                # Print progress every 20 steps
                if step_count % 20 == 0:
                    p1_health = info.get('p1_health_percentage', 0)
                    p2_health = info.get('p2_health_percentage', 0)
                    print(f"      Step {step_count}: P1={p1_health:.1f}% P2={p2_health:.1f}% Done={done}")
                
            except Exception as e:
                print(f"    Error during step: {e}")
                break
        
        # Get final round outcome
        if current_round.is_round_active():
            print(f"    Round ended due to max steps ({max_steps})")
            # If round didn't finish naturally, return timeout
            return RoundOutcome.TIMEOUT
        
        round_stats = current_round.get_stats()
        if round_stats:
            print(f"    Round completed: {round_stats.outcome.value} in {step_count} steps")
            return round_stats.outcome
        else:
            print(f"    Round completed with unknown outcome")
            return RoundOutcome.ERROR
    
    def _print_round_result(self, outcome: RoundOutcome, arcade_info: Dict[str, Any]):
        """Print the result of a round"""
        match_info = f"P1={arcade_info.get('match_p1_wins', 0)}-{arcade_info.get('match_p2_wins', 0)}"
        
        if outcome == RoundOutcome.PLAYER_WIN:
            print(f"    🎉 P1 WINS! Match: {match_info}")
        elif outcome == RoundOutcome.PLAYER_LOSS:
            print(f"    💀 P1 LOSES! Match: {match_info}")
        elif outcome == RoundOutcome.DRAW:
            print(f"    🤝 DRAW! Match: {match_info}")
        else:
            print(f"    ⏰ {outcome.value.upper()}! Match: {match_info}")
    
    def _get_training_results(self) -> Dict[str, Any]:
        """Get training results and statistics"""
        arcade_stats = None
        if self.arcade_episode:
            arcade_stats = self.arcade_episode.get_stats()
        
        return {
            'total_rounds': self.total_rounds,
            'total_matches': self.total_matches,
            'p1_wins': self.p1_wins,
            'p2_wins': self.p2_wins,
            'p1_win_rate': self.p1_wins / max(1, self.total_rounds),
            'arcade_outcome': arcade_stats.outcome.value if arcade_stats else 'unknown',
            'opponents_beaten': arcade_stats.opponents_beaten if arcade_stats else 0,
            'arcade_duration': arcade_stats.duration if arcade_stats else 0,
        }
    
    def _print_final_results(self, results: Dict[str, Any]):
        """Print final training results"""
        print("\n" + "=" * 60)
        print("🏁 TRAINING COMPLETED")
        print("=" * 60)
        print(f"Total rounds played: {results['total_rounds']}")
        print(f"Total matches: {results['total_matches']}")
        print(f"P1 wins: {results['p1_wins']} ({results['p1_win_rate']:.1%})")
        print(f"P2 wins: {results['p2_wins']}")
        print(f"Opponents beaten: {results['opponents_beaten']}")
        print(f"Arcade outcome: {results['arcade_outcome']}")
        print(f"Training duration: {results['arcade_duration']:.1f}s")
        print("=" * 60)
    
    def _cleanup(self):
        """Clean up resources"""
        if self.arcade_episode:
            self.arcade_episode.close()
        if self.arcade_env:
            self.arcade_env.close()


def main():
    """Main function to run the simple arcade training"""
    
    print("Simple Arcade Training for Bloody Roar 2")
    print("=" * 50)
    print("This demonstrates the integrated arcade training flow:")
    print("- ArcadeEpisode: Match progression")
    print("- RoundSubEpisode: Individual rounds with win detection")
    print("- Random actions for demo (replace with RL agent)")
    print("=" * 50)
    
    # Configuration
    MAX_OPPONENTS = 3  # Start with 3 opponents for testing
    MAX_ROUNDS_PER_MATCH = 15  # Prevent infinite loops
    
    # Create trainer
    trainer = SimpleArcadeTrainer(max_opponents=MAX_OPPONENTS)
    
    # Run training
    try:
        results = trainer.run_arcade_training(max_rounds_per_match=MAX_ROUNDS_PER_MATCH)
        
        # Success message
        print(f"\n✅ Training session completed successfully!")
        print(f"Next steps: Replace random actions with RL agent")
        
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()