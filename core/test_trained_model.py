#!/usr/bin/env python3
"""
Test Trained Model Script

Load a saved model and watch it play through arcade mode.
Useful for evaluating training progress.
"""

import torch
import numpy as np
import time
import os
import sys
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.arcade_rl_environment import ArcadeRLEnvironment
from core.rl_training_simple import PPOAgent


class ModelTester:
    """Test a trained model in arcade mode"""
    
    def __init__(self, model_path: str, arcade_opponents: int = 3):
        self.model_path = model_path
        self.arcade_opponents = arcade_opponents
        
        # Initialize environment and agent
        print("🎮 Initializing Test Environment...")
        self.env = ArcadeRLEnvironment(matches_to_win=arcade_opponents)
        self.agent = PPOAgent(
            state_dim=self.env.get_observation_space_size(),
            action_dim=self.env.get_action_space_size()
        )
        
        # Load the trained model
        self.load_model()
        
        print(f"✅ Model tester ready:")
        print(f"   Model: {model_path}")
        print(f"   Arcade: {arcade_opponents} opponents")
    
    def load_model(self):
        """Load the trained model"""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        
        try:
            print(f"📂 Loading model from {self.model_path}...")
            checkpoint = torch.load(self.model_path)
            self.agent.policy.load_state_dict(checkpoint['policy_state_dict'])
            self.agent.value.load_state_dict(checkpoint['value_state_dict'])
            
            # Load metadata if available
            if 'metadata' in checkpoint:
                metadata = checkpoint['metadata']
                print(f"📊 Model Info:")
                print(f"   Episodes trained: {metadata.get('episode_count', 'Unknown')}")
                print(f"   Arcade completions: {metadata.get('arcade_completions', 'Unknown')}")
                print(f"   Best progress: {metadata.get('best_arcade_progress', 'Unknown')}/{self.arcade_opponents}")
                print(f"   Performance score: {metadata.get('performance_score', 'Unknown'):.3f}")
                print(f"   Saved: {metadata.get('timestamp', 'Unknown')}")
            
            print("✅ Model loaded successfully!")
            
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            raise
    
    def test_model(self, num_tests: int = 3, show_actions: bool = True):
        """Test the model on multiple arcade runs"""
        print(f"\n🧪 Testing model on {num_tests} arcade runs...")
        print("="*60)
        
        results = []
        
        for test_num in range(num_tests):
            print(f"\n🎮 TEST RUN {test_num + 1}/{num_tests}")
            print("-" * 40)
            
            result = self.run_single_test(show_actions=show_actions)
            results.append(result)
            
            print(f"Result: {result['final_status']}")
            print(f"Progress: {result['opponents_beaten']}/{self.arcade_opponents}")
            print(f"Duration: {result['duration']:.1f}s")
        
        # Summary
        self.print_test_summary(results)
        
        return results
    
    def run_single_test(self, show_actions: bool = True) -> dict:
        """Run a single arcade test"""
        start_time = time.time()
        
        # Reset environment
        state = self.env.reset()
        
        total_reward = 0
        total_steps = 0
        opponents_beaten = 0
        final_status = "incomplete"
        
        done = False
        while not done:
            # Select action (no exploration, use best action)
            action, _, _ = self.agent.select_action(state)
            action_name = self.env.get_actions()[action]
            
            if show_actions:
                current_opponent = int(state[19])
                print(f"  Opponent {current_opponent}: {action_name}")
            
            # Environment step
            next_state, reward, done, info = self.env.step(action)
            
            total_reward += reward
            total_steps += 1
            state = next_state
            
            # Check for significant events
            if info.get('match_completed', False):
                if info.get('match_winner') == 'PLAYER 1':
                    opponents_beaten = info.get('arcade_wins', 0)
                    print(f"    ✅ Beat opponent! Total: {opponents_beaten}")
                else:
                    print(f"    ❌ Lost to opponent {info.get('current_opponent', '?')}")
            
            if info.get('arcade_completed', False):
                final_status = "completed"
                opponents_beaten = self.arcade_opponents
                print(f"    🎉 ARCADE COMPLETED!")
                break
            
            # Prevent infinite loops
            if total_steps > 1000:
                final_status = "timeout"
                break
        
        duration = time.time() - start_time
        
        return {
            'final_status': final_status,
            'opponents_beaten': opponents_beaten,
            'total_reward': total_reward,
            'total_steps': total_steps,
            'duration': duration
        }
    
    def print_test_summary(self, results: list):
        """Print summary of all test runs"""
        print("\n" + "="*60)
        print("📊 TEST SUMMARY")
        print("="*60)
        
        # Calculate statistics
        completions = sum(1 for r in results if r['final_status'] == 'completed')
        avg_progress = np.mean([r['opponents_beaten'] for r in results])
        avg_reward = np.mean([r['total_reward'] for r in results])
        avg_duration = np.mean([r['duration'] for r in results])
        
        print(f"Arcade Completions: {completions}/{len(results)} ({completions/len(results)*100:.1f}%)")
        print(f"Average Progress: {avg_progress:.1f}/{self.arcade_opponents} opponents")
        print(f"Average Reward: {avg_reward:+.1f}")
        print(f"Average Duration: {avg_duration:.1f}s")
        
        # Individual results
        print(f"\nIndividual Results:")
        for i, result in enumerate(results, 1):
            status_emoji = "🎉" if result['final_status'] == 'completed' else "❌"
            print(f"  Run {i}: {status_emoji} {result['opponents_beaten']}/{self.arcade_opponents} "
                  f"({result['total_reward']:+.1f} reward, {result['duration']:.1f}s)")
        
        print("="*60)
    
    def close(self):
        """Clean up"""
        self.env.close()


def main():
    """Main testing function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test a trained arcade model")
    parser.add_argument("--model", "-m", 
                       default="models/arcade_ppo_latest.pth",
                       help="Path to model file")
    parser.add_argument("--opponents", "-o", 
                       type=int, default=3,
                       help="Number of opponents in arcade")
    parser.add_argument("--tests", "-t", 
                       type=int, default=3,
                       help="Number of test runs")
    parser.add_argument("--no-actions", 
                       action="store_true",
                       help="Don't show individual actions")
    
    args = parser.parse_args()
    
    print("🧪 TRAINED MODEL TESTER")
    print("="*50)
    
    try:
        # Create tester
        tester = ModelTester(
            model_path=args.model,
            arcade_opponents=args.opponents
        )
        
        # Run tests
        results = tester.test_model(
            num_tests=args.tests,
            show_actions=not args.no_actions
        )
        
    except FileNotFoundError as e:
        print(f"❌ {e}")
        print("💡 Tip: Train a model first with train_arcade_interactive.py")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if 'tester' in locals():
            tester.close()
        print("\n✅ Testing complete!")


if __name__ == "__main__":
    main()