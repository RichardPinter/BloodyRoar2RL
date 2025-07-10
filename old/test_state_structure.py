# test_state_structure.py
import numpy as np
from game_state import GameState, PlayerState
from state_normalizer import StateNormalizer
from state_history import StateHistory

# Create a test game state
p1 = PlayerState(x=500, y=400, health=85.0)
p2 = PlayerState(x=900, y=420, health=70.0)
game_state = GameState(player1=p1, player2=p2)

print("Raw Game State:")
print(f"P1: pos=({p1.x}, {p1.y}), health={p1.health}")
print(f"P2: pos=({p2.x}, {p2.y}), health={p2.health}")
print(f"Distance: {game_state.distance:.1f}")

# Test normalization
normalizer = StateNormalizer()
normalized = normalizer.normalize_state(game_state)

print("\nNormalized State Vector:")
print(normalized)
print(f"Shape: {normalized.shape}")

# Test history
history = StateHistory()
history.add_state(game_state, 0.0)

# Simulate movement
p1_new = PlayerState(x=520, y=400, health=85.0)
p2_new = PlayerState(x=880, y=420, health=65.0)
new_state = GameState(player1=p1_new, player2=p2_new)
history.add_state(new_state, 0.033)  # ~30fps

# Calculate velocities
state_with_velocity = history.calculate_velocities()
print(f"\nVelocities calculated:")
print(f"P1 velocity: ({state_with_velocity.player1.velocity_x:.1f}, {state_with_velocity.player1.velocity_y:.1f})")