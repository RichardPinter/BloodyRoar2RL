#!/usr/bin/env python3
"""
Controller Configuration

Centralized mapping between action names and BizHawk button names.
This is the single source of truth for all controller mappings.
"""

# Main action mappings
ACTIONS = {
    # Fighting actions
    'transform': 'P1 ○',       # Circle = Transform
    'kick': 'P1 X',           # X = Kick  
    'punch': 'P1 ◻',         # Square = Punch
    'special': 'P1 △',        # Triangle = Special
    'block': 'P1 R1',         # R1 = Block
    'throw': 'P1 R2',         # R2 = Throw (alternative)
    
    # Movement
    'jump': 'P1 D-Pad Up',
    'squat': 'P1 D-Pad Down',
    'left': 'P1 D-Pad Left',
    'right': 'P1 D-Pad Right',
    
    # System buttons
    'start': 'P1 Start',
    'select': 'P1 Select',
    'stop': '',               # Clear all inputs
    
    # Shoulder buttons
    'l1': 'P1 L1',
    'l2': 'P1 L2',
    'r1': 'P1 R1',
    'r2': 'P1 R2',
}

# Combinations (for complex moves)
COMBINATIONS = {
    'beast_transform': ['P1 L1', 'P1 L2'],  # L1 + L2
    'super_special': ['P1 △', 'P1 R1'],     # Triangle + R1
    # Add more combinations as needed
}

# Action categories for RL training
ACTION_CATEGORIES = {
    'movement': ['jump', 'squat', 'left', 'right'],
    'attacks': ['kick', 'punch', 'special'],
    'defense': ['block'],
    'special': ['transform'],
    'system': ['start', 'select', 'stop'],
}

# Helper functions
def get_button_name(action: str) -> str:
    """Get BizHawk button name for an action"""
    return ACTIONS.get(action, '')

def get_all_actions() -> list:
    """Get list of all available actions"""
    return list(ACTIONS.keys())

def get_actions_by_category(category: str) -> list:
    """Get actions in a specific category"""
    return ACTION_CATEGORIES.get(category, [])

def is_valid_action(action: str) -> bool:
    """Check if action is valid"""
    return action in ACTIONS

# Generate action space for RL (0-based indices)
def get_action_space_mapping():
    """Get mapping from action indices to action names for RL"""
    actions = get_all_actions()
    return {i: action for i, action in enumerate(actions)}

def get_action_index(action: str) -> int:
    """Get RL action index for an action name"""
    actions = get_all_actions()
    try:
        return actions.index(action)
    except ValueError:
        return -1  # Invalid action

# Display mappings
def print_mappings():
    """Print all action mappings for debugging"""
    print("Controller Action Mappings:")
    print("=" * 40)
    for action, button in ACTIONS.items():
        print(f"{action:12} → {button}")
    
    print("\nCombinations:")
    print("=" * 40)
    for combo, buttons in COMBINATIONS.items():
        print(f"{combo:12} → {' + '.join(buttons)}")

if __name__ == "__main__":
    print_mappings()
    print(f"\nTotal actions: {len(ACTIONS)}")
    print(f"Action categories: {list(ACTION_CATEGORIES.keys())}")