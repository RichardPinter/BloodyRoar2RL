import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Copy your constants
FRAME_STACK = 10
NUM_ACTIONS = 7
EXTRA_DIM = 4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LEARNING_RATE = 1e-4

# Copy your DQNNet class here
class DQNNet(nn.Module):
    def __init__(self, in_ch, n_actions, extra_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, 32, 8, 4)
        self.conv2 = nn.Conv2d(32, 64, 4, 2)
        self.conv3 = nn.Conv2d(64, 64, 3, 1)
        conv_out   = 64 * 7 * 7
        # now fc1 expects conv_out + extra_dim
        self.fc1   = nn.Linear(conv_out + extra_dim, 512)
        self.out   = nn.Linear(512, n_actions)

    def forward(self, x, extra):
        # x: (B, in_ch, H, W), extra: (B, extra_dim)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.flatten(1)                   # (B, conv_out)
        x = torch.cat([x, extra], dim=1)   # (B, conv_out+extra_dim)
        x = F.relu(self.fc1(x))
        return self.out(x)

# Run the test
def test_can_learn():
    """Test if the model can learn a simple pattern"""
    print(f"Testing if model can learn basic Q-values on {DEVICE}...")
    
    # Create a tiny test problem
    test_net = DQNNet(FRAME_STACK, NUM_ACTIONS, EXTRA_DIM).to(DEVICE)
    test_optimizer = torch.optim.Adam(test_net.parameters(), lr=LEARNING_RATE)
    
    # Simple pattern: action 0 always gives +1 reward, others give -1
    for i in range(100):
        # Fake state (random noise)
        state = torch.randn(1, FRAME_STACK, 84, 84).to(DEVICE)
        extras = torch.randn(1, EXTRA_DIM).to(DEVICE)
        
        # Get Q-values
        q_values = test_net(state, extras)
        
        # Target: Q-value for action 0 should be high, others low
        target_q = torch.tensor([10.0] + [-10.0] * (NUM_ACTIONS-1)).to(DEVICE)
        
        loss = F.mse_loss(q_values.squeeze(), target_q)
        
        test_optimizer.zero_grad()
        loss.backward()
        test_optimizer.step()
        
        if i % 20 == 0:
            print(f"  Step {i}: Q[0]={q_values[0,0].item():.2f}, "
                  f"Q[1]={q_values[0,1].item():.2f}, Loss={loss.item():.4f}")
    
    # Final check
    final_q = q_values.squeeze().detach().cpu().numpy()
    print(f"\nFinal Q-values: {final_q}")
    print(f"Q[0] > others? {final_q[0] > final_q[1:].max()}")
    print("✓ Basic learning test complete\n")

if __name__ == "__main__":
    test_can_learn()