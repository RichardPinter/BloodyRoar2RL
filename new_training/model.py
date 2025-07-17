import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class DQNNet(nn.Module):
    def __init__(self, in_ch: int, n_actions: int):
        super().__init__()
        # Convolutional feature extractor
        self.conv1 = nn.Conv2d(in_ch, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        # Compute flattened conv output size
        convw = (( ( (84 - 8) // 4 + 1 ) - 4 ) // 2 + 1 - 3) + 1
        convh = convw  # assuming square input
        conv_out_size = 64 * convw * convh

        # Fully connected layers
        self.fc1 = nn.Linear(conv_out_size, 512)
        self.out = nn.Linear(512, n_actions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.out(x)


def build_nets(
    device: torch.device,
    frame_stack: int,
    num_actions: int,
    lr: float
) -> tuple[DQNNet, DQNNet, optim.Optimizer]:
    """
    Initialize policy and target networks along with optimizer.

    Returns:
        policy_net: the online network
        target_net: the target network (initialized with policy weights)
        optimizer: Adam optimizer for policy_net
    """
    policy_net = DQNNet(frame_stack, num_actions).to(device)
    target_net = DQNNet(frame_stack, num_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    optimizer = optim.Adam(policy_net.parameters(), lr=lr)
    return policy_net, target_net, optimizer
