import torch.nn as nn


class DQNNetwork(nn.Module):
    """Simple feedforward network that maps state -> Q-values for each action."""

    def __init__(self, state_size: int, action_size: int, hidden1: int = 128, hidden2: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_size, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, action_size),
        )

    def forward(self, x):
        return self.net(x)