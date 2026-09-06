import torch.nn as nn


class DQN(nn.Module):
    """
    Dueling DQN for 34-dim one-hot encoded features.
    Input:  34 dims (5 directions * 6 categories + 4 bomb states)
    Output: 6 Q-values (one per action)
    """
    def __init__(self, n_observations, n_actions):
        super().__init__()

        self.trunk = nn.Sequential(
            nn.Linear(n_observations, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.LayerNorm(32),
            nn.ReLU(),
        )

        self.value_head = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

        self.advantage_head = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, n_actions),
        )

    def forward(self, x):
        f = self.trunk(x)
        v = self.value_head(f)
        a = self.advantage_head(f)
        return v + a - a.mean(dim=1, keepdim=True)
