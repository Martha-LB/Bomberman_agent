import torch.nn as nn


class DQN(nn.Module):
    """
    Dueling DQN: splits into value stream + advantage stream.
    Q(s,a) = V(s) + A(s,a) - mean(A(s,·))
    This stabilizes training because the network learns state value
    separately from action advantages.
    Input: 57-dim feature vector (from features.py)
    Output: Q-value for each of 6 actions
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
