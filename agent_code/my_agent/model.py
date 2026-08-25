import torch
import torch.nn as nn
import torch.nn.functional as F

ACTIONS = ['UP', 'DOWN', 'LEFT', 'RIGHT', 'WAIT', 'BOMB']
N_ACTIONS = len(ACTIONS)
N_FEATURES = 7 * 17 * 17   # 2023


class PPONetwork(nn.Module):
    """
    Shared trunk + separate actor/critic heads.
    Input: flat feature vector (2023,)
    Actor output: action logits (6,)
    Critic output: state value scalar (1,)
    """
    def __init__(self):
        super().__init__()

        # shared feature extractor
        self.trunk = nn.Sequential(
            nn.Linear(N_FEATURES, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
        )

        self.actor  = nn.Linear(128, N_ACTIONS)   # policy head
        self.critic = nn.Linear(128, 1)            # value head

    def forward(self, x):
        features = self.trunk(x)
        action_probs = F.softmax(self.actor(features), dim=-1)
        value = self.critic(features)
        return action_probs, value

    def get_action_and_value(self, x):
        """Sample action + return log_prob and value (used during rollout)."""
        features = self.trunk(x)
        logits = self.actor(features)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), self.critic(features)

    def evaluate_actions(self, x, actions):
        """Re-evaluate stored actions (used during PPO update)."""
        features = self.trunk(x)
        logits = self.actor(features)
        dist = torch.distributions.Categorical(logits=logits)
        return dist.log_prob(actions), dist.entropy(), self.critic(features)
