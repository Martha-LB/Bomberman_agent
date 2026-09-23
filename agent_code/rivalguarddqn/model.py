from pathlib import Path

import torch
import torch.nn as nn


MODEL_FORMAT_VERSION = 1
FEATURE_VERSION = 16


class DQN(nn.Module):
    """Estimate a Q value for each action."""

    def __init__(self, n_observations, n_actions):
        # Build the shared layers and the two Dueling-DQN heads
        super().__init__()
        # Shared feature layers
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
        # How good this state is
        self.value_head = nn.Sequential(
            nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, 1)
        )
        # How good each action is
        self.advantage_head = nn.Sequential(
            nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, n_actions)
        )

    def forward(self, x):
        # Return one Q value for each action
        features = self.trunk(x)
        value = self.value_head(features)
        advantage = self.advantage_head(features)
        return value + advantage - advantage.mean(dim=1, keepdim=True)


def save_model(
    path: Path,
    policy_net: DQN,
) -> None:
    # Save on CPU so model is easy to move
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": MODEL_FORMAT_VERSION,
        "policy_state_dict": {
            key: value.detach().cpu()
            for key, value in policy_net.state_dict().items()
        },
        "config": {
            "agent": "rivalguarddqn",
            "feature_version": FEATURE_VERSION,
        },
    }
    torch.save(payload, path)


def _torch_load(path: Path):
    # Load a model on the CPU
    return torch.load(path, map_location="cpu", weights_only=False)



def load_model(
    path: Path,
    device: torch.device,
    n_observations: int,
    n_actions: int,
) -> DQN:
    # Load a model and check that its format still matches this agent
    # Check file type and version first
    payload = _torch_load(Path(path))
    if not isinstance(payload, dict) or "policy_state_dict" not in payload:
        raise ValueError(f"{path} is not a RivalGuard DQN model file.")
    if payload.get("format_version") != MODEL_FORMAT_VERSION:
        raise ValueError(
            f"Unsupported model format: {payload.get('format_version')}"
        )

    config = dict(payload.get("config", {}))
    feature_version = int(config.get("feature_version", -1))
    if feature_version != FEATURE_VERSION:
        raise ValueError(
            f"Model feature version {feature_version} is incompatible; "
            f"RivalGuard DQN requires Feature V{FEATURE_VERSION}."
        )

    # Input size must match our 108 features
    policy_state = dict(payload["policy_state_dict"])
    model_width = int(policy_state["trunk.0.weight"].shape[1])
    if model_width != n_observations:
        raise ValueError(
            f"Model input width {model_width} is incompatible with "
            f"RivalGuard DQN width {n_observations}."
        )

    policy_net = DQN(n_observations, n_actions).to(device)
    policy_net.load_state_dict(policy_state)
    return policy_net
