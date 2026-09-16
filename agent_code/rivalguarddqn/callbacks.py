import random
from pathlib import Path

import numpy as np
import torch

from .feature import Feature, N_FEATS, TILE_TYPES
from .model import load_model


ACTIONS = ["UP", "RIGHT", "DOWN", "LEFT", "WAIT", "BOMB"]
N_ACTIONS = len(ACTIONS)

VALID_ACTION_SLICE = slice(5 * len(TILE_TYPES), 5 * len(TILE_TYPES) + N_ACTIONS)
DANGER_ETA_SLICE = slice(VALID_ACTION_SLICE.stop, VALID_ACTION_SLICE.stop + 5)
SAFE_SPACE_SLICE = slice(DANGER_ETA_SLICE.stop, DANGER_ETA_SLICE.stop + 5)
PESSIMISTIC_SAFETY_SLICE = slice(
    SAFE_SPACE_SLICE.stop,
    SAFE_SPACE_SLICE.stop + 6,
)
COIN_DELTA_SLICE = slice(PESSIMISTIC_SAFETY_SLICE.stop, PESSIMISTIC_SAFETY_SLICE.stop + 5)
CRATE_DELTA_SLICE = slice(COIN_DELTA_SLICE.stop, COIN_DELTA_SLICE.stop + 5)
CRATE_YIELD_SLICE = slice(CRATE_DELTA_SLICE.stop, CRATE_DELTA_SLICE.stop + 5)
ENEMY_DELTA_SLICE = slice(CRATE_YIELD_SLICE.stop, CRATE_YIELD_SLICE.stop + 5)
BOMB_SLICE = slice(ENEMY_DELTA_SLICE.stop, ENEMY_DELTA_SLICE.stop + 5)
GLOBAL_SLICE = slice(BOMB_SLICE.stop, BOMB_SLICE.stop + 8)
RECENT_VISIT_SLICE = slice(GLOBAL_SLICE.stop, GLOBAL_SLICE.stop + 5)
MOTION_HISTORY_SLICE = slice(
    RECENT_VISIT_SLICE.stop,
    RECENT_VISIT_SLICE.stop + 4 * 2,
)
COIN_CONTEST_SLICE = slice(
    MOTION_HISTORY_SLICE.stop,
    MOTION_HISTORY_SLICE.stop + 5,
)

MODEL_PATH = Path(__file__).resolve().parent / "model.pt"
EXPLORATION_UNIFORM_MIX = 0.15


def encode_feature(feature):
    """Flatten structured Feature V16 into a 108-dimensional vector."""
    if not feature:
        return np.zeros(N_FEATS, dtype=np.float32)

    tiles = np.asarray(feature["tiles"], dtype=np.int64)
    tile_one_hot = np.zeros((5, len(TILE_TYPES)), dtype=np.float32)
    tile_one_hot[np.arange(5), tiles] = 1.0
    encoded = np.concatenate([
        tile_one_hot.ravel(),
        np.asarray(feature["valid"], dtype=np.float32),
        np.asarray(feature["danger_eta"], dtype=np.float32),
        np.asarray(feature["safe_space"], dtype=np.float32),
        np.asarray(feature["pessimistic_safety"], dtype=np.float32),
        np.asarray(feature["coin_delta"], dtype=np.float32),
        np.asarray(feature["crate_delta"], dtype=np.float32),
        np.asarray(feature["crate_yield"], dtype=np.float32),
        np.asarray(feature["enemy_delta"], dtype=np.float32),
        np.asarray(feature["bomb"], dtype=np.float32),
        np.asarray(feature["global"], dtype=np.float32),
        np.asarray(feature["recent_visit"], dtype=np.float32),
        np.asarray(feature["motion_history"], dtype=np.float32),
        np.asarray(feature["coin_contest"], dtype=np.float32),
    ])
    if encoded.shape != (N_FEATS,):
        raise ValueError(
            f"Feature V16 encoded to {encoded.shape}, expected ({N_FEATS},)."
        )
    return encoded


def action_mask(feature):
    """Return the feature extractor's physical and survivability mask."""
    mask = np.asarray(feature["valid"], dtype=bool).copy()
    if not mask.any():
        mask[ACTIONS.index("WAIT")] = True
    return mask


def exploration_mask(feature):
    """Return safe exploration actions without purposeless bomb drops.

    This mask is used only for epsilon-random behaviour. Greedy actions and
    Bellman targets continue to see every hard-valid action, so the learned
    policy remains responsible for the final BOMB decision.
    """
    mask = action_mask(feature)
    bomb_index = ACTIONS.index("BOMB")
    bomb = np.asarray(feature["bomb"], dtype=np.float32)
    has_coin_or_attack_target = bool(np.any(bomb[1:4] > 0.0))
    if mask[bomb_index] and not has_coin_or_attack_target:
        mask[bomb_index] = False
    if not mask.any():
        return action_mask(feature)
    return mask


def exploration_action_probabilities(
    feature,
    uniform_mix=EXPLORATION_UNIFORM_MIX,
):
    """Weight random exploration by deterministic escape capacity."""
    mask = exploration_mask(feature)
    valid_t = torch.as_tensor(mask, dtype=torch.bool)
    capacities = np.concatenate([
        np.asarray(feature["safe_space"], dtype=np.float32),
        np.asarray([feature["bomb"][0]], dtype=np.float32),
    ])
    capacity_t = torch.as_tensor(
        capacities, dtype=torch.float32
    ).clamp_min(0.05)
    logits = torch.log(capacity_t)
    logits = logits.masked_fill(~valid_t, -torch.inf)
    probabilities = torch.softmax(logits, dim=0)
    uniform = valid_t.to(torch.float32) / valid_t.sum()
    return (1.0 - float(uniform_mix)) * probabilities + float(
        uniform_mix
    ) * uniform


def action_mask_from_encoded(encoded_states):
    if encoded_states.ndim != 2 or encoded_states.shape[1] != N_FEATS:
        raise ValueError(
            f"Expected (batch, {N_FEATS}), got {tuple(encoded_states.shape)}"
        )
    mask = (encoded_states[:, VALID_ACTION_SLICE] > 0.5).clone()
    no_action = ~mask.any(dim=1)
    mask[no_action, ACTIONS.index("WAIT")] = True
    return mask


def setup(self):
    if torch.cuda.is_available():
        self.device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        self.device = torch.device("mps")
    else:
        self.device = torch.device("cpu")
    # Training without opponent-aware hard pruning produced the strongest
    # policy; deployment enables it deterministically as a safety layer.
    self.feature = Feature(use_opponent_safety_filter=not self.train)

    if self.train:
        return

    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"No RivalGuard DQN model found at {MODEL_PATH}."
        )

    self.policy_net = load_model(
        MODEL_PATH, self.device, N_FEATS, N_ACTIONS
    )
    self.policy_net.eval()
    self.logger.info(
        f"Loaded RivalGuard DQN model {MODEL_PATH} (DQN + OASF)"
    )


def act(self, game_state: dict) -> str:
    if game_state is None:
        return "WAIT"

    feature = self.feature(game_state)
    valid = action_mask(feature)
    encoded = torch.tensor(
        encode_feature(feature), dtype=torch.float32, device=self.device
    ).unsqueeze(0)

    # Exploration remains stochastic inside the deterministic safety filter.
    # mask and favours actions with larger exact escape capacity.
    if self.train and random.random() < self.epsilon:
        probabilities = exploration_action_probabilities(feature)
        action_index = int(torch.multinomial(probabilities, 1).item())
        action = ACTIONS[action_index]
        self.feature.note_action(game_state, action)
        return action

    with torch.no_grad():
        scores = self.policy_net(encoded)
        valid_t = torch.as_tensor(valid, dtype=torch.bool, device=self.device)
        scores = scores.masked_fill(~valid_t.unsqueeze(0), -torch.inf)
    action = ACTIONS[int(scores.argmax(dim=1).item())]
    self.feature.note_action(game_state, action)
    return action
