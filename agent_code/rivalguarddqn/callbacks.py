import os
import random
from pathlib import Path

import numpy as np
import torch

from .feature import Feature, N_FEATS, TILE_TYPES
from .model import load_model


ACTIONS = ["UP", "RIGHT", "DOWN", "LEFT", "WAIT", "BOMB"]
N_ACTIONS = len(ACTIONS)

# Locations of the feature groups in the 108-value model input. Use these to make everything clear.
# The first 35 values describe 5 nearby tiles with 7 possible tile types.
# Allowed actions.
VALID_ACTION_SLICE = slice(5 * len(TILE_TYPES), 5 * len(TILE_TYPES) + N_ACTIONS)
# Time until each nearby tile becomes dangerous.
DANGER_ETA_SLICE = slice(VALID_ACTION_SLICE.stop, VALID_ACTION_SLICE.stop + 5)
# Escape space after each move or WAIT.
SAFE_SPACE_SLICE = slice(DANGER_ETA_SLICE.stop, DANGER_ETA_SLICE.stop + 5)
# Safety when possible enemy actions are included.
PESSIMISTIC_SAFETY_SLICE = slice(
    SAFE_SPACE_SLICE.stop,
    SAFE_SPACE_SLICE.stop + 6,
)
# Change in distance to a coin after each move or WAIT.
COIN_DELTA_SLICE = slice(PESSIMISTIC_SAFETY_SLICE.stop, PESSIMISTIC_SAFETY_SLICE.stop + 5)
# Change in distance to a useful crate position.
CRATE_DELTA_SLICE = slice(COIN_DELTA_SLICE.stop, COIN_DELTA_SLICE.stop + 5)
# Expected crate value after each move or WAIT.
CRATE_YIELD_SLICE = slice(CRATE_DELTA_SLICE.stop, CRATE_DELTA_SLICE.stop + 5)
ENEMY_DELTA_SLICE = slice(CRATE_YIELD_SLICE.stop, CRATE_YIELD_SLICE.stop + 5)
# Information about dropping a bomb now.
BOMB_SLICE = slice(ENEMY_DELTA_SLICE.stop, ENEMY_DELTA_SLICE.stop + 5)
# General information such as time, score, and remaining coins.
GLOBAL_SLICE = slice(BOMB_SLICE.stop, BOMB_SLICE.stop + 8)
# Whether each nearby tile was visited recently.
RECENT_VISIT_SLICE = slice(GLOBAL_SLICE.stop, GLOBAL_SLICE.stop + 5)
# The agent's last four movements.
MOTION_HISTORY_SLICE = slice(
    RECENT_VISIT_SLICE.stop,
    RECENT_VISIT_SLICE.stop + 8,
)
# Whether the agent or an enemy can reach a visible coin first.
COIN_CONTEST_SLICE = slice(
    MOTION_HISTORY_SLICE.stop,
    MOTION_HISTORY_SLICE.stop + 5,
)

MODEL_PATH = Path(__file__).resolve().parent / "model.pt"
EXPLORATION_UNIFORM_MIX = 0.15


def state_to_features(feature):
    # Turn the feature groups into the 108 values used by the DQN

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

    # check the shape
    if encoded.shape != (N_FEATS,):
        raise ValueError(
            f"Feature encoded to {encoded.shape}, but we expected ({N_FEATS},)."
        )
    return encoded


def action_mask(feature):
    # Return the actions that can be used now
    mask = np.asarray(feature["valid"], dtype=bool).copy()
    # if nothing is valid, at least we can wait
    if not mask.any():
        mask[ACTIONS.index("WAIT")] = True
    return mask


def exploration_mask(feature):
    # Keep random exploration safe and avoid bombs with no clear target
    # Get safe actions
    mask = action_mask(feature)
    # Keep BOMB only when it has a useful target.
    bomb_index = ACTIONS.index("BOMB")
    bomb = np.asarray(feature["bomb"], dtype=np.float32)
    has_coin_or_attack_target = bool(np.any(bomb[1:4] > 0.0))
    if mask[bomb_index] and not has_coin_or_attack_target:
        mask[bomb_index] = False
    if not mask.any():
        return action_mask(feature)
    return mask


def exploration_action_probabilities(feature, uniform_mix=EXPLORATION_UNIFORM_MIX,):
    # Here give safer actions a higher chance during random exploration
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

    # Keep a small uniform chance so every allowed action can still be tried.
    return (1.0 - float(uniform_mix)) * probabilities + float(
        uniform_mix) * uniform


def action_mask_from_encoded(encoded_states):
    """ Read allowed actions from a batch of encoded states during training """
    if encoded_states.ndim != 2 or encoded_states.shape[1] != N_FEATS:
        raise ValueError(
            f"Need (batch, {N_FEATS}), got {tuple(encoded_states.shape)}"
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

    # Training leaves OASF off. Evaluation uses it.
    default_oasf = not self.train
    oasf_setting = os.environ.get("RIVALGUARD_USE_OASF")
    use_oasf = (
        default_oasf
        if oasf_setting is None
        else oasf_setting.strip() == "1"
    )
    self.feature = Feature(use_opponent_safety_filter=use_oasf)

    if self.train:
        return

    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"There is no RivalGuard DQN model found at {MODEL_PATH}."
        )

    self.policy_net = load_model(
        MODEL_PATH, self.device, N_FEATS, N_ACTIONS
    )
    self.policy_net.eval()
    self.logger.info(
        f"Loaded RivalGuard DQN model {MODEL_PATH}"
    )


def act(self, game_state: dict) -> str:
    if game_state is None:
        return "WAIT"

    # Compute features we need
    feature = self.feature(game_state)
    # Get actions allowed
    valid = action_mask(feature)
    encoded = torch.tensor(
        state_to_features(feature), dtype=torch.float32, device=self.device
    ).unsqueeze(0)

    # Exploration is random, but it only uses safe actions.
    # Actions with more escape space are more likely to be chosen.
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
