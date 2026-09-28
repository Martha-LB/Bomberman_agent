import json
import pickle
import random
from collections import deque
from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import events as e
from .callbacks import ACTIONS, N_ACTIONS, N_FEATS, SCORE_SCALE, encode_feature
from .model import DQN


_DIRECTION_ACTIONS = ACTIONS[:5]

MODEL_PATH = Path(__file__).parent / "model.pt"
STATE_PATH = Path(__file__).parent / "train_state.json"
SNAPSHOT_DIR   = Path(__file__).parent / "_checkpoint_sweep"
SNAPSHOT_EVERY = 100

LR              = 3e-4  # was 1e-3
GAMMA           = 0.9
# bombs/round swung 13.8 -> 44.2 between round 6030 and 6270 with no
# reward change in between - Q-updates overshooting rather than a bad
# reward scale. Dropped LR ~3x, the standard fix for an oscillating
# (not converging) DQN loss.
BATCH_SIZE      = 64
MEMORY_SIZE     = 10_000
TARGET_UPDATE   = 1_000
EPSILON_START   = 0.5
EPSILON_MIN     = 0.18  # was 0.1
EPSILON_DECAY   = 0.9995
# a too-low floor lets the policy settle into "never bomb" with too
# little random action left to rediscover that bombing pays off.
SAVE_EVERY      = 20


GAME_REWARDS = {
    e.MOVED_LEFT:     -1,
    e.MOVED_RIGHT:    -1,
    e.MOVED_UP:       -1,
    e.MOVED_DOWN:     -1,
    e.WAITED:         -5,
    e.BOMB_DROPPED:   -2,
    e.INVALID_ACTION: -20,
    e.CRATE_DESTROYED:  3,
    e.COIN_FOUND:       5,
    e.COIN_COLLECTED:  10,
    e.KILLED_OPPONENT: 50,
    e.KILLED_SELF:    -160,
    e.GOT_KILLED:     -100,
}

SHAPING_SCALE  = 5.0
ESCAPE_WEIGHT  = 6.0


def _potential(feature) -> float:
    """Phi(s) - a function of the state only, never of an action."""
    if not feature or len(feature) < 21:
        return 0.0
    coin_pot    = feature[10] / SCORE_SCALE   # coin_score,  WAIT direction
    crate_pot   = feature[15] / SCORE_SCALE   # crate_score, WAIT direction
    open_routes = sum(1 for i in range(5) if feature[i] not in ('dead', 'block'))
    return SHAPING_SCALE * (coin_pot + crate_pot) + ESCAPE_WEIGHT * open_routes


# ── Action-conditioned "life-saving" reward shaping ─────

DANGER_ESCAPE_REWARD = 6
DANGER_ENTER_PENALTY = -6
DANGER_STAY_PENALTY  = -6
BOMB_SAFE_REWARD      = 3
BOMB_UNSAFE_PENALTY   = -15


def _danger_shaping(feature_old, action: str) -> float:

    if not feature_old or len(feature_old) < 5 or action not in _DIRECTION_ACTIONS:
        return 0.0
    was_doomed = feature_old[4] == 'dead'
    dir_idx    = _DIRECTION_ACTIONS.index(action)
    if dir_idx == 4:  # WAIT - destination is the current tile itself
        return DANGER_STAY_PENALTY if was_doomed else 0.0
    dest_doomed = feature_old[dir_idx] == 'dead'
    if was_doomed and not dest_doomed:
        return DANGER_ESCAPE_REWARD
    if was_doomed and dest_doomed:
        return DANGER_STAY_PENALTY
    if not was_doomed and dest_doomed:
        return DANGER_ENTER_PENALTY
    return 0.0


def _bomb_shaping(feature_old, action: str, events: List[str]) -> float:

    if action != 'BOMB' or e.BOMB_DROPPED not in events:
        return 0.0
    if not feature_old or len(feature_old) < 5:
        return 0.0
    return BOMB_UNSAFE_PENALTY if feature_old[4] == 'dead' else BOMB_SAFE_REWARD


# ── Setup ───────────────────────────────────────────────
def setup_training(self):
    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if MODEL_PATH.exists():
        with open(MODEL_PATH, "rb") as f:
            self.policy_net = pickle.load(f).to(self.device)
        self.logger.info(f"Resumed from {MODEL_PATH}")
    else:
        self.policy_net = DQN(N_FEATS, N_ACTIONS).to(self.device)
        self.logger.info("Fresh model.")

    self.target_net = DQN(N_FEATS, N_ACTIONS).to(self.device)
    self.target_net.load_state_dict(self.policy_net.state_dict())
    self.target_net.eval()

    self.optimizer   = optim.AdamW(self.policy_net.parameters(), lr=LR)
    self.loss_fn     = nn.MSELoss()
    self.memory      = deque(maxlen=MEMORY_SIZE)
    self.epsilon     = EPSILON_START
    self.steps_done  = 0
    self.round_count = 0
    self.recent_rewards = deque(maxlen=100)


    if STATE_PATH.exists():
        with open(STATE_PATH) as f:
            state = json.load(f)
        self.epsilon     = state.get("epsilon", EPSILON_START)
        self.round_count = state.get("round_count", 0)
        self.logger.info(f"Resumed training state: epsilon={self.epsilon:.3f}, round_count={self.round_count}")


def _reward(events: List[str]) -> float:
    return sum(GAME_REWARDS.get(ev, 0) for ev in events)


# ── Store transitions with symmetry augmentation ────────
def _store_simple(self, feature, action, reward, feature_next, done):
    """
    Apply each of 12 symmetries to (feature, action) and (feature_next, action)
    in lockstep, then store as replay transitions.
    """
    from .symmetry import adjust_state, adjust_action

    rotations = [0, 90, 180, 270]
    flips     = [(False, False), (True, False), (False, True)]
    next_default = np.zeros(N_FEATS, dtype=np.float32)

    for rot in rotations:
        for fh, fv in flips:
            aug_f = adjust_state(feature, rot, fh, fv)
            aug_a = adjust_action(action, rot, fh, fv)
            if done or not feature_next:
                aug_fn_enc = next_default
            else:
                aug_fn     = adjust_state(feature_next, rot, fh, fv)
                aug_fn_enc = encode_feature(aug_fn)

            ai = ACTIONS.index(aug_a)
            self.memory.append((
                torch.tensor(encode_feature(aug_f), dtype=torch.float32),
                ai,
                reward,
                torch.tensor(aug_fn_enc, dtype=torch.float32),
                done,
            ))


# ── DQN update ──────────────────────────────────────────
def _optimize(self):
    if len(self.memory) < BATCH_SIZE:
        return

    batch = random.sample(self.memory, BATCH_SIZE)
    states, actions, rewards, next_states, dones = zip(*batch)

    states_t      = torch.stack(states).to(self.device)
    actions_t     = torch.tensor(actions, dtype=torch.int64).unsqueeze(1).to(self.device)
    rewards_t     = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(self.device)
    next_states_t = torch.stack(next_states).to(self.device)
    dones_t       = torch.tensor(dones, dtype=torch.bool).unsqueeze(1).to(self.device)

    q_curr = self.policy_net(states_t).gather(1, actions_t)

    # Double DQN: policy net selects action, target net evaluates
    with torch.no_grad():
        best_actions = self.policy_net(next_states_t).argmax(1, keepdim=True)
        q_next       = self.target_net(next_states_t).gather(1, best_actions)
        q_next[dones_t] = 0.0
        q_target = rewards_t + GAMMA * q_next

    loss = self.loss_fn(q_curr, q_target)
    self.optimizer.zero_grad()
    loss.backward()

    torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=10.0)
    self.optimizer.step()

    self.steps_done += 1
    if self.steps_done % TARGET_UPDATE == 0:
        self.target_net.load_state_dict(self.policy_net.state_dict())


# ── Callbacks ───────────────────────────────────────────
def game_events_occurred(self, old_state, action, new_state, events: List[str]):
    if old_state is None:
        return

    feat_old = self.feature(old_state)
    feat_new = self.feature(new_state) if new_state else None

    reward = _reward(events)
    # potential-based shaping: Phi(s') is 0 at a terminal state by convention
    reward += GAMMA * _potential(feat_new) - _potential(feat_old)
    # action-conditioned life-saving shaping (see note above _danger_shaping)
    reward += _danger_shaping(feat_old, action) + _bomb_shaping(feat_old, action, events)
    done   = new_state is None

    _store_simple(self, feat_old, action, reward, feat_new, done)
    _optimize(self)


def end_of_round(self, last_state, last_action, events: List[str]):
    feat_old = self.feature(last_state)
    reward = _reward(events)
    # potential-based shaping: Phi(terminal) = 0 by convention
    reward += GAMMA * 0.0 - _potential(feat_old)
    # action-conditioned life-saving shaping (see note above _danger_shaping)
    reward += _danger_shaping(feat_old, last_action) + _bomb_shaping(feat_old, last_action, events)

    _store_simple(self, feat_old, last_action, reward, None, done=True)
    _optimize(self)

    if self.epsilon > EPSILON_MIN:
        self.epsilon *= EPSILON_DECAY

    self.recent_rewards.append(reward)
    self.round_count += 1

    if self.round_count % 10 == 0:
        avg = np.mean(self.recent_rewards)
        print(f"[Round {self.round_count}] eps={self.epsilon:.3f} | "
              f"avg100={avg:.2f} | memory={len(self.memory)}")

    if self.round_count % SAVE_EVERY == 0:
        with open(MODEL_PATH, "wb") as f:
            pickle.dump(self.policy_net, f)
        with open(STATE_PATH, "w") as f:
            json.dump({"epsilon": self.epsilon, "round_count": self.round_count}, f)
        self.logger.info(f"Saved → {MODEL_PATH} (epsilon={self.epsilon:.3f}, round_count={self.round_count})")

    if self.round_count % SNAPSHOT_EVERY == 0:
        SNAPSHOT_DIR.mkdir(exist_ok=True)
        snap_path = SNAPSHOT_DIR / f"model_r{self.round_count}.pt"
        with open(snap_path, "wb") as f:
            pickle.dump(self.policy_net, f)
        self.logger.info(f"Snapshot → {snap_path}")
