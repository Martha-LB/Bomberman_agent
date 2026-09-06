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
from .callbacks import ACTIONS, N_ACTIONS, N_FEATS, encode_feature
from .model import DQN

MODEL_PATH = Path(__file__).parent / "model.pt"
STATE_PATH = Path(__file__).parent / "train_state.json"

# ── Hyperparameters ─────────────────────────────────────
LR              = 1e-3
GAMMA           = 0.9
BATCH_SIZE      = 64
MEMORY_SIZE     = 10_000
TARGET_UPDATE   = 1_000
EPSILON_START   = 0.5
EPSILON_MIN     = 0.1
EPSILON_DECAY   = 0.9995
SAVE_EVERY      = 100

# ── Custom events (matching feature_is_everything) ──────
MOVE_TO_DEAD    = "MOVE_TO_DEAD"
MOVE_TO_TARGET  = "MOVE_TO_TARGET"
ATTACK_TARGET   = "ATTACK_TARGET"
ATTACK_ENEMY    = "ATTACK_ENEMY"
KILL_ENEMY      = "KILL_ENEMY"

GAME_REWARDS = {
    e.MOVED_LEFT:     -1,
    e.MOVED_RIGHT:    -1,
    e.MOVED_UP:       -1,
    e.MOVED_DOWN:     -1,
    e.WAITED:         -5,
    e.BOMB_DROPPED:   -5,
    e.INVALID_ACTION: -20,
    e.COIN_COLLECTED:  10,
    e.KILLED_OPPONENT: 50,
    e.KILLED_SELF:    -300,
    e.GOT_KILLED:     -100,
    MOVE_TO_DEAD:     -100,
    MOVE_TO_TARGET:    50,
    ATTACK_TARGET:     50,
    ATTACK_ENEMY:      20,
    KILL_ENEMY:        500,
}


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

    # BUGFIX (2026-09-06): epsilon/round_count used to always reset here, even
    # when resuming a checkpoint that had already decayed epsilon way down.
    # Every time --train 1 was invoked again (e.g. moving to the next
    # curriculum stage, or just re-running training), the agent would suddenly
    # start acting ~50% randomly again on top of an already-good policy,
    # flooding the (size-limited) replay buffer with degraded experience -
    # found this by accident while running a hyperparameter sweep: even the
    # "unchanged hyperparameters, just train longer" control run collapsed a
    # loot-crate score of 573/20 rounds down to 0/20 rounds after 600 more
    # rounds of "training". Now epsilon/round_count are persisted alongside
    # model.pt and restored here, so resuming continues the schedule instead
    # of restarting it. A checkpoint saved before this fix has no
    # train_state.json yet, so its very next training run will still reset
    # once - after that, it persists correctly.
    if STATE_PATH.exists():
        with open(STATE_PATH) as f:
            state = json.load(f)
        self.epsilon     = state.get("epsilon", EPSILON_START)
        self.round_count = state.get("round_count", 0)
        self.logger.info(f"Resumed training state: epsilon={self.epsilon:.3f}, round_count={self.round_count}")


# ── Custom event detection ──────────────────────────────
def _add_custom_events(feature, action, events: List[str]):
    """Inject custom events based on the token feature."""
    if not feature:
        return

    action_idx = ACTIONS.index(action) if action in ACTIONS else -1

    # movement events (action indices 0-4)
    if 0 <= action_idx < 5:
        tile = feature[action_idx]
        if tile == 'target':
            events.append(MOVE_TO_TARGET)
        elif tile == 'dead':
            events.append(MOVE_TO_DEAD)

    # bomb events
    if action == 'BOMB':
        if 'enemy' in feature[:5]:
            events.append(ATTACK_ENEMY)
        if feature[-1] == 'target':
            events.append(ATTACK_TARGET)
        if feature[-1] == 'KILL!':
            events.append(KILL_ENEMY)


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

    _add_custom_events(feat_old, action, events)
    reward = _reward(events)
    done   = new_state is None

    _store_simple(self, feat_old, action, reward, feat_new, done)
    _optimize(self)


def end_of_round(self, last_state, last_action, events: List[str]):
    feat_old = self.feature(last_state)
    _add_custom_events(feat_old, last_action, events)
    reward = _reward(events)

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
