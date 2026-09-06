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
from .callbacks import ACTIONS, N_ACTIONS, N_FEATS
from .model import DQN
from .symmetry import all_symmetries

MODEL_PATH = Path(__file__).parent / "model.pt"

# ── Hyperparameters ───────────────────────────────────────────────
LR               = 1e-3
GAMMA            = 0.9
BATCH_SIZE       = 64
MEMORY_SIZE      = 10_000
TARGET_UPDATE    = 1_000    # steps between target-net syncs
EPSILON_START    = 0.5
EPSILON_MIN      = 0.05
EPSILON_DECAY    = 0.9995
SAVE_EVERY       = 100      # rounds

# ── Custom events (reward shaping) ────────────────────────────────
MOVE_TO_DANGER   = "MOVE_TO_DANGER"    # walked into explosion / bomb zone
SAFE_WAIT        = "SAFE_WAIT"         # waited because all moves were dangerous
TOWARD_COIN      = "TOWARD_COIN"       # moved in the direction of nearest coin
TOWARD_CRATE     = "TOWARD_CRATE"      # moved toward a crate worth bombing
GOOD_BOMB        = "GOOD_BOMB"         # dropped bomb with safe escape + hits crate/enemy
BAD_BOMB         = "BAD_BOMB"          # dropped bomb with no escape route
LOOP_PENALTY = "LOOP_PENALTY"


GAME_REWARDS = {
    # framework events
    e.COIN_COLLECTED:    40,
    e.KILLED_OPPONENT:   80,
    e.KILLED_SELF:       -30,
    e.GOT_KILLED:        -10,
    e.CRATE_DESTROYED:    5,
    e.INVALID_ACTION:    -3,
    e.WAITED:            -2,
    e.MOVED_LEFT:        -1,
    e.MOVED_RIGHT:       -1,
    e.MOVED_UP:          -1,
    e.MOVED_DOWN:        -1,
    e.SURVIVED_ROUND:    10,
    # custom events
    MOVE_TO_DANGER:     -50,
    SAFE_WAIT:           10,
    TOWARD_COIN:         15,
    TOWARD_CRATE:         5,
    GOOD_BOMB:           20,
    BAD_BOMB:           -60,
    LOOP_PENALTY:       -3.0,
}


# ── Setup ─────────────────────────────────────────────────────────
def setup_training(self):
    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    self.position_history = deque(maxlen=20)

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

    self.optimizer  = optim.AdamW(self.policy_net.parameters(), lr=LR, amsgrad=True)
    self.loss_fn    = nn.MSELoss()
    self.memory     = deque(maxlen=MEMORY_SIZE)
    self.epsilon    = EPSILON_START
    self.steps_done = 0
    self.round_count = 0
    self.recent_rewards = deque(maxlen=100)


# ── Custom event detection ─────────────────────────────────────────
def _add_custom_events(self, old_state, action, new_state, events: List[str]):
    if old_state is None:
        return

    feats_old = self.feature(old_state)
    # feature layout: (5,11) per direction; safe count is col 10
    matrix  = feats_old[:55].reshape(5, 11)
    safe_5  = matrix[:, 10]          # safe count for each direction
    scores  = matrix[:, 5:10]        # [crate_score, coin_score, e0, e1, e2]
    DIR_IDX = {'UP': 0, 'RIGHT': 1, 'DOWN': 2, 'LEFT': 3, 'WAIT': 4}

    if action == 'BOMB':
        bomb_valid = bool(feats_old[55])
        if not bomb_valid:
            events.append(BAD_BOMB)
        else:
            events.append(GOOD_BOMB)

    elif action == 'WAIT':
        # safe wait = every move direction is dangerous, staying is the only option
        if safe_5[4] > 0 and not any(safe_5[:4]):
            events.append(SAFE_WAIT)
        elif safe_5[4] == 0:
            events.append(MOVE_TO_DANGER)

    else:
        idx = DIR_IDX.get(action)
        if idx is not None:
            if safe_5[idx] == 0:
                events.append(MOVE_TO_DANGER)
            # reward moving toward best coin/crate direction
            coin_scores = scores[:, 1]
            crate_scores = scores[:, 0]
            if coin_scores[idx] > 0 and coin_scores[idx] == coin_scores.max():
                events.append(TOWARD_COIN)
            elif crate_scores[idx] > 0 and crate_scores[idx] == crate_scores.max():
                events.append(TOWARD_CRATE)


def _reward(events: List[str]) -> float:
    return sum(GAME_REWARDS.get(ev, 0) for ev in events)


# ── Store transitions with symmetry augmentation ──────────────────
def _store_simple(self, feats, action, reward, feats_next, done):
    """Store all 12 symmetries (rotations/flips) of a transition in replay memory."""
    next_f = np.zeros(N_FEATS, dtype=np.float32) if (done or feats_next is None) else feats_next
    for aug_f, aug_a in all_symmetries(feats, action):
        ai = ACTIONS.index(aug_a)
        # augment next state with the same transform
        self.memory.append((
            torch.tensor(aug_f,   dtype=torch.float32),
            ai,
            reward,
            torch.tensor(next_f,  dtype=torch.float32),
            done,
        ))


# ── DQN update ───────────────────────────────────────────────────
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

    # current Q values
    q_curr = self.policy_net(states_t).gather(1, actions_t)

    # target Q values (Double DQN style)
    with torch.no_grad():
        best_actions = self.policy_net(next_states_t).argmax(1, keepdim=True)
        q_next       = self.target_net(next_states_t).gather(1, best_actions)
        q_next[dones_t] = 0.0
        q_target = rewards_t + GAMMA * q_next

    loss = self.loss_fn(q_curr, q_target)
    self.optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
    self.optimizer.step()

    self.steps_done += 1
    if self.steps_done % TARGET_UPDATE == 0:
        self.target_net.load_state_dict(self.policy_net.state_dict())


# ── Callbacks ────────────────────────────────────────────────────
def game_events_occurred(self, old_state, action, new_state, events: List[str]):
    if old_state is None:
        return


    # 检测是否在原地打转
    if new_state is not None:
        pos = new_state['self'][-1]
        self.position_history.append(pos)
        # 统计最近20步里当前位置出现了几次
        recent = list(self.position_history)
        visit_freq = recent.count(pos) / len(recent)
    
        if visit_freq > 0.3:   # 超过30%的时间在同一个格子
            events.append(LOOP_PENALTY)
    _add_custom_events(self, old_state, action, new_state, events)
    reward    = _reward(events)
    feats_old = self.feature(old_state)
    feats_new = self.feature(new_state) if new_state else None
    done      = new_state is None

    _store_simple(self, feats_old, action, reward, feats_new, done)
    _optimize(self)


def end_of_round(self, last_state, last_action, events: List[str]):
    _add_custom_events(self, last_state, last_action, None, events)
    reward    = _reward(events)
    feats_old = self.feature(last_state)

    _store_simple(self, feats_old, last_action, reward, None, done=True)
    _optimize(self)

    # epsilon decay
    if self.epsilon > EPSILON_MIN:
        self.epsilon *= EPSILON_DECAY

    total_r = reward   # single-step reward at end; full episode sum not tracked here
    self.recent_rewards.append(total_r)
    self.round_count += 1

    if self.round_count % 10 == 0:
        avg = np.mean(self.recent_rewards)
        print(f"[Round {self.round_count}] eps={self.epsilon:.3f} | "
              f"avg100={avg:.2f} | memory={len(self.memory)}")

    if self.round_count % SAVE_EVERY == 0:
        with open(MODEL_PATH, "wb") as f:
            pickle.dump(self.policy_net, f)
        self.logger.info(f"Saved → {MODEL_PATH}")

    self.position_history.clear()
