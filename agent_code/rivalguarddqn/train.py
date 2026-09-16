"""Compact Double-DQN training for ``my_agent``.

The submitted policy is already stored in ``model.pt``.  This module contains
only the learning loop needed to train the same network: uniform current-stage
replay, compact cross-stage rehearsal, five-step returns, Double-DQN targets,
a target network and D4 augmentation.
"""

import os
import random
from collections import deque
from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import events as e
import settings as s
from .callbacks import (
    ACTIONS,
    COIN_CONTEST_SLICE,
    COIN_DELTA_SLICE,
    CRATE_DELTA_SLICE,
    CRATE_YIELD_SLICE,
    DANGER_ETA_SLICE,
    ENEMY_DELTA_SLICE,
    MOTION_HISTORY_SLICE,
    N_ACTIONS,
    N_FEATS,
    PESSIMISTIC_SAFETY_SLICE,
    RECENT_VISIT_SLICE,
    SAFE_SPACE_SLICE,
    VALID_ACTION_SLICE,
    action_mask_from_encoded,
    encode_feature,
)
from .feature import TILE_TYPES
from .model import DQN, load_model, save_model
from .symmetry import D4_SYMMETRIES, adjust_action


GAMMA = 0.99
N_STEP = 5
BATCH_SIZE = 64
REPLAY_CAPACITY = 80_000
REHEARSAL_CAPACITY = 6_000
REHEARSAL_FRACTION = 0.25
LEARNING_RATE = 1e-4
TARGET_UPDATE_STEPS = 5_000
GRADIENT_CLIP = 10.0
EPSILON_START = 1.0
EPSILON_END = 0.05
EPSILON_DECAY_STEPS = 150_000
MODEL_PATH = Path(__file__).resolve().parent / "model.pt"
TRAINING_STATE_PATH = Path(__file__).resolve().parent / "training_state.pt"
TRAINING_STAGE = os.environ.get("MY_AGENT_STAGE", "default").strip() or "default"
FRESH_TRAINING = os.environ.get("MY_AGENT_FRESH", "0") == "1"
BEST_WINDOW = 100

EVENT_REWARDS = {
    e.MOVED_LEFT: -0.02,
    e.MOVED_RIGHT: -0.02,
    e.MOVED_UP: -0.02,
    e.MOVED_DOWN: -0.02,
    e.WAITED: -0.10,
    e.BOMB_DROPPED: -0.02,
    e.INVALID_ACTION: -5.0,
    e.CRATE_DESTROYED: 0.10,
    e.COIN_FOUND: 2.0,
    e.COIN_COLLECTED: 10.0,
    e.KILLED_OPPONENT: 50.0,
    e.KILLED_SELF: -60.0,
    e.GOT_KILLED: -30.0,
}


class ReplayBuffer:
    def __init__(self, capacity):
        self.data = deque(maxlen=int(capacity))

    def __len__(self):
        return len(self.data)

    def add(self, transition):
        self.data.append(transition)

    def sample(self, count):
        return random.sample(self.data, count)


class ReservoirBuffer:
    """Uniform bounded sample of all transitions seen across stages."""

    def __init__(self, capacity):
        self.capacity = int(capacity)
        self.data = []
        self.seen = 0

    def add(self, transition):
        self.seen += 1
        if len(self.data) < self.capacity:
            self.data.append(transition)
            return
        index = random.randrange(self.seen)
        if index < self.capacity:
            self.data[index] = transition


def _pack_replay(transitions):
    if not transitions:
        return {"size": 0}
    states, actions, rewards, next_states, dones, steps = zip(*transitions)
    return {
        "size": len(transitions),
        "states": torch.stack(states).to(torch.float16),
        "actions": torch.tensor(actions, dtype=torch.int8),
        "rewards": torch.tensor(rewards, dtype=torch.float32),
        "next_states": torch.stack(next_states).to(torch.float16),
        "dones": torch.tensor(dones, dtype=torch.bool),
        "steps": torch.tensor(steps, dtype=torch.int8),
    }


def _unpack_replay(payload):
    size = int(payload.get("size", 0))
    if size == 0:
        return []
    return [
        (
            payload["states"][index].float(),
            int(payload["actions"][index]),
            float(payload["rewards"][index]),
            payload["next_states"][index].float(),
            bool(payload["dones"][index]),
            int(payload["steps"][index]),
        )
        for index in range(size)
    ]


def _load_training_state():
    if FRESH_TRAINING or not TRAINING_STATE_PATH.is_file():
        return None
    try:
        return torch.load(
            TRAINING_STATE_PATH, map_location="cpu", weights_only=False
        )
    except TypeError:
        return torch.load(TRAINING_STATE_PATH, map_location="cpu")


def _save_training_state(self):
    payload = {
        "stage": TRAINING_STAGE,
        "latest_policy": {
            key: value.detach().cpu()
            for key, value in self.policy_net.state_dict().items()
        },
        "training_steps": self.training_steps,
        "environment_steps": self.environment_steps,
        "round_count": self.round_count,
        "epsilon": self.epsilon,
        "best_metric": self.best_metric,
        "recent_rewards": list(self.recent_rewards),
        "recent_scores": list(self.recent_scores),
        "recent_suicides": list(self.recent_suicides),
        "recent_enemy_deaths": list(self.recent_enemy_deaths),
        "recent_invalid": list(self.recent_invalid),
        "reservoir_seen": self.stage_reservoir.seen,
        "reservoir": _pack_replay(self.stage_reservoir.data),
    }
    temporary = TRAINING_STATE_PATH.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(TRAINING_STATE_PATH)


def _event_reward(events: List[str]) -> float:
    return float(sum(EVENT_REWARDS.get(event, 0.0) for event in events))


def _bomb_reward(feature, action, events: List[str]) -> float:
    """Bridge the bomb fuse without rewarding unsafe or purposeless bombs."""
    if not feature or action != "BOMB" or e.BOMB_DROPPED not in events:
        return 0.0
    bomb = np.asarray(feature["bomb"], dtype=np.float32)
    if bomb[0] <= 0.0:
        return 0.0
    if bomb[3] > 0.0:
        return 3.0 * float(bomb[3]) * max(1, s.MAX_AGENTS - 1)
    if bomb[2] > 0.0:
        return 0.1 * float(bomb[2]) * max(1, s.MAX_AGENTS - 1)
    return 0.0 if bomb[1] > 0.0 else -0.18


def _state_potential(feature) -> float:
    """Dense progress signal; terminal states have zero potential."""
    if not feature:
        return 0.0
    global_values = np.asarray(feature["global"], dtype=np.float32)
    if global_values[7] > 0.5:
        navigation = 1.0 - float(global_values[0])
    elif global_values[6] > 0.0:
        crate_closeness = 1.0 - float(global_values[1])
        crate_yield = max(
            float(np.max(feature["crate_yield"])),
            float(feature["bomb"][1]),
        )
        navigation = 0.65 * crate_closeness + 0.35 * crate_yield
    else:
        navigation = 0.0

    safety = 0.5 * float(np.max(feature["safe_space"]))
    novelty = 0.25 * (1.0 - float(feature["recent_visit"][4]))
    return navigation + safety + novelty


def _reward(feature, action, next_feature, events: List[str]) -> float:
    base = _event_reward(events) + _bomb_reward(feature, action, events)
    shaped = GAMMA * _state_potential(next_feature) - _state_potential(feature)
    return float(base + shaped)


_TILES_SLICE = slice(0, 5 * len(TILE_TYPES))
_DIRECTIONAL_SLICES = (
    VALID_ACTION_SLICE,
    DANGER_ETA_SLICE,
    SAFE_SPACE_SLICE,
    PESSIMISTIC_SAFETY_SLICE,
    COIN_DELTA_SLICE,
    CRATE_DELTA_SLICE,
    CRATE_YIELD_SLICE,
    ENEMY_DELTA_SLICE,
    COIN_CONTEST_SLICE,
    RECENT_VISIT_SLICE,
)


def _direction_map(rotation, flip_h):
    return [
        ACTIONS.index(adjust_action(ACTIONS[index], rotation, flip_h))
        for index in range(4)
    ]


def _transform_state(state, rotation, flip_h):
    """Apply one D4 board symmetry to an encoded feature vector."""
    transformed = state.clone()
    index_map = _direction_map(rotation, flip_h)

    source_tiles = state[_TILES_SLICE].reshape(5, len(TILE_TYPES))
    target_tiles = transformed[_TILES_SLICE].reshape(5, len(TILE_TYPES))
    for old_index, new_index in enumerate(index_map):
        target_tiles[new_index] = source_tiles[old_index]
    target_tiles[4] = source_tiles[4]

    for group_slice in _DIRECTIONAL_SLICES:
        source = state[group_slice]
        target = transformed[group_slice]
        for old_index, new_index in enumerate(index_map):
            target[new_index] = source[old_index]
        target[4] = source[4]

    source_motion = state[MOTION_HISTORY_SLICE].reshape(-1, 2)
    target_motion = transformed[MOTION_HISTORY_SLICE].reshape(-1, 2)
    dx, dy = source_motion[:, 0].clone(), source_motion[:, 1].clone()
    if rotation == 0:
        transformed_dx, transformed_dy = dx, dy
    elif rotation == 90:
        transformed_dx, transformed_dy = -dy, dx
    elif rotation == 180:
        transformed_dx, transformed_dy = -dx, -dy
    elif rotation == 270:
        transformed_dx, transformed_dy = dy, -dx
    else:
        raise ValueError(f"Unsupported rotation: {rotation}")
    if flip_h:
        transformed_dx = -transformed_dx
    target_motion[:, 0] = transformed_dx
    target_motion[:, 1] = transformed_dy
    return transformed


def setup_training(self):
    state = _load_training_state()
    same_stage = bool(state and state.get("stage") == TRAINING_STAGE)

    if same_stage:
        self.policy_net = DQN(N_FEATS, N_ACTIONS).to(self.device)
        self.policy_net.load_state_dict(state["latest_policy"])
    elif state and MODEL_PATH.is_file():
        # A new phase starts from the previous phase's best policy, while the
        # separate replay state carries the earlier experience distribution.
        self.policy_net = load_model(
            MODEL_PATH, self.device, N_FEATS, N_ACTIONS
        )
    else:
        self.policy_net = DQN(N_FEATS, N_ACTIONS).to(self.device)

    self.target_net = DQN(N_FEATS, N_ACTIONS).to(self.device)
    self.target_net.load_state_dict(self.policy_net.state_dict())
    self.target_net.eval()
    self.optimizer = optim.AdamW(
        self.policy_net.parameters(), lr=LEARNING_RATE, amsgrad=True
    )
    self.loss_fn = nn.SmoothL1Loss()
    self.memory = ReplayBuffer(REPLAY_CAPACITY)
    restored_replay = _unpack_replay(state.get("reservoir", {})) if state else []
    self.historical_memory = list(restored_replay)
    self.stage_reservoir = ReservoirBuffer(REHEARSAL_CAPACITY)
    self.stage_reservoir.data.extend(restored_replay[-REHEARSAL_CAPACITY:])
    self.stage_reservoir.seen = max(
        len(self.stage_reservoir.data),
        int(state.get("reservoir_seen", 0)) if state else 0,
    )
    self.n_step_buffer = deque()
    self.training_steps = int(state.get("training_steps", 0)) if state else 0
    self.environment_steps = int(state.get("environment_steps", 0)) if state else 0
    self.round_count = int(state.get("round_count", 0)) if same_stage else 0
    self.epsilon = float(state.get("epsilon", EPSILON_START)) if state else EPSILON_START
    self.best_metric = float(state.get("best_metric", -float("inf"))) if same_stage else -float("inf")
    self.recent_rewards = deque(
        state.get("recent_rewards", ()) if same_stage else (), maxlen=BEST_WINDOW
    )
    self.recent_scores = deque(
        state.get("recent_scores", ()) if same_stage else (), maxlen=BEST_WINDOW
    )
    self.recent_suicides = deque(
        state.get("recent_suicides", ()) if same_stage else (), maxlen=BEST_WINDOW
    )
    self.recent_enemy_deaths = deque(
        state.get("recent_enemy_deaths", ()) if same_stage else (), maxlen=BEST_WINDOW
    )
    self.recent_invalid = deque(
        state.get("recent_invalid", ()) if same_stage else (), maxlen=BEST_WINDOW
    )
    self.recent_losses = deque(maxlen=100)
    self.episode_reward = 0.0
    self.episode_coins = 0
    self.episode_kills = 0
    self.episode_invalid = 0


def _encoded(feature):
    return torch.tensor(encode_feature(feature), dtype=torch.float32)


def _emit_n_step(self):
    state, action = self.n_step_buffer[0][:2]
    reward = 0.0
    next_state = self.n_step_buffer[0][3]
    done = False
    steps = 0
    for _, _, one_reward, one_next, one_done in list(self.n_step_buffer)[:N_STEP]:
        reward += (GAMMA ** steps) * one_reward
        next_state = one_next
        done = one_done
        steps += 1
        if done:
            break
    transition = (state, action, reward, next_state, done, steps)
    self.memory.add(transition)
    self.stage_reservoir.add(transition)


def _append_transition(self, feature, action, reward, next_feature, done):
    next_state = (
        torch.zeros(N_FEATS, dtype=torch.float32)
        if done or not next_feature
        else _encoded(next_feature)
    )
    self.n_step_buffer.append(
        (_encoded(feature), ACTIONS.index(action), reward, next_state, done)
    )
    if done:
        while self.n_step_buffer:
            _emit_n_step(self)
            self.n_step_buffer.popleft()
    elif len(self.n_step_buffer) >= N_STEP:
        _emit_n_step(self)
        self.n_step_buffer.popleft()


def _update_epsilon(self):
    progress = min(1.0, self.environment_steps / EPSILON_DECAY_STEPS)
    self.epsilon = EPSILON_START + progress * (EPSILON_END - EPSILON_START)


def _optimize(self):
    historical_count = min(
        len(self.historical_memory),
        int(round(BATCH_SIZE * REHEARSAL_FRACTION)),
    )
    current_count = BATCH_SIZE - historical_count
    if len(self.memory) < current_count:
        return

    batch = self.memory.sample(current_count)
    if historical_count:
        batch.extend(random.sample(self.historical_memory, historical_count))
        random.shuffle(batch)
    states, actions, rewards, next_states, dones, steps = zip(
        *batch
    )
    transformed_states = []
    transformed_next_states = []
    transformed_actions = []
    for state, action, next_state in zip(states, actions, next_states):
        rotation, flip_h = random.choice(D4_SYMMETRIES)
        transformed_states.append(_transform_state(state, rotation, flip_h))
        transformed_next_states.append(
            _transform_state(next_state, rotation, flip_h)
        )
        transformed_actions.append(
            ACTIONS.index(adjust_action(ACTIONS[action], rotation, flip_h))
        )

    states_t = torch.stack(transformed_states).to(self.device)
    next_states_t = torch.stack(transformed_next_states).to(self.device)
    actions_t = torch.tensor(
        transformed_actions, dtype=torch.long, device=self.device
    ).unsqueeze(1)
    rewards_t = torch.tensor(
        rewards, dtype=torch.float32, device=self.device
    ).unsqueeze(1)
    dones_t = torch.tensor(dones, dtype=torch.bool, device=self.device).unsqueeze(1)
    steps_t = torch.tensor(steps, dtype=torch.float32, device=self.device).unsqueeze(1)

    predicted = self.policy_net(states_t).gather(1, actions_t)
    with torch.no_grad():
        valid_next = action_mask_from_encoded(next_states_t)
        online_next = self.policy_net(next_states_t).masked_fill(
            ~valid_next, -torch.inf
        )
        best_next_actions = online_next.argmax(dim=1, keepdim=True)
        target_next = self.target_net(next_states_t).gather(1, best_next_actions)
        target_next[dones_t] = 0.0
        discounts = torch.pow(torch.full_like(steps_t, GAMMA), steps_t)
        targets = rewards_t + discounts * target_next

    loss = self.loss_fn(predicted, targets)
    self.optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), GRADIENT_CLIP)
    self.optimizer.step()

    self.training_steps += 1
    self.recent_losses.append(float(loss.item()))
    if self.training_steps % TARGET_UPDATE_STEPS == 0:
        self.target_net.load_state_dict(self.policy_net.state_dict())


def _observe(self, old_state, action, new_state, events):
    feature = self.feature(old_state)
    next_feature = self.feature(new_state) if new_state else None
    reward = _reward(feature, action, next_feature, list(events))
    _append_transition(
        self, feature, action, reward, next_feature, new_state is None
    )
    self.episode_reward += reward
    self.episode_coins += int(e.COIN_COLLECTED in events)
    self.episode_kills += int(e.KILLED_OPPONENT in events)
    self.episode_invalid += int(e.INVALID_ACTION in events)
    self.environment_steps += 1
    _update_epsilon(self)
    _optimize(self)


def game_events_occurred(self, old_game_state, self_action, new_game_state, events):
    if old_game_state is not None:
        _observe(self, old_game_state, self_action, new_game_state, events)


def end_of_round(self, last_game_state, last_action, events):
    suicided = e.KILLED_SELF in events
    killed_by_enemy = e.GOT_KILLED in events and not suicided
    if last_game_state is not None:
        _observe(self, last_game_state, last_action, None, events)

    self.round_count += 1
    self.recent_rewards.append(self.episode_reward)
    self.recent_scores.append(self.episode_coins + 5 * self.episode_kills)
    self.recent_suicides.append(int(suicided))
    self.recent_enemy_deaths.append(int(killed_by_enemy))
    self.recent_invalid.append(self.episode_invalid)

    if len(self.recent_scores) == BEST_WINDOW:
        metric = (
            float(np.mean(self.recent_scores))
            - 2.0 * float(np.mean(self.recent_suicides))
            - 2.0 * float(np.mean(self.recent_enemy_deaths))
            - 0.1 * float(np.mean(self.recent_invalid))
        )
        if metric > self.best_metric:
            self.best_metric = metric
            save_model(MODEL_PATH, self.policy_net)
            self.logger.info(
                "Saved new stage-best model.pt: "
                f"stage={TRAINING_STAGE}, utility={metric:.3f}"
            )

    _save_training_state(self)
    if self.round_count % 20 == 0:
        mean_reward = float(np.mean(self.recent_rewards))
        mean_score = float(np.mean(self.recent_scores))
        mean_loss = float(np.mean(self.recent_losses)) if self.recent_losses else 0.0
        print(
            f"[R{self.round_count:5d}] eps={self.epsilon:.3f} "
            f"| reward={mean_reward:+7.2f} | score={mean_score:5.2f} "
            f"| loss={mean_loss:.3f} | replay={len(self.memory)} "
            f"| history={len(self.historical_memory)} "
            f"| reservoir={len(self.stage_reservoir.data)}"
        )

    self.episode_reward = 0.0
    self.episode_coins = 0
    self.episode_kills = 0
    self.episode_invalid = 0
