import os
import math
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
TARGET_UPDATE_STEPS = 5_000
GRADIENT_CLIP = 10.0

# Each new curriculum stage gets some fresh exploration, while later stages use smaller updates so they do not quickly overwrite earlier skills.
STAGE_SCHEDULES = {
    "P1": {
        "epsilon": (1.00, 0.05, 150_000),
        "learning_rate": (2e-4, 3e-5, 200_000),
    },
    "P2": {
        "epsilon": (0.20, 0.05, 75_000),
        "learning_rate": (2e-5, 8e-6, 120_000),
    },
    "P3": {
        "epsilon": (0.10, 0.03, 75_000),
        "learning_rate": (1e-5, 4e-6, 120_000),
    },
    "P4": {
        "epsilon": (0.06, 0.02, 75_000),
        "learning_rate": (5e-6, 2e-6, 120_000),
    },
}

# Files used for the best model and training state.
MODEL_PATH = Path(__file__).resolve().parent / "model.pt"
TRAINING_STATE_PATH = Path(__file__).resolve().parent / "training_state.pt"

# Select the curriculum stage or request a new model.
TRAINING_STAGE = os.environ.get("MY_AGENT_STAGE", "default").strip() or "default"
FRESH_TRAINING = os.environ.get("MY_AGENT_FRESH", "0") == "1"
TRAINING_SEED_TEXT = os.environ.get("MY_AGENT_TRAINING_SEED", "").strip()

# Compare models using the most recent 100 rounds.
BEST_WINDOW = 100

# Shaping fades as the corresponding measured skill improves. It comes back
# faster after a regression than it disappears after an improvement.
PERFORMANCE_ALPHA_K = 1.2
PERFORMANCE_ALPHA_WARMUP = 20
PERFORMANCE_ALPHA_DECAY_RATE = 0.003
PERFORMANCE_ALPHA_RECOVERY_RATE = 0.02
PERFORMANCE_TARGETS = {
    "P1": {"coins": 2.4, "kills": 2.0},
    "P2": {"coins": 2.4, "kills": 1.0},
    "P3": {"coins": 2.4, "kills": 0.5},
    "P4": {"coins": 2.4, "kills": 0.4},
}

EVENT_REWARDS = {
    e.MOVED_LEFT: -0.02,
    e.MOVED_RIGHT: -0.02,
    e.MOVED_UP: -0.02,
    e.MOVED_DOWN: -0.02,
    e.WAITED: -0.10,
    e.BOMB_DROPPED: -0.02,
    e.INVALID_ACTION: -5.0,
    e.COIN_COLLECTED: 10.0,
    e.KILLED_OPPONENT: 50.0,
}
AUXILIARY_REWARDS = {
    e.CRATE_DESTROYED: 0.10,
    e.COIN_FOUND: 2.0,
}
SELF_DEATH_REWARD = -60.0
ENEMY_DEATH_REWARD = -30.0


class ReplayBuffer:
    # Keep recent transitions from the current stage.

    def __init__(self, capacity):
        # Create a buffer with a fixed maximum size.
        self.data = deque(maxlen=int(capacity))

    def __len__(self):
        # Return the number of stored transitions
        return len(self.data)

    def add(self, transition):
        # Add one transition
        self.data.append(transition)

    def sample(self, count):
        # Sample transitions without replacement.
        return random.sample(self.data, count)


class ReservoirBuffer:
    # Keep a small sample of experience from all stages to protect the learning process

    def __init__(self, capacity):
        # Create an empty reservoir
        self.capacity = int(capacity)
        self.data = []
        self.seen = 0

    def add(self, transition):
        # Add a transition with reservoir sampling
        self.seen += 1
        if len(self.data) < self.capacity:
            self.data.append(transition)
            return
        index = random.randrange(self.seen)
        if index < self.capacity:
            self.data[index] = transition


def _pack_replay(transitions):
    # Pack replay data into small tensors for saving
    # Empty replay needs no tensors
    if not transitions:
        return {"size": 0}
    (
        states,
        actions,
        base_rewards,
        potential_rewards,
        next_states,
        dones,
        steps,
    ) = zip(*transitions)
    return {
        "size": len(transitions),
        "states": torch.stack(states).to(torch.float16),
        "actions": torch.tensor(actions, dtype=torch.int8),
        "base_rewards": torch.tensor(base_rewards, dtype=torch.float32),
        "potential_rewards": torch.tensor(
            potential_rewards, dtype=torch.float32
        ),
        "next_states": torch.stack(next_states).to(torch.float16),
        "dones": torch.tensor(dones, dtype=torch.bool),
        "steps": torch.tensor(steps, dtype=torch.int8),
    }


def _unpack_replay(payload):
    # Restore replay transitions from saved tensors.
    size = int(payload.get("size", 0))
    if size == 0:
        return []
    # Older compact states stored one already-combined reward. Treat it as a base reward so an existing training_state.pt remains loadable.
    legacy_rewards = payload.get("rewards")
    base_rewards = payload.get("base_rewards", legacy_rewards)
    potential_rewards = payload.get("potential_rewards")
    if potential_rewards is None:
        potential_rewards = torch.zeros(size, dtype=torch.float32)
    return [
        (
            payload["states"][index].float(),
            int(payload["actions"][index]),
            float(base_rewards[index]),
            float(potential_rewards[index]),
            payload["next_states"][index].float(),
            bool(payload["dones"][index]),
            int(payload["steps"][index]),
        )
        for index in range(size)
    ]


def _load_training_state():
    # Load the latest training state unless fresh training was requested
    # Fresh training ignores old progress
    if FRESH_TRAINING or not TRAINING_STATE_PATH.is_file():
        return None
    try:
        return torch.load(
            TRAINING_STATE_PATH, map_location="cpu", weights_only=False
        )
    except TypeError:
        return torch.load(TRAINING_STATE_PATH, map_location="cpu")


def _save_training_state(self):
    # Save progress and cross-stage replay data
    # Keep enough data to continue next time
    payload = {
        "stage": TRAINING_STAGE,
        # Latest network, may be different from best model.pt
        "latest_policy": {
            key: value.detach().cpu()
            for key, value in self.policy_net.state_dict().items()
        },
        # Training and schedule progress
        "training_steps": self.training_steps,
        "environment_steps": self.environment_steps,
        "stage_environment_steps": self.stage_environment_steps,
        "round_count": self.round_count,
        "epsilon": self.epsilon,
        "best_metric": self.best_metric,
        # Recent rounds for averages and model selection
        "recent_rewards": list(self.recent_rewards),
        "recent_scores": list(self.recent_scores),
        "recent_suicides": list(self.recent_suicides),
        "recent_enemy_deaths": list(self.recent_enemy_deaths),
        "recent_invalid": list(self.recent_invalid),
        "recent_coins": list(self.recent_coins),
        "recent_kills": list(self.recent_kills),
        # Current auxiliary reward strength
        "coin_alpha": self.current_coin_alpha,
        "combat_alpha": self.current_combat_alpha,
        # Small replay sample, not the full replay buffer
        "reservoir_seen": self.stage_reservoir.seen,
        "reservoir": _pack_replay(self.stage_reservoir.data),
    }
    temporary = TRAINING_STATE_PATH.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(TRAINING_STATE_PATH)


def _event_reward(events: List[str], coin_alpha: float) -> float:
    # Return task rewards plus the current coin-learning assistance
    # Real rewards always stay, coin help follows alpha
    reward = float(sum(EVENT_REWARDS.get(event, 0.0) for event in events))
    reward += float(coin_alpha) * sum(
        AUXILIARY_REWARDS.get(event, 0.0) for event in events
    )

    # A self-kill also emits GOT_KILLED. Count only one death penalty.
    if e.KILLED_SELF in events:
        reward += SELF_DEATH_REWARD
    elif e.GOT_KILLED in events:
        reward += ENEMY_DEATH_REWARD
    return reward


def _bomb_reward(
    feature,
    action,
    events: List[str],
    combat_alpha: float,
) -> float:
    # Give a small early signal for a useful bomb
    # Reward only a bomb really dropped
    if not feature or action != "BOMB" or e.BOMB_DROPPED not in events:
        return 0.0
    bomb = np.asarray(feature["bomb"], dtype=np.float32)
    if bomb[0] <= 0.0:
        return 0.0
    if bomb[3] > 0.0:
        return (
            float(combat_alpha)
            * 3.0
            * float(bomb[3])
            * max(1, s.MAX_AGENTS - 1)
        )
    if bomb[2] > 0.0:
        return (
            float(combat_alpha)
            * 0.1
            * float(bomb[2])
            * max(1, s.MAX_AGENTS - 1)
        )
    return 0.0 if bomb[1] > 0.0 else -0.18


def _state_potential_components(feature) -> np.ndarray:
    # Return separate coin, combat, and permanent safety potentials
    if not feature:
        return np.zeros(3, dtype=np.float32)
    global_values = np.asarray(feature["global"], dtype=np.float32)
    # Visible coin first, hidden coin crate second
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

    novelty = 0.25 * (1.0 - float(feature["recent_visit"][4]))
    bomb = np.asarray(feature["bomb"], dtype=np.float32)
    attack = float(bomb[0]) * min(
        1.0,
        float(bomb[2]) + 2.0 * float(bomb[3]),
    )
    attack *= float(
        global_values[7] <= 0.5
        and global_values[6] <= 0.0
        and global_values[5] <= 0.0
    )
    safety = 0.5 * float(np.max(feature["safe_space"]))
    return np.asarray(
        [navigation + novelty, attack, safety],
        dtype=np.float32,
    )


def _reward_parts(self, feature, action, next_feature, events: List[str]):
    # Keep task and potential rewards separate inside replay
    # Base reward from game events
    base = _event_reward(events, self.current_coin_alpha)
    base += _bomb_reward(
        feature,
        action,
        events,
        self.current_combat_alpha,
    )
    # Reward useful state change, not one fixed action
    potential_change = (
        GAMMA * _state_potential_components(next_feature)
        - _state_potential_components(feature)
    )
    potential = (
        self.current_coin_alpha * float(potential_change[0])
        + self.current_combat_alpha * float(potential_change[1])
        + float(potential_change[2])
    )
    return float(base), float(potential)


def _performance_alpha(mean_value, target):
    # Better skill means less training help
    skill = min(max(float(mean_value) / float(target), 0.0), 1.0)
    alpha = 1.0 - math.tanh(PERFORMANCE_ALPHA_K * skill)
    return float(alpha), float(skill)


def _move_alpha(current, target):
    # Lost skill brings help back faster
    rate = (
        PERFORMANCE_ALPHA_RECOVERY_RATE
        if target > current
        else PERFORMANCE_ALPHA_DECAY_RATE
    )
    return float(current + rate * (target - current))


def _update_performance_alphas(self):
    # Adjust coin and combat assistance from recent real performance
    # Wait until recent average is useful
    if len(self.recent_coins) < PERFORMANCE_ALPHA_WARMUP:
        return
    targets = PERFORMANCE_TARGETS.get(
        TRAINING_STAGE.upper(), PERFORMANCE_TARGETS["P1"]
    )
    coin_target, self.coin_skill = _performance_alpha(
        np.mean(self.recent_coins), targets["coins"]
    )
    combat_target, self.combat_skill = _performance_alpha(
        np.mean(self.recent_kills), targets["kills"]
    )
    self.current_coin_alpha = _move_alpha(
        self.current_coin_alpha, coin_target
    )
    self.current_combat_alpha = _move_alpha(
        self.current_combat_alpha, combat_target
    )


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
    # Map the four move directions after a board transform
    return [
        ACTIONS.index(adjust_action(ACTIONS[index], rotation, flip_h))
        for index in range(4)
    ]


def _transform_state(state, rotation, flip_h):
    # Rotate and flip one encoded state
    # Start with a copy, then move directional features
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
        raise ValueError(f"Wrong rotation: {rotation}")
    if flip_h:
        transformed_dx = -transformed_dx
    target_motion[:, 0] = transformed_dx
    target_motion[:, 1] = transformed_dy
    return transformed


def setup_training(self):
    # Set up the networks, optimizer, replay data, and counters

    # An optional fixed seed makes two training experiments comparable.
    if TRAINING_SEED_TEXT:
        training_seed = int(TRAINING_SEED_TEXT)
        random.seed(training_seed)
        np.random.seed(training_seed)
        torch.manual_seed(training_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(training_seed)

    # Load same stage or start from last stage model
    state = _load_training_state()
    same_stage = bool(state and state.get("stage") == TRAINING_STAGE)

    if same_stage:
        # Continue the exact same stage.
        self.policy_net = DQN(N_FEATS, N_ACTIONS).to(self.device)
        self.policy_net.load_state_dict(state["latest_policy"])
    elif state and MODEL_PATH.is_file():
        # Start a new stage from the previous stage's best model.
        self.policy_net = load_model(
            MODEL_PATH, self.device, N_FEATS, N_ACTIONS
        )
    else:
        self.policy_net = DQN(N_FEATS, N_ACTIONS).to(self.device)

    # Every stage has its own exploration and learning rate
    schedule = STAGE_SCHEDULES.get(
        TRAINING_STAGE.upper(), STAGE_SCHEDULES["P1"]
    )
    self.epsilon_start, self.epsilon_end, self.epsilon_decay_steps = (
        schedule["epsilon"]
    )
    self.lr_start, self.lr_end, self.lr_decay_steps = schedule[
        "learning_rate"
    ]

    # Target network changes slower than policy network
    self.target_net = DQN(N_FEATS, N_ACTIONS).to(self.device)
    # Start target network from current policy
    self.target_net.load_state_dict(self.policy_net.state_dict())
    self.target_net.eval()

    # Continue schedule only inside same stage
    self.stage_environment_steps = (
        int(state.get("stage_environment_steps", 0)) if same_stage else 0
    )
    self.current_lr = _linear_schedule(
        self.lr_start,
        self.lr_end,
        self.stage_environment_steps,
        self.lr_decay_steps,
    )
    # Optimizer starts new after every program launch
    self.optimizer = optim.AdamW(
        self.policy_net.parameters(), lr=self.current_lr, amsgrad=True
    )
    self.loss_fn = nn.SmoothL1Loss()
    # Current replay plus small history from older stages
    self.memory = ReplayBuffer(REPLAY_CAPACITY)

    # Reservoir is small replay saved from old training
    restored_replay = _unpack_replay(state.get("reservoir", {})) if state else []
    self.historical_memory = list(restored_replay)

    # Build a new reservoir and seed it with old samples
    self.stage_reservoir = ReservoirBuffer(REHEARSAL_CAPACITY)
    self.stage_reservoir.data.extend(restored_replay[-REHEARSAL_CAPACITY:])
    self.stage_reservoir.seen = max(
        len(self.stage_reservoir.data),
        int(state.get("reservoir_seen", 0)) if state else 0,
    )
    # No unfinished n-step transition after restart
    self.n_step_buffer = deque()

    # Total counters continue across stages
    self.training_steps = int(state.get("training_steps", 0)) if state else 0
    self.environment_steps = int(state.get("environment_steps", 0)) if state else 0

    # Round count belongs only to this stage
    self.round_count = int(state.get("round_count", 0)) if same_stage else 0

    # Get epsilon from current stage progress
    self.epsilon = _linear_schedule(
        self.epsilon_start,
        self.epsilon_end,
        self.stage_environment_steps,
        self.epsilon_decay_steps,
    )

    # New stage starts a new best-model comparison
    self.best_metric = float(state.get("best_metric", -float("inf"))) if same_stage else -float("inf")

    # Restore recent 100 rounds only inside same stage
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
    self.recent_coins = deque(
        state.get("recent_coins", ()) if same_stage else (), maxlen=BEST_WINDOW
    )
    self.recent_kills = deque(
        state.get("recent_kills", ()) if same_stage else (), maxlen=BEST_WINDOW
    )

    # Alpha continues into next curriculum stage
    self.current_coin_alpha = float(state.get("coin_alpha", 1.0)) if state else 1.0
    self.current_combat_alpha = (
        float(state.get("combat_alpha", 1.0)) if state else 1.0
    )

    # Skill is recalculated from new recent rounds
    self.coin_skill = 0.0
    self.combat_skill = 0.0

    # Loss history is only for current console output
    self.recent_losses = deque(maxlen=100)

    # Counters for the round being played now
    self.episode_reward = 0.0
    self.episode_coins = 0
    self.episode_kills = 0
    self.episode_invalid = 0


def _encoded(feature):
    # Encode one feature dictionary as a CPU tensor
    return torch.tensor(encode_feature(feature), dtype=torch.float32)


def _emit_n_step(self):
    # Turn the oldest buffered steps into one n-step transition

    # Join up to N_STEP rewards
    state, action = self.n_step_buffer[0][:2]
    base_reward = 0.0
    potential_reward = 0.0
    next_state = self.n_step_buffer[0][4]
    done = False
    steps = 0
    for (
        _, _, one_base, one_potential, one_next, one_done
    ) in list(self.n_step_buffer)[:N_STEP]:
        discount = GAMMA ** steps
        base_reward += discount * one_base
        potential_reward += discount * one_potential
        next_state = one_next
        done = one_done
        steps += 1
        if done:
            break
    transition = (
        state,
        action,
        base_reward,
        potential_reward,
        next_state,
        done,
        steps,
    )
    self.memory.add(transition)
    self.stage_reservoir.add(transition)


def _append_transition(
    self,
    feature,
    action,
    base_reward,
    potential_reward,
    next_feature,
    done,
):
    # Add one step and emit ready n-step transitions

    # Terminal state has no next feature
    next_state = (
        torch.zeros(N_FEATS, dtype=torch.float32)
        if done or not next_feature
        else _encoded(next_feature)
    )
    self.n_step_buffer.append(
        (
            _encoded(feature),
            ACTIONS.index(action),
            base_reward,
            potential_reward,
            next_state,
            done,
        )
    )
    # End of round flushes every remaining short transition
    if done:
        while self.n_step_buffer:
            _emit_n_step(self)
            self.n_step_buffer.popleft()
    elif len(self.n_step_buffer) >= N_STEP:
        _emit_n_step(self)
        self.n_step_buffer.popleft()


def _linear_schedule(start, end, step, decay_steps):
    # Move linearly from start to end over a fixed number of steps
    progress = min(1.0, max(0.0, step / max(1, decay_steps)))
    return start + progress * (end - start)


def _update_schedules(self):
    # Update exploration and learning rate within the current stage
    # Both schedules use stage steps
    self.epsilon = _linear_schedule(
        self.epsilon_start,
        self.epsilon_end,
        self.stage_environment_steps,
        self.epsilon_decay_steps,
    )
    self.current_lr = _linear_schedule(
        self.lr_start,
        self.lr_end,
        self.stage_environment_steps,
        self.lr_decay_steps,
    )
    for group in self.optimizer.param_groups:
        group["lr"] = self.current_lr


def _optimize(self):
    # Run one Double-DQN update
    # Keep some old-stage examples in every batch
    historical_count = min(
        len(self.historical_memory),
        int(round(BATCH_SIZE * REHEARSAL_FRACTION)),
    )
    current_count = BATCH_SIZE - historical_count
    if len(self.memory) < current_count:
        return

    # Mix current-stage data with a small sample from older stages.
    batch = self.memory.sample(current_count)
    if historical_count:
        batch.extend(random.sample(self.historical_memory, historical_count))
        random.shuffle(batch)
    (
        states,
        actions,
        base_rewards,
        potential_rewards,
        next_states,
        dones,
        steps,
    ) = zip(
        *batch
    )
    # Random board symmetries provide extra training examples.
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
        np.asarray(base_rewards) + np.asarray(potential_rewards),
        dtype=torch.float32,
        device=self.device,
    ).unsqueeze(1)
    dones_t = torch.tensor(dones, dtype=torch.bool, device=self.device).unsqueeze(1)
    steps_t = torch.tensor(steps, dtype=torch.float32, device=self.device).unsqueeze(1)

    # Q value of the action actually taken
    predicted = self.policy_net(states_t).gather(1, actions_t)
    with torch.no_grad():
        # The online network chooses the action; the target network scores it.
        valid_next = action_mask_from_encoded(next_states_t)
        online_next = self.policy_net(next_states_t).masked_fill(
            ~valid_next, -torch.inf
        )
        best_next_actions = online_next.argmax(dim=1, keepdim=True)
        target_next = self.target_net(next_states_t).gather(1, best_next_actions)
        target_next[dones_t] = 0.0
        discounts = torch.pow(torch.full_like(steps_t, GAMMA), steps_t)
        targets = rewards_t + discounts * target_next

    # One gradient update
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
    # Store one step, update counters, and train once
    # Build both sides of the transition
    feature = self.feature(old_state)
    next_feature = self.feature(new_state) if new_state else None
    base_reward, potential_reward = _reward_parts(
        self, feature, action, next_feature, list(events)
    )
    _append_transition(
        self,
        feature,
        action,
        base_reward,
        potential_reward,
        next_feature,
        new_state is None,
    )
    reward = base_reward + potential_reward
    self.episode_reward += reward
    self.episode_coins += int(e.COIN_COLLECTED in events)
    self.episode_kills += int(e.KILLED_OPPONENT in events)
    self.episode_invalid += int(e.INVALID_ACTION in events)
    self.environment_steps += 1
    self.stage_environment_steps += 1
    _update_schedules(self)
    _optimize(self)


def game_events_occurred(self, old_game_state, self_action, new_game_state, events):
    # Handle a normal game step
    if old_game_state is not None:
        _observe(self, old_game_state, self_action, new_game_state, events)


def end_of_round(self, last_game_state, last_action, events):
    """Finish the last step, save progress, and reset round counters."""
    # Finish final transition first
    suicided = e.KILLED_SELF in events
    killed_by_enemy = e.GOT_KILLED in events and not suicided
    if last_game_state is not None:
        _observe(self, last_game_state, last_action, None, events)

    self.round_count += 1
    self.recent_rewards.append(self.episode_reward)
    self.recent_scores.append(self.episode_coins + 5 * self.episode_kills)
    self.recent_coins.append(self.episode_coins)
    self.recent_kills.append(self.episode_kills)
    self.recent_suicides.append(int(suicided))
    self.recent_enemy_deaths.append(int(killed_by_enemy))
    self.recent_invalid.append(self.episode_invalid)
    _update_performance_alphas(self)

    # Save the best recent policy as model.pt
    if len(self.recent_scores) == BEST_WINDOW:
        # Prefer score, but penalize deaths and invalid actions.
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
            f"| lr={self.current_lr:.1e} "
            f"| alpha={self.current_coin_alpha:.3f}/"
            f"{self.current_combat_alpha:.3f} "
            f"(coin_skill={self.coin_skill:.2f}, "
            f"combat_skill={self.combat_skill:.2f}) "
            f"| reward={mean_reward:+7.2f} | score={mean_score:5.2f} "
            f"| loss={mean_loss:.3f} | replay={len(self.memory)} "
            f"| history={len(self.historical_memory)} "
            f"| reservoir={len(self.stage_reservoir.data)}"
        )

    self.episode_reward = 0.0
    self.episode_coins = 0
    self.episode_kills = 0
    self.episode_invalid = 0
