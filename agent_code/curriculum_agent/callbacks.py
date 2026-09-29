import os
import random
import zlib
from types import SimpleNamespace

import numpy as np

from agent_code.coin_collector_agent import callbacks as collector_callbacks
from agent_code.evasive_agent import callbacks as evasive_callbacks
from agent_code.peaceful_agent import callbacks as peaceful_callbacks
from agent_code.random_agent import callbacks as random_callbacks
from agent_code.rule_based_agent import callbacks as rule_callbacks


# Chance of using each opponent policy in one round.
PHASE_WEIGHTS = {
    "P1": {
        "static": 0.35,
        "peaceful": 0.35,
        "evasive": 0.20,
        "random": 0.08,
        "coin_collector": 0.02,
        "rule": 0.00,
    },
    "P2": {
        "static": 0.00,
        "peaceful": 0.00,
        "evasive": 0.55,
        "random": 0.10,
        "coin_collector": 0.20,
        "rule": 0.15,
    },
    "P3": {
        "static": 0.02,
        "peaceful": 0.03,
        "evasive": 0.15,
        "random": 0.10,
        "coin_collector": 0.40,
        "rule": 0.30,
    },
    "P4": {
        "static": 0.00,
        "peaceful": 0.00,
        "evasive": 0.07,
        "random": 0.03,
        "coin_collector": 0.20,
        "rule": 0.70,
    },
}
POLICIES = tuple(PHASE_WEIGHTS["P1"])


def _seed(logger_name):
    # Give each curriculum opponent a repeatable seed.
    base_seed = int(os.environ.get("MY_AGENT_CURRICULUM_SEED", "10000"))

    # Turn the agent name into a stable number so each opponent gets its own seed.
    return (base_seed + zlib.crc32(logger_name.encode("utf-8"))) % (2**32)


def setup(self):
    # Read the stage and prepare the opponent policies.
    phase = os.environ.get("MY_AGENT_STAGE", "P1").strip().upper() or "P1"

    seed = _seed(self.logger.name)
    self.curriculum_phase = phase
    self.curriculum_rng = np.random.default_rng(seed)
    self.curriculum_round = None
    self.curriculum_policy = None

    # Imported policies need their own callback state.
    self.peaceful_state = SimpleNamespace(logger=self.logger, train=False)
    peaceful_callbacks.setup(self.peaceful_state)
    self.random_state = SimpleNamespace(logger=self.logger, train=False)
    random_callbacks.setup(self.random_state)
    self.evasive_state = SimpleNamespace(logger=self.logger, train=False)
    evasive_callbacks.setup(self.evasive_state)
    self.evasive_state.rng = np.random.default_rng(seed + 1)
    self.collector_state = SimpleNamespace(logger=self.logger, train=False)
    collector_callbacks.setup(self.collector_state)
    self.rule_state = SimpleNamespace(logger=self.logger, train=False)
    rule_callbacks.setup(self.rule_state)
    random.seed(seed)


def _sample_policy(self):
    # Sample one policy for the next round
    weights = PHASE_WEIGHTS[self.curriculum_phase]
    probabilities = [weights[name] for name in POLICIES]
    return str(self.curriculum_rng.choice(POLICIES, p=probabilities))


def _round_policy(self, game_state):
    # Keep the sampled policy unchanged for one round
    round_number = int(game_state["round"])
    if round_number != self.curriculum_round:
        self.curriculum_round = round_number
        self.curriculum_policy = _sample_policy(self)
    return self.curriculum_policy


def act(self, game_state):
    # Use the policy sampled for the current round
    policy = _round_policy(self, game_state)
    if policy == "static":
        return "WAIT"
    if policy == "peaceful":
        return peaceful_callbacks.act(self.peaceful_state, game_state)
    if policy == "evasive":
        return evasive_callbacks.act(self.evasive_state, game_state)
    if policy == "random":
        return random_callbacks.act(self.random_state, game_state)
    if policy == "coin_collector":
        return collector_callbacks.act(self.collector_state, game_state)
    if policy == "rule":
        return rule_callbacks.act(self.rule_state, game_state)
    raise RuntimeError(f"Unknown curriculum policy: {policy}")
