import io
import pickle
import random
from pathlib import Path

import numpy as np
import torch

from .features import Feature
from .model import DQN

ACTIONS   = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
N_ACTIONS = len(ACTIONS)

# ── feature encoding ────────────────────────────────────
# 5 direction slots × 6 categories = 30
# + 4 bomb-status one-hots = 34 total
DIR_CATEGORIES  = ['block', 'free', 'coin', 'enemy', 'dead', 'target']
BOMB_CATEGORIES = ['False', 'True', 'dead', 'target', 'KILL!']  # 5 states
# Note: layout is [5*6 + 4] = 34; we use 4 bomb slots by mapping 'target'→'True'
# For clean 34-dim: use 5 categories for bomb, then trim... let's use 4:
# Actually: has_bomb slot ('True'/'False'/'dead'/'target' = 4) + bomb_status separately
# Simplest match to the reference (input_dim=34): use one-hot per slot
DIR_IDX  = {c: i for i, c in enumerate(DIR_CATEGORIES)}          # 6 classes
BOMB_IDX = {c: i for i, c in enumerate(['False', 'True', 'dead', 'KILL!'])}  # 4 classes

N_FEATS  = 5 * len(DIR_CATEGORIES) + len(BOMB_IDX)   # 5*6 + 4 = 34

MODEL_PATH = Path(__file__).parent / "model.pt"


def encode_feature(feature):
    """
    Convert 6-token string feature into 34-dim one-hot float vector.
    feature layout: [up, right, down, left, center, bomb_status]
    - Directions (5 slots × 6 categories) = 30
    - Bomb slot (4 categories) = 4
    Total = 34
    """
    if not feature or len(feature) < 6:
        return np.zeros(N_FEATS, dtype=np.float32)

    vec = np.zeros(N_FEATS, dtype=np.float32)

    # 5 direction slots × 6 categories
    for i in range(5):
        cat = feature[i]
        if cat in DIR_IDX:
            vec[i * 6 + DIR_IDX[cat]] = 1.0

    # bomb slot: map to 4 categories
    bomb = feature[5]
    # possible values: 'True', 'False', 'dead', 'target', 'KILL!'
    # merge 'target' → 'True' (means "we can drop bomb and it's the best move")
    if bomb == 'target':
        bomb = 'True'
    if bomb in BOMB_IDX:
        vec[5 * 6 + BOMB_IDX[bomb]] = 1.0
    else:
        vec[5 * 6 + BOMB_IDX['False']] = 1.0  # fallback

    return vec


def setup(self):
    self.device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    self.feature = Feature()

    if self.train:
        # setup_training will handle model init
        return

    # inference: load model
    if MODEL_PATH.exists():
        # handle CUDA→CPU mapping
        original = torch.storage._load_from_bytes
        torch.storage._load_from_bytes = lambda b: torch.load(
            io.BytesIO(b), map_location='cpu', weights_only=False
        )
        try:
            with open(MODEL_PATH, "rb") as f:
                self.policy_net = pickle.load(f).to(self.device)
            self.policy_net.eval()
            self.logger.info(f"Loaded model from {MODEL_PATH}")
        finally:
            torch.storage._load_from_bytes = original
    else:
        self.logger.warning("No model found — acting randomly.")
        self.policy_net = DQN(N_FEATS, N_ACTIONS).to(self.device)
        self.policy_net.eval()


def act(self, game_state: dict) -> str:
    if game_state is None:
        return random.choice(ACTIONS)

    feature = self.feature(game_state)
    encoded = encode_feature(feature)
    feat_t  = torch.tensor(encoded, dtype=torch.float32).unsqueeze(0).to(self.device)

    # ε-greedy during training
    if self.train and random.random() < self.epsilon:
        return random.choice(ACTIONS)

    with torch.no_grad():
        q = self.policy_net(feat_t)
    return ACTIONS[q.argmax(dim=1).item()]