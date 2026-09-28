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

# ──this part is feature encoding ────────────────────────────────────
# 5 direction slots × 5 categories = 25, + 3 bomb-status one-hots = 28
# + 15 numeric distance scores (coin/crate/enemy x 5 directions) = 43 total

DIR_CATEGORIES  = ['block', 'free', 'coin', 'enemy', 'dead']
BOMB_CATEGORIES = ['False', 'True', 'dead']

DIR_IDX  = {c: i for i, c in enumerate(DIR_CATEGORIES)}   # 5 classes
BOMB_IDX = {c: i for i, c in enumerate(BOMB_CATEGORIES)}  # 3 classes

N_ONEHOT = 5 * len(DIR_CATEGORIES) + len(BOMB_IDX)   # 5*5 + 3 = 28
N_SCORES = 15                                         # coin/crate/enemy x 5 dirs
N_FEATS  = N_ONEHOT + N_SCORES                        # 43

# coin-heaven has 50 coins on ~176 free tiles, so raw coin_score (sum of
# 1/dist over every coin) comes out around 6-7 in a typical state - 6-7x
# the scale of the 0/1 one-hot dims feeding the same first layer. 
SCORE_SCALE = 10.0

MODEL_PATH = Path(__file__).parent / "model.pt"


def encode_feature(feature):
    """
    Convert the 21-element feature list into a 43-dim float vector.
    feature layout: [up, right, down, left, center, bomb_status,
                      coin_score x5, crate_score x5, enemy_score x5]
    - Directions (5 slots × 5 categories) = 25
    - Bomb slot (3 categories) = 3
    - Numeric distance scores (3 kinds × 5 directions) = 15
    Total = 43
    """
    if not feature or len(feature) < 21:
        return np.zeros(N_FEATS, dtype=np.float32)

    vec = np.zeros(N_FEATS, dtype=np.float32)
    n_dir_cats = len(DIR_CATEGORIES)

    # 5 direction slots × 5 categories
    for i in range(5):
        cat = feature[i]
        if cat in DIR_IDX:
            vec[i * n_dir_cats + DIR_IDX[cat]] = 1.0

    # bomb slot: map to 3 categories
    bomb = feature[5]
    if bomb in BOMB_IDX:
        vec[5 * n_dir_cats + BOMB_IDX[bomb]] = 1.0
    else:
        vec[5 * n_dir_cats + BOMB_IDX['False']] = 1.0  # fallback

    # 15 numeric distance scores, scaled down to the one-hot dims' order of
    # magnitude (see SCORE_SCALE note above)
    vec[N_ONEHOT:N_ONEHOT + N_SCORES] = np.array(feature[6:6 + N_SCORES], dtype=np.float32) / SCORE_SCALE

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