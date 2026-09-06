import pickle
import random
from pathlib import Path
from collections import deque

import torch

from .features import Feature
from .model import DQN

ACTIONS   = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
N_ACTIONS = len(ACTIONS)
N_FEATS   = 57

MODEL_PATH = Path(__file__).parent / "model.pt"


def setup(self):
    self.position_history = deque(maxlen=40)  # 记录最近20步位置
    self.device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    self.feature = Feature()

    if self.train:
        # training: setup_training() will load/init the network
        return

    # inference only
    if MODEL_PATH.exists():
        with open(MODEL_PATH, "rb") as f:
            self.policy_net = pickle.load(f).to(self.device)
        self.policy_net.eval()
        self.logger.info(f"Model loaded from {MODEL_PATH}")
    else:
        self.logger.warning("No model found — acting randomly.")
        self.policy_net = DQN(N_FEATS, N_ACTIONS).to(self.device)
        self.policy_net.eval()


# callbacks.py
def act(self, game_state: dict) -> str:
    if game_state is None:
        return random.choice(ACTIONS)

    # 记录位置历史
    pos = game_state['self'][-1]
    self.position_history.append(pos)

    features = self.feature(game_state)
    feat_t   = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(self.device)

    if self.train and random.random() < self.epsilon:
        return random.choice(ACTIONS)

    with torch.no_grad():
        q = self.policy_net(feat_t)
    return ACTIONS[q.argmax(dim=1).item()]
