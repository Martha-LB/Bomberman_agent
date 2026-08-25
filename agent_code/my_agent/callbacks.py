import os
import numpy as np
import torch
from pathlib import Path
from .model import PPONetwork, ACTIONS

AGENT_DIR  = Path(__file__).parent
MODEL_PATH = AGENT_DIR / "model.pt"

def setup(self):
    """Load model or init fresh."""
    self.device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    self.model   = PPONetwork().to(self.device)
    self.epsilon = 0.0   # default; overwritten by setup_training() when training

    if MODEL_PATH.exists():
        self.model.load_state_dict(torch.load(MODEL_PATH, map_location=self.device))
        self.logger.info(f"Loaded model from {MODEL_PATH}")
    else:
        self.logger.info("No saved model found, starting with random weights.")

    self.model.eval()


def act(self, game_state: dict) -> str:
    """Pick action from policy."""

    if game_state is None:
        return np.random.choice(ACTIONS)

    obs = state_to_features(game_state)
    obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)

    with torch.no_grad():
        action_probs, _ = self.model(obs_tensor)

    # epsilon-greedy exploration during training; pure greedy otherwise
    if self.train and np.random.random() < self.epsilon:
        action_idx = np.random.randint(len(ACTIONS))
    else:
        action_idx = torch.argmax(action_probs, dim=1).item()

    action = ACTIONS[action_idx]
    if action == 'BOMB':
        action = np.random.choice(['UP','DOWN','LEFT','RIGHT'])  # 随机换一个移动


    return ACTIONS[action_idx]


def state_to_features(game_state: dict) -> np.ndarray:
    """
    Convert game_state dict to a flat feature vector.

    Channels (each 17x17):
      0 - walls (field == 1)
      1 - crates (field == -1)
      2 - coins
      3 - bombs (value = remaining timer / BOMB_TIMER, 0 if no bomb)
      4 - explosions
      5 - self position
      6 - opponents
    Returns shape: (7, 17, 17) flattened to (7*17*17,) = 2023
    """
    if game_state is None:
        return np.zeros(7 * 17 * 17, dtype=np.float32)

    field = game_state['field']          # (17,17): 1=wall, -1=crate, 0=free
    bombs = game_state['bombs']          # [((x,y), timer), ...]
    explosion_map = game_state['explosion_map']
    coins = game_state['coins']          # [(x,y), ...]
    self_x, self_y = game_state['self'][3]
    others = game_state['others']        # [(name, score, has_bomb, (x,y)), ...]

    channels = np.zeros((7, 17, 17), dtype=np.float32)

    channels[0] = (field == 1).astype(np.float32)    # walls
    channels[1] = (field == -1).astype(np.float32)   # crates

    for (cx, cy) in coins:
        channels[2][cx, cy] = 1.0

    for ((bx, by), timer) in bombs:
        channels[3][bx, by] = timer / 4.0             # normalize by BOMB_TIMER=4

    channels[4] = (explosion_map > 0).astype(np.float32)

    channels[5][self_x, self_y] = 1.0

    for (_, _, _, (ox, oy)) in others:
        channels[6][ox, oy] = 1.0

    return channels.flatten()