import os
import numpy as np
import torch
import torch.optim as optim
from collections import deque
from pathlib import Path

from .model import PPONetwork, N_ACTIONS, ACTIONS
from .callbacks import state_to_features
import events as e

# always save relative to this file, not the cwd
AGENT_DIR = Path(__file__).parent
MODEL_PATH = AGENT_DIR / "model.pt"

# ── Hyperparameters ───────────────────────────────────────────────
LR              = 3e-4
GAMMA           = 0.99      # discount factor
GAE_LAMBDA      = 0.95      # GAE smoothing
CLIP_EPS        = 0.2       # PPO clip range
ENTROPY_COEF = 0.05      # encourage exploration
VALUE_COEF      = 0.5       # critic loss weight
MAX_GRAD_NORM   = 0.5
UPDATE_EPOCHS   = 4         # PPO epochs per update
BATCH_SIZE      = 64
ROLLOUT_LEN     = 400       # collect full episode before update

EPSILON_START   = 1.0
EPSILON_END     = 0.05
EPSILON_DECAY = 0.9998

SAVE_EVERY      = 100       # save model every N rounds


# ── Reward shaping ────────────────────────────────────────────────
def shape_reward(events_list, old_state, new_state, action_history=[]):
    reward = 0.0

    # 只保留最核心的信号，其他全去掉
    if e.COIN_COLLECTED  in events_list: reward += 5.0
    if e.INVALID_ACTION  in events_list: reward -= 1.0   # 撞墙才罚
    if e.WAITED          in events_list: reward -= 0.0   # 不罚等待
    if e.KILLED_SELF     in events_list: reward -= 3.0

    # 靠近金币是唯一 shaping
    if old_state is not None and new_state is not None:
        reward += _coin_proximity_delta(old_state, new_state)

    return reward

def _crate_proximity_delta(old_state, new_state):
    """靠近箱子也给小奖励（为了炸开金币）"""
    field = old_state['field']
    crates = list(zip(*np.where(field == -1)))
    if not crates:
        return 0.0
    sx_old, sy_old = old_state['self'][3]
    sx_new, sy_new = new_state['self'][3]
    dists_old = [abs(cx-sx_old) + abs(cy-sy_old) for (cx,cy) in crates]
    dists_new = [abs(cx-sx_new) + abs(cy-sy_new) for (cx,cy) in crates]
    delta = min(dists_old) - min(dists_new)
    return 0.1 * delta


def _coin_proximity_delta(old_state, new_state):
    coins = old_state.get('coins', [])
    if not coins:
        return 0.0
    sx_old, sy_old = old_state['self'][3]
    sx_new, sy_new = new_state['self'][3]
    dists_old = [abs(cx-sx_old) + abs(cy-sy_old) for (cx,cy) in coins]
    dists_new = [abs(cx-sx_new) + abs(cy-sy_new) for (cx,cy) in coins]
    delta = min(dists_old) - min(dists_new)
    return 0.3 * delta   # 0.1 → 0.3


# ── PPO rollout buffer ────────────────────────────────────────────
class RolloutBuffer:
    def __init__(self):
        self.clear()

    def clear(self):
        self.obs, self.actions, self.log_probs = [], [], []
        self.rewards, self.values, self.dones   = [], [], []

    def add(self, obs, action, log_prob, reward, value, done):
        self.obs.append(obs)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)

    def compute_returns(self, last_value, gamma, lam):
        """Generalized Advantage Estimation (GAE)."""
        advantages = []
        gae = 0.0
        values = self.values + [last_value]
        for t in reversed(range(len(self.rewards))):
            delta = self.rewards[t] + gamma * values[t+1] * (1 - self.dones[t]) - values[t]
            gae   = delta + gamma * lam * (1 - self.dones[t]) * gae
            advantages.insert(0, gae)
        returns = [adv + val for adv, val in zip(advantages, self.values)]
        return advantages, returns


# ── setup_training ────────────────────────────────────────────────
def setup_training(self):
    self.device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    self.model    = PPONetwork().to(self.device)
    self.optimizer = optim.Adam(self.model.parameters(), lr=LR)
    self.buffer   = RolloutBuffer()
    self.epsilon  = EPSILON_START
    self.round_count = 0
    self.action_history = []

    # tracking
    self.episode_rewards = []
    self.recent_rewards  = deque(maxlen=100)

    self.logger.info(f"Training on {self.device}")


# ── game_events_occurred ──────────────────────────────────────────
def game_events_occurred(self, old_game_state, self_action, new_game_state, events):
    if old_game_state is None:
        return

    obs     = state_to_features(old_game_state)
    self.action_history.append(self_action)
    reward = shape_reward(events, old_game_state, new_game_state, self.action_history)

    action_idx = ACTIONS.index(self_action) if self_action in ACTIONS else 0

    obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
    with torch.no_grad():
        action_probs, value = self.model(obs_t)
        dist = torch.distributions.Categorical(probs=action_probs)
        log_prob = dist.log_prob(torch.tensor([action_idx]).to(self.device))

    self.buffer.add(
        obs       = obs,
        action    = action_idx,
        log_prob  = log_prob.item(),
        reward    = reward,
        value     = value.item(),
        done      = 0,
    )


# ── end_of_round ──────────────────────────────────────────────────
def end_of_round(self, last_game_state, last_action, events):
    # handle last transition
    game_events_occurred(self, last_game_state, last_action, None, events)
    self.action_history = []

    # mark episode done
    if self.buffer.dones:
        self.buffer.dones[-1] = 1


    # 诊断：统计这轮做了什么
    if self.round_count % 10 == 0:
        from collections import Counter
        action_counts = Counter(self.buffer.actions)
        action_names  = {i: a for i, a in enumerate(ACTIONS)}
        readable = {action_names[k]: v for k, v in action_counts.items()}
        coins = e.COIN_COLLECTED  # 这轮收了几个金币
        n_coins = sum(1 for r in self.buffer.rewards if r >= 1.9)
        print(f"  actions: {readable} | coins collected: {n_coins}")

    total_reward = sum(self.buffer.rewards)
    self.recent_rewards.append(total_reward)
    self.episode_rewards.append(total_reward)

    # ── PPO update ────────────────────────────────────────────────
    _ppo_update(self)
    self.buffer.clear()

    # decay epsilon
    self.epsilon = max(EPSILON_END, self.epsilon * EPSILON_DECAY)
    self.round_count += 1

    # logging
    if self.round_count % 10 == 0:
        avg = np.mean(self.recent_rewards)
        print(f"[Round {self.round_count}] reward: {total_reward:.2f} | avg100: {avg:.2f} | eps: {self.epsilon:.3f}")
        self.logger.info(
            f"Round {self.round_count} | reward: {total_reward:.2f} "
            f"| avg100: {avg:.2f} | eps: {self.epsilon:.3f}"
        )

    # save
    if self.round_count % SAVE_EVERY == 0:
        torch.save(self.model.state_dict(), MODEL_PATH)
        self.logger.info(f"Model saved at round {self.round_count} → {MODEL_PATH}")


# ── PPO update step ───────────────────────────────────────────────
def _ppo_update(self):
    if len(self.buffer.obs) < 2:
        return

    # compute returns & advantages via GAE
    advantages, returns = self.buffer.compute_returns(
        last_value=0.0, gamma=GAMMA, lam=GAE_LAMBDA
    )

    obs_t   = torch.FloatTensor(np.array(self.buffer.obs)).to(self.device)
    acts_t  = torch.LongTensor(self.buffer.actions).to(self.device)
    lp_old  = torch.FloatTensor(self.buffer.log_probs).to(self.device)
    adv_t   = torch.FloatTensor(advantages).to(self.device)
    ret_t   = torch.FloatTensor(returns).to(self.device)

    # normalize advantages
    adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

    n = len(obs_t)
    self.model.train()

    for _ in range(UPDATE_EPOCHS):
        # mini-batch shuffle
        indices = np.random.permutation(n)
        for start in range(0, n, BATCH_SIZE):
            idx = indices[start:start + BATCH_SIZE]
            if len(idx) < 2:
                continue

            log_probs, entropy, values = self.model.evaluate_actions(obs_t[idx], acts_t[idx])

            # PPO clipped objective
            ratio       = (log_probs - lp_old[idx]).exp()
            clip_ratio  = torch.clamp(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS)
            actor_loss  = -torch.min(ratio * adv_t[idx], clip_ratio * adv_t[idx]).mean()

            # value loss
            critic_loss = ((values.squeeze() - ret_t[idx]) ** 2).mean()

            loss = actor_loss + VALUE_COEF * critic_loss - ENTROPY_COEF * entropy.mean()

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), MAX_GRAD_NORM)
            self.optimizer.step()

    self.model.eval()


