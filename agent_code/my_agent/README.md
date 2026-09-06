# my_agent (PPO)

The team's first Bomberman RL agent, built around Proximal Policy Optimization (PPO) with an actor-critic network. Where `my_agent2`/`my_agent3` hand-engineer perception (BFS pathfinding, bomb-danger maps, kill-opportunity detection) and only let a small Dueling DQN decide the tactical trade-off, `my_agent` goes the other direction: it hands the network a raw multi-channel encoding of the board and lets it learn perception *and* tactics end-to-end. That makes it the natural "does hand-engineering actually help?" baseline against the other two agents.

## Design Philosophy

No rule-based preprocessing at all: no BFS, no danger map, no lookahead simulation. The trade-off is explicit — a much simpler feature pipeline, at the cost of a much higher-variance learning signal (the network has to discover on its own that a bomb about to go off is dangerous, purely from reward feedback). PPO's clipped surrogate objective and GAE-smoothed advantages exist specifically to keep that higher-variance signal from destabilizing training, which is why this agent uses a heavier-weight training algorithm than the Double-DQN approach the other two agents use.

## Architecture

```
game_state (dict)
    ↓
state_to_features()                    ← raw multi-channel encoding (callbacks.py)
    7 channels × 17×17, no hand-crafted scores or BFS distances
    ↓
flatten → 2023-dim vector (7 × 17 × 17)
    ↓
PPONetwork shared trunk (2023 → 512 → 256 → 128, ReLU between layers)
    ├─ actor head  (128 → 6 logits) → softmax → action_probs
    └─ critic head (128 → 1)        → state value estimate
    ↓
argmax(action_probs)  [inference]
ε-greedy override     [training only, see "Training" below]
    ↓
action
```

## Setup

```bash
conda create -n bomberman python=3.11
conda activate bomberman
pip install -r requirements.txt
```

## Files

| File | Purpose |
|------|---------|
| `callbacks.py` | `setup()` + `act()`: loads `model.pt`, builds the 2023-dim raw-channel encoding via `state_to_features()`, greedy/ε-greedy action selection |
| `model.py` | `PPONetwork`: shared trunk + actor/critic heads; `get_action_and_value()` (rollout) and `evaluate_actions()` (PPO update) helpers |
| `train.py` | `setup_training()`, `game_events_occurred()`, `end_of_round()`, reward shaping, the `RolloutBuffer` (GAE), and the PPO update step |
| `model.pt` | Saved network weights (auto-generated during training) |

## Feature encoding: 7 channels × 17 × 17

Unlike `my_agent2`'s 57-dim or `my_agent3`'s 34-dim hand-crafted per-direction vectors, `my_agent` just stacks the raw board into 7 binary/scalar channels and flattens:

| Channel | Content |
|---:|---|
| 0 | Crates (`field == 1`) |
| 1 | Walls (`field == -1`) |
| 2 | Coins (one-hot at each coin position) |
| 3 | Bombs — value is `remaining_timer / BOMB_TIMER` (0 where there's no bomb) |
| 4 | Explosions (`explosion_map > 0`) |
| 5 | Own position (one-hot) |
| 6 | Opponent positions (one-hot, all opponents in one channel) |

No danger-timing, no reachability, no distance scores — everything the network knows about "is this tile safe" it has to infer itself from channels 3/4 (bomb timers, explosion map) across training.

## Reward shaping

`shape_reward()` in `train.py` intentionally keeps only a few signals ("只保留最核心的信号，其他全去掉" — deliberately pared down from an earlier, noisier version):

| Event | Reward | Note |
|---|---:|---|
| `COIN_COLLECTED` | +5.0 | |
| `INVALID_ACTION` | −1.0 | Only walking into a wall/crate is penalized |
| `WAITED` | 0.0 | Explicitly not penalized |
| `KILLED_SELF` | −3.0 | |
| Coin-proximity shaping | ±0.3 × Δ(Manhattan distance to nearest coin) | Dense per-step shaping via `_coin_proximity_delta()` |

**Known gap:** `_crate_proximity_delta()` is fully implemented right next to `_coin_proximity_delta()` but is never called from `shape_reward()` — crate-directed shaping is currently a no-op. Worth a deliberate decision (wire it in, or delete the dead function) rather than leaving it as an accident.

## Training: PPO

- **Rollout**: one full episode per update (up to `MAX_STEPS=400`), matching `ROLLOUT_LEN=400`
- **Advantage estimation**: GAE, `γ=0.99`, `λ=0.95`
- **Objective**: clipped surrogate, `CLIP_EPS=0.2`, `UPDATE_EPOCHS=4` passes per rollout, mini-batches of `BATCH_SIZE=64`
- **Entropy bonus**: `ENTROPY_COEF=0.05` (encourages continued exploration through the policy's own stochasticity)
- **Value loss weight**: `VALUE_COEF=0.5`
- **Optimizer**: Adam, `lr=3e-4`, gradient-norm clipped at `MAX_GRAD_NORM=0.5`
- **Checkpointing**: every `SAVE_EVERY=100` rounds

**Design note (not a bug, but non-standard):** on top of PPO's own entropy-driven exploration, `act()` also applies an ε-greedy override during training (`EPSILON_START=1.0 → EPSILON_END=0.05`, decay `0.9998`/round) — if `random() < ε`, a uniformly random action is taken instead of `argmax(action_probs)`. Canonical PPO usually relies solely on sampling from the policy distribution (`dist.sample()`) for exploration; this implementation instead takes the *greedy* action outside of ε, and a *random* action inside it, never actually sampling from the softmax during rollout. This is still internally consistent — `game_events_occurred()` recomputes `log_prob` for whichever action was actually taken from the same `old_state`, so the importance-sampling ratio used by the clipped objective stays mathematically valid — but the two exploration mechanisms (entropy bonus + ε-greedy) are somewhat redundant and it's worth deciding on purpose whether to keep both before the report, rather than it being left over from an earlier DQN-style draft of this agent.

## Training curriculum

```bash
# Stage 1: coin navigation
python main.py play --agents my_agent --train 1 --scenario coin-heaven --no-gui --n-rounds 1000

# Stage 2: bomb usage on crates
python main.py play --agents my_agent --train 1 --scenario loot-crate --no-gui --n-rounds 2000

# Stage 3: vs random_agent
python main.py play --agents my_agent random_agent --train 1 --scenario classic --no-gui --n-rounds 1000

# Stage 4: vs rule_based_agent
python main.py play --agents my_agent rule_based_agent --train 1 --scenario classic --no-gui --n-rounds 2000

# Stage 5 (optional): self-play — run two copies of the same checkpoint against each other
python main.py play --agents my_agent my_agent --train 1 --scenario classic --no-gui --n-rounds 2000
```

## Inference

Remove `--train 1` to play with the trained model:

```bash
python main.py play --agents my_agent rule_based_agent --scenario classic
```

## Bugs found and fixed this session

- **`act()` had dead BOMB-avoidance code that never ran**: a leftover branch reassigned `action` to a random movement whenever the model chose `BOMB`, but then the function's `return` statement used `ACTIONS[action_idx]` (the original choice) instead of the reassigned `action` — so the "avoid bombing" logic silently never took effect. Removed the dead branch; the model's own BOMB choice is now respected as intended.
- **`setup_training()` was discarding the loaded checkpoint on every run**: it unconditionally ran `self.model = PPONetwork().to(self.device)` *after* `callbacks.setup()` had already loaded `model.pt` (if present), silently restarting PPO from scratch every single training invocation. Fixed by removing the re-initialization — `setup_training()` now only builds the optimizer and rollout-tracking state, reusing whatever `self.model` `setup()` already loaded.

## Known Issues

- No formal hyperparameter sweep has been run (PPO's LR/clip range/entropy coefficient/reward weights are initial guesses, not tuned).
- `_crate_proximity_delta()` exists but is unused (see "Reward shaping" above).
- ε-greedy exploration layered on top of PPO's own stochastic/entropy-driven exploration is non-standard (see "Training" above) — not incorrect, but worth a deliberate call.
- Unlike `my_agent2`/`my_agent3`, this agent has not yet been run through the systematic coin-heaven / loot-crate / vs-`peaceful_agent` / vs-`coin_collector_agent` evaluation protocol (see `results/` and those two agents' READMEs) — recommended before the report, so all three agents are compared under the same conditions.
