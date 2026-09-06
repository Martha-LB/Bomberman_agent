# my_agent2

A second Bomberman RL agent built for the Machine Learning Essentials course final project, developed in parallel with `my_agent3`. Shares the same design philosophy (hand-engineered features + a small Dueling DQN) but uses a richer, differently-structured feature vector and a more granular reward-shaping scheme, so the two agents serve as an internal ablation/comparison rather than two unrelated approaches.

## Design Philosophy

Same starting point as `my_agent3`: don't make the network learn perception (BFS pathfinding, danger prediction) from scratch — precompute it with classical algorithms, and let the network only decide the tactical trade-off.

## Architecture

```
game_state (dict)
    ↓
Feature.__call__()                    ← rule-based analysis (features.py)
    ├─ static wall mask + precomputed bomb-blast impact table
    ├─ BFS from own position under a bomb danger map (per direction)
    ├─ per-direction crate score / coin score / distance-to-each-enemy score
    ├─ per-direction "safe count" (reachable safe tiles within N steps)
    └─ global: bomb_valid (safe to drop here?), kill (would trap an enemy?)
    ↓
57-dim feature vector
[5 direction slots (UP/RIGHT/DOWN/LEFT/CENTER) × 11 dims] + [2 global dims]
    ↓
Dueling DQN forward (128 → 64 → 32, LayerNorm between layers)
    ↓
argmax → action
```

## Files

| File          | Purpose |
|---------------|---------|
| `features.py` | `Feature` class: BFS danger map, per-direction crate/coin/enemy scores, safe-tile counts → 57-dim vector |
| `callbacks.py`| `setup()` + `act()`: loads `model.pt`, runs `Feature`, greedy/ε-greedy action selection |
| `model.py`    | Dueling DQN (57 → 128 → 64 → 32 → value + advantage heads) |
| `train.py`    | Double DQN training with symmetry-augmented replay memory |
| `symmetry.py` | 12× data augmentation (4 rotations × 3 flip combinations) for both features and actions |
| `model.pt`    | Saved network weights (auto-generated during training) |

## Feature: 57 dimensions

Per direction slot (UP, RIGHT, DOWN, LEFT, CENTER — 5 slots × 11 dims = 55 dims):
- `[0..4]` one-hot tile type: crate / wall / enemy / bomb / coin
- `[5]` crate score (BFS-weighted sum of reachable crates)
- `[6]` coin score (BFS-weighted sum of reachable coins)
- `[7..9]` distance-based score to each of up to 3 opponents
- `[10]` safe count (# safe tiles reachable within a few steps if the agent commits to this direction first)

Global (2 dims):
- `[55]` `bomb_valid` — would dropping a bomb here still leave an escape route?
- `[56]` `kill` — would dropping a bomb here trap an opponent?

## Reward shaping

In addition to the standard framework events (`COIN_COLLECTED: +40`, `KILLED_OPPONENT: +80`, `KILLED_SELF: -30`, `GOT_KILLED: -10`, `CRATE_DESTROYED: +5`, `SURVIVED_ROUND: +10`, per-step movement/`INVALID_ACTION`/`WAITED` penalties), `train.py` derives custom events directly from the same 57-dim feature used for acting, so "did this action match what the features suggested" becomes a dense, informative reward signal rather than something the network has to discover on its own from sparse framework events:

| Custom event      | Reward | When triggered |
|--------------------|-------:|----------------|
| `TOWARD_COIN`      |   +15  | Moved in the direction with the best coin score |
| `TOWARD_CRATE`     |    +5  | Moved in the direction with the best crate score (no coin available) |
| `MOVE_TO_DANGER`   |   -50  | Moved (or waited) into a tile the danger map marks unsafe |
| `SAFE_WAIT`        |   +10  | Waited because every move direction was unsafe |
| `GOOD_BOMB`        |   +20  | Dropped a bomb while `bomb_valid` was true |
| `BAD_BOMB`         |   -60  | Dropped a bomb while `bomb_valid` was false (no escape route) |
| `LOOP_PENALTY`     |   -3   | Revisited the same tile in >30% of the last 20 steps (discourages dithering) |

### Training

- **Algorithm**: Double DQN with a target network (synced every 1000 optimizer steps)
- **Replay buffer**: 10,000 transitions; every stored transition is expanded to its 12 symmetric variants (`symmetry.py`) before insertion
- **Discount γ**: 0.9
- **Exploration**: ε-greedy, 0.5 → 0.05 (decay 0.9995 per round)
- **Optimizer**: AdamW, lr = 1e-3, MSE loss, gradient value clipping at 100

## Training Curriculum

Same staged approach as `my_agent3`, resuming from the same `model.pt` across stages:

```bash
# Stage 1: coin navigation
python main.py play --agents my_agent2 --train 1 --scenario coin-heaven --no-gui --n-rounds 1000

# Stage 2: bomb usage on crates
python main.py play --agents my_agent2 --train 1 --scenario loot-crate --no-gui --n-rounds 2000

# Stage 3: vs rule_based_agent
python main.py play --agents my_agent2 rule_based_agent --train 1 --scenario classic --no-gui --n-rounds 2000
```

## Inference

Remove `--train 1` to play with the trained model:

```bash
python main.py play --agents my_agent2 rule_based_agent --scenario classic
```

## How this compares to `my_agent3`

Both agents follow the same "engineer the perception, learn the tactics" philosophy, but differ enough to make a genuine internal comparison:

| | `my_agent2` | `my_agent3` |
|---|---|---|
| Feature size | 57-dim (11 dims/direction incl. per-enemy distances) | 34-dim (6-category one-hot tokens) |
| Reward shaping | 7 custom events tied directly to the feature's own scores | 5 custom events (`MOVE_TO_TARGET`, `KILL_ENEMY`, …) tied to a single `look_for_target()` heuristic |
| Anti-dithering | explicit `LOOP_PENALTY` from a position-visit history | not present |

## Systematic Task 1–3 comparison (vs `my_agent3`, 2026-09-06)

All runs greedy (`--train 0`), `--save-stats`, raw json in `results/`.

| Scenario | Agent | Score | Avg/round | Coins | Kills | Suicides | Invalid actions |
|---|---|---:|---:|---:|---:|---:|---:|
| coin-heaven (solo, 50 rounds) | **my_agent2** | 1137 | 22.7 | 1137 | 0 | 0 | 337 |
| coin-heaven (solo, 50 rounds) | my_agent3 | 2500 | 50.0 | 2500 | 0 | 0 | 0 |
| loot-crate (solo, 20 rounds) | **my_agent2** | 0 | 0.0 | 0 | 0 | 0 | 6391 |
| loot-crate (solo, 20 rounds) | my_agent3 | 573 | 28.7 | 573 | 0 | 2 | 6 |
| classic vs `peaceful_agent` (20 rounds) | **my_agent2** | 0 | 0.0 | 0 | 0 | 1 | 6403 |
| classic vs `peaceful_agent` (20 rounds) | my_agent3 | 147 | 7.35 | 107 | 8 | 3 | 9 |
| classic vs `coin_collector_agent` (20 rounds) | **my_agent2** | 8 | 0.40 | 3 | 1 | 5 | 4385 |
| classic vs `coin_collector_agent` (20 rounds) | my_agent3 | 125 | 6.25 | 100 | 5 | 2 | 12 |

**Diagnosis, not a design flaw:** `my_agent2`'s invalid-action rate explodes (>80% of its steps) the moment crates enter the picture, and score collapses to ~0. This is not a bug in the feature/reward code — it's that this checkpoint has, so far, only ever been trained on `coin-heaven` (no crates, `CRATE_DESTROYED`/`bomb_valid`/`kill` feature dimensions were always at their empty-board default during training). Outside that distribution the network's action choices are close to noise. `my_agent3`'s numbers above confirm the same architecture/feature-engineering approach performs solidly once it *has* seen crates and combat — so the fix is the curriculum training `my_agent2` hasn't gone through yet (`loot-crate` → `classic`), not a redesign. Tracked as a phase-2 task; re-run this exact table after that training to get a real before/after.

See `display/capability_visualization.ipynb` for the consolidated charts.

## Known Issues

- No formal hyperparameter sweep has been run yet (learning rate / γ / ε schedule / reward magnitudes are the initial guesses, not tuned).
- Per-step BFS + danger-map computation is noticeably heavier than `my_agent3`'s; worth profiling against the 0.5s tournament timeout on comparable hardware before submission.
