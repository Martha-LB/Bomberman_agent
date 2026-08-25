# my_agent3

A Bomberman RL agent built for the Machine Learning Essentials course competition at Heidelberg University.

## Design Philosophy

**Expert-engineered features + lightweight neural decision-making.**

Instead of feeding raw map pixels to a large network, this agent pre-computes strategic information (pathfinding, danger evaluation, kill opportunities) using classical algorithms, then lets a small DQN handle only the final tactical trade-offs.

## Architecture

```
game_state (dict)
    ↓
Feature.__call__()                    ← rule-based analysis
    ├─ BFS pathfinding
    ├─ danger map (bomb blast prediction)
    ├─ safe position search
    ├─ look_for_target()              ← weighted scoring of 5 directions
    └─ is_chance_to_kill()            ← trap detection for enemies
    ↓
6-token string feature
[up, right, down, left, center, bomb]
each token: 'block'/'free'/'coin'/'enemy'/'dead'/'target'/'KILL!'
    ↓
encode_feature() → 34-dim one-hot vector
    ↓
Dueling DQN forward
    ↓
argmax → action
```

## Files

| File          | Purpose |
|---------------|---------|
| `features.py` | 6-token structured feature with BFS + `look_for_target` |
| `callbacks.py`| `setup()` + `act()` + `encode_feature()` |
| `model.py`    | Dueling DQN (34 → 128 → 64 → 32 → value + advantage) |
| `train.py`    | Double DQN training with symmetry augmentation |
| `symmetry.py` | 12× data augmentation via rotations and flips |
| `model.pt`    | Saved network weights (auto-generated during training) |

## Key Design Choices

### Feature: 6-token string, then one-hot encoded

Each of 5 direction slots takes one of 6 categories:
`block`, `free`, `coin`, `enemy`, `dead`, `target`

The `target` token is set by `look_for_target()`, which runs BFS from each safe direction and picks the highest-scoring one based on:
- Reachable coins weighted by 1/distance
- Reachable crates (bomb value) weighted by 1/distance
- Nearest enemy (`main_enemy`) weighted by 1/distance
- Escape safety when in danger

The bomb slot takes one of: `True` (bomb available), `False`, `dead` (unsafe to drop), `target` (best to drop), `KILL!` (would trap enemy).

Total encoded: `5 directions × 6 categories + 4 bomb states = 34 dims`.

### Dynamic weights based on danger

`look_for_target()` switches its scoring priorities depending on immediate safety:

```python
if not in danger:
    weights = {coins: 300, main_enemy: 30, other_enemies: -5, crates: 1}
else:
    weights = {coins: 300, main_enemy: 0, other_enemies: -300, escape: 10}
```

This lets the same rule engine act aggressively when safe and defensively when threatened.

### Training

- **Algorithm**: Double DQN with target network (updated every 1000 steps)
- **Replay buffer**: 10,000 transitions with symmetry augmentation (each stored 12 times)
- **Discount γ**: 0.9
- **Exploration**: ε-greedy from 0.5 → 0.1 (decay 0.9995 per round)
- **Optimizer**: AdamW, lr = 1e-3, MSE loss

### Custom events (reward shaping)

| Event               | Reward | When triggered |
|---------------------|-------:|----------------|
| `MOVE_TO_TARGET`    |   +50  | Moved in the direction marked as `target` |
| `MOVE_TO_DEAD`      |  -100  | Moved into a tile marked as `dead` |
| `ATTACK_TARGET`     |   +50  | Dropped bomb when bomb slot is `target` |
| `ATTACK_ENEMY`      |   +20  | Dropped bomb with an enemy in adjacent tile |
| `KILL_ENEMY`        |  +500  | Dropped bomb when kill opportunity was detected |

Plus standard framework events (`COIN_COLLECTED: +10`, `KILLED_SELF: -300`, etc.).

## Training Curriculum

Train through progressive scenarios, resuming from the same `model.pt`:

```bash
# Stage 1: coin navigation
python main.py play --agents my_agent3 --train 1 --scenario coin-heaven --no-gui --n-rounds 1000

# Stage 2: bomb usage on crates
python main.py play --agents my_agent3 --train 1 --scenario loot-crate --no-gui --n-rounds 2000

# Stage 3: 1v1 combat
python main.py play --agents my_agent3 rule_based_agent --train 1 --scenario classic --no-gui --n-rounds 2000

# Stage 4: 4-player free-for-all
python main.py play --agents my_agent3 rule_based_agent rule_based_agent rule_based_agent --train 1 --scenario classic --no-gui --n-rounds 2000
```

## Inference

Remove `--train 1` to play with the trained model:

```bash
python main.py play --agents my_agent3 rule_based_agent rule_based_agent peaceful_agent --scenario classic
```

## Current Performance (100 rounds, classic scenario)

| Agent                     | Total Score | Avg/round | Kills | Coins | Suicides |
|---------------------------|------------:|----------:|------:|------:|---------:|
| **my_agent3**             |    **506**  |  **5.06** | **40**| **306**|   **20** |
| rule_based_agent          |        371  |     3.71  |    41 |   166 |      53  |
| peaceful_agent            |          2  |     0.02  |     0 |     2 |       0  |

## Known Issues & Planned Improvements

**Current problems:**
- Self-kill is still too frequent
- Coin collection rate could improve (guide capability)
