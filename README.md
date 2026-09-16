# Bomberman Reinforcement Learning

Building a Reinforcement Learning agent to play the classic **Bomberman** game.

## The Task

Bomberman is a grid-based multi-agent game where up to 4 players compete to:
- Collect coins for points
- Destroy crates (some hide coins)
- Bomb opponents to eliminate them

Each round lasts up to 400 steps on a 17×17 grid. The winner is whoever scores the highest. Actions are one of `UP / DOWN / LEFT / RIGHT / WAIT / BOMB`, and each step must return in under 0.5 seconds.

The framework is provided by [ukoethe/bomberman_rl](https://github.com/ukoethe/bomberman_rl); this project focuses on building a competitive agent within it.

## Repository Structure

```
bomberman/
├── agent_code/               # all agents live here
│   ├── rivalguarddqn/        ← final submission
│   ├── rule_based_agent/     ← framework baseline
│   ├── random_agent/         ← framework baseline
│   └── ...
├── environment.py            # game engine
├── main.py                   # entry point (train / play / replay)
├── settings.py               # game constants (grid size, bomb power, etc.)
├── events.py                 # event definitions used for reward shaping
├── results/                  # per-match statistics (JSON)
├── logs/                     # game logs
├── replays/                  # saved games for playback
└── compare_agents.py         # visualization script for agent comparison
```

## Final Agent: `rivalguarddqn`

RivalGuard DQN uses a **Dueling Double DQN** with 108 state and action-consequence features. These features describe coins, crates, opponents, danger timing, escape capacity, recent movement, and coin competition. An Opponent-Aware Safety Filter removes invalid or provably fatal actions; the learned Q-network ranks the remaining actions.

See [`agent_code/rivalguarddqn/README.md`](agent_code/rivalguarddqn/README.md) for the full design.

## Key Findings So Far

- **Action-consequence features are more useful than a hand-written target action.** The network learns how to trade off coins, crates, combat, and safety.
- **Time-aware escape search matters.** A tile can be walkable now but still lead to a forced death several steps later.
- **D4 symmetry augmentation improves data efficiency.** Rotations and reflections teach equivalent board situations together.
- **Potential-based reward shaping provides dense feedback while keeping the game score as the main objective.**
- **Curriculum training with replay rehearsal helps retain earlier skills.**

## Quick Start

**Setup:**
```bash
conda create -n bomberman python=3.11
conda activate bomberman
pip install torch numpy matplotlib pygame tqdm
```

**Train:**
```bash
MY_AGENT_STAGE=P1 MY_AGENT_FRESH=1 \
python main.py play --agents rivalguarddqn curriculum_agent curriculum_agent curriculum_agent \
  --train 1 --scenario classic --no-gui --n-rounds 600
```

**Play (with GUI):**
```bash
python main.py play --agents rivalguarddqn rule_based_agent --scenario classic
```

**Evaluate against opponents:**
```bash
python main.py play --agents rivalguarddqn rule_based_agent random_agent peaceful_agent \
    --scenario classic --no-gui --n-rounds 100 --save-stats
```

**Visualize results:**
```bash
python compare_agents.py results/<latest>.json
```

**Save a replay:**
```bash
python main.py play --agents rivalguarddqn --scenario classic --n-rounds 1 --save-replay
python main.py replay replays/<file>.pt
```

## Scenarios

Defined in `settings.py`, from easiest to hardest for training:

| Scenario     | Crates | Coins | Use case |
|--------------|-------:|------:|----------|
| `empty`      |     0  |    0  | Debugging movement |
| `coin-heaven`|     0  |   50  | Learn navigation |
| `loot-crate` |  75%   |   50  | Learn bomb usage |
| `classic`    |  75%   |    9  | Tournament format |

## Framework Notes

- **Timeout**: 0.5 s per step in play mode; unlimited during training
- **Log level**: set in `settings.py` (`LOG_GAME`, `LOG_AGENT_CODE`)
- **Reproducibility**: `--seed <int>` fixes randomness
