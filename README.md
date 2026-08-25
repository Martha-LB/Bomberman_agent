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
bomberman_rl/
├── agent_code/               # all agents live here
│   ├── my_agent3/            ← my final submission
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

## My Agent: `my_agent3`

Uses a **Dueling DQN** on top of hand-engineered 34-dim features. The features encode BFS pathfinding results, danger maps, and a rule-based "best target" hint — so the network only needs to learn tactical trade-offs, not perception.

See [`agent_code/my_agent3/README.md`](agent_code/my_agent3/README.md) for the full design.

## Key Findings So Far

- **Feature engineering matters more than algorithm choice.** A well-designed 34-dim feature with Q-learning outperforms a naive DQN on raw pixels.
- **BFS with danger awareness is essential.** Without it, the agent cannot reliably distinguish "safe path" from "walkable path".
- **Symmetry augmentation (12×) is a free win.** The map is symmetric under rotation and reflection; each transition can teach the network 12 variants.
- **Reward shaping needs care.** Sparse framework events (`COIN_COLLECTED`, `KILLED_SELF`) aren't enough. Custom events like `MOVE_TO_TARGET` and `MOVE_TO_DEAD`, informed by the rule engine, are what let the agent learn quickly.
- **Curriculum learning helps.** Training progressively (`coin-heaven` → `loot-crate` → `classic`) is more stable than throwing the full scenario at a fresh agent.

## Quick Start

**Setup:**
```bash
conda create -n bomberman python=3.11
conda activate bomberman
pip install torch numpy matplotlib pygame tqdm
```

**Train:**
```bash
python main.py play --agents my_agent --train 1 --scenario coin-heaven --no-gui --n-rounds 1000
```

**Play (with GUI):**
```bash
python main.py play --agents my_agent rule_based_agent --scenario classic
```

**Evaluate against opponents:**
```bash
python main.py play --agents my_agent rule_based_agent random_agent peaceful_agent \
    --scenario classic --no-gui --n-rounds 100 --save-stats
```

**Visualize results:**
```bash
python compare_agents.py results/<latest>.json
```

**Save a replay:**
```bash
python main.py play --agents my_agent3 --scenario classic --n-rounds 1 --save-replay
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
