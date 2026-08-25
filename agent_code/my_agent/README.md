# my_ppo_agent

PPO agent for bomberman-rl competition.

## Setup

```bash
conda create -n bomberman python=3.11
conda activate bomberman
pip install torch numpy
```

## File structure

```
agent_code/
└── my_ppo_agent/
    ├── __init__.py
    ├── callbacks.py   ← act() and state_to_features()
    ├── train.py       ← PPO update, reward shaping
    ├── model.py       ← neural network definition
    └── model.pt       ← saved weights (auto-generated)
```

## Training

```bash
# Stage 1: learn to collect coins (no enemies)
python main.py play --agents my_ppo_agent --train 1 --scenario coin-heaven --no-gui n-rounds 500

# Stage 2: add crates
python main.py play --agents my_ppo_agent --train 1 --scenario loot-crate --no-gui --n-rounds 500

# Stage 3: vs random enemy
python main.py play --agents my_ppo_agent random_agent --train 1 --scenario classic --no-gui --n-rounds 1000

# Stage 4: vs rule-based agent
python main.py play --agents my_ppo_agent rule_based_agent --train 1 --scenario classic --no-gui --n-rounds 1000

# Stage 5: self-play (run two copies)
python main.py play --agents my_ppo_agent my_ppo_agent --train 1 --scenario classic --no-gui --n-rounds 2000
```

## Watch trained agent

```bash
python main.py play --agents my_ppo_agent rule_based_agent --scenario classic
```

## Tips

- still many problem to solve.
- not complete.