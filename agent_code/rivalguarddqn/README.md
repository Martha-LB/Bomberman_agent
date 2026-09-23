# RivalGuard DQN

`RivalGuard DQN` is a dueling Double DQN agent for the classic Bomberman scenario.
It uses a learned policy together with an Opponent-Aware Safety Filter.

## Method

We use 108 input features.

The neural network has a shared `128 -> 64 -> 32` body, followed by separate
value and advantage heads. Training uses Double DQN, five-step returns, a
target network and D4 board-symmetry augmentation.

We use OASF to remove actions that are invalid or have no simulated escape route. It could
also checks whether an opponent could block an important first move. OASF is turn off during training and turn on during evaluation and normal play.

## Evaluation

The saved model produced the following result in evaluation:

| Agent | Win rate | Average score |
| --- | ---: | ---: |
| **RivalGuard DQN** | **0.5400** | **5.0567 +/- 0.1447** |
| rule_based_agent_0 | 0.1467 | 2.3633 +/- 0.0551 |
| rule_based_agent_1 | 0.1933 | 2.6567 +/- 0.0987 |
| rule_based_agent_2 | 0.2233 | 2.8600 +/- 0.0436 |

The win rate counts both outright wins and ties for first place. Scores show the mean and standard deviation over three runs.

## Curriculum learning

Exploration and learning rate restart at a smaller value for every new stage:

| Stage | Epsilon | Learning rate |
| --- | --- | --- |
| P1 | 1.00 -> 0.05 | 2e-4 -> 3e-5 |
| P2 | 0.20 -> 0.05 | 2e-5 -> 8e-6 |
| P3 | 0.10 -> 0.03 | 1e-5 -> 4e-6 |
| P4 | 0.06 -> 0.02 | 5e-6 -> 2e-6 |

The schedule continues from its saved position when the same stage is resumed.
It starts from the new stage's initial values when `MY_AGENT_STAGE` changes.

The opponent mix follows the Agent8 advanced curriculum:

| Stage | Curriculum mix |
| --- | --- |
| P1 | 35% static, 35% peaceful, 20% evasive, 8% random, 2% coin collector |
| P2 | 4% static, 4% peaceful, 42% evasive, 10% random, 25% coin collector, 15% rule |
| P3 | 2% static, 3% peaceful, 15% evasive, 10% random, 40% coin collector, 30% rule |
| P4 | 7% evasive, 3% random, 20% coin collector, 70% rule |

```bash
MY_AGENT_STAGE=P1 \
MY_AGENT_FRESH=1 \
MY_AGENT_CURRICULUM_SEED=11000 \
python main.py play --no-gui \
  --agents rivalguarddqn curriculum_agent curriculum_agent curriculum_agent \
  --train 1 \
  --scenario classic \
  --n-rounds 600
```

```bash
MY_AGENT_STAGE=P2 \
MY_AGENT_CURRICULUM_SEED=12000 \
python main.py play --no-gui \
  --agents rivalguarddqn curriculum_agent curriculum_agent curriculum_agent \
  --train 1 \
  --scenario classic \
  --n-rounds 400
```

```bash
MY_AGENT_STAGE=P3 \
MY_AGENT_CURRICULUM_SEED=13000 \
python main.py play --no-gui \
  --agents rivalguarddqn curriculum_agent curriculum_agent curriculum_agent \
  --train 1 \
  --scenario classic \
  --n-rounds 400
```

```bash
MY_AGENT_STAGE=P4 \
MY_AGENT_CURRICULUM_SEED=14000 \
python main.py play --no-gui \
  --agents rivalguarddqn curriculum_agent curriculum_agent curriculum_agent \
  --train 1 \
  --scenario classic \
  --n-rounds 400
```
