# my_agent

`RivalGuard DQN` is a dueling Double DQN agent for the classic Bomberman scenario.
It uses a learned policy together with an Opponent-Aware Safety Filter (OASF).

## Method

The policy uses 108 input features. They describe:

- nearby tiles and valid actions;
- bombs, explosions and possible escape routes;
- coins, crates and expected crate value;
- opponents and coin competition;
- recent movement and the current game state.

The neural network has a shared `128 -> 64 -> 32` body, followed by separate
value and advantage heads. Training uses Double DQN, five-step returns, a
target network and D4 board-symmetry augmentation.

OASF removes actions that are invalid or have no simulated escape route. It
also checks whether an opponent could block an important first move. OASF is
automatically disabled during training and enabled during evaluation and normal
play. No environment variable is needed to control it.

## Files

- `callbacks.py`: loads the model and selects actions.
- `feature.py`: builds the 108 features and applies OASF.
- `model.py`: defines and loads the neural network.
- `symmetry.py`: contains board transformations used in training.
- `train.py`: contains the training and replay logic.
- `model.pt`: contains the final trained policy.

## Run one game

Run this command from the project root:

```bash
python main.py play \
  --agents rivalguarddqn rule_based_agent rule_based_agent rule_based_agent \
  --scenario classic \
  --n-rounds 1
```

## Evaluation

```bash
python evaluate_agents.py \
  --agent rivalguarddqn \
  --opponents rule_based_agent rule_based_agent rule_based_agent \
  --scenario classic \
  --rounds 100 \
  --runs 3 \
  --seed 40000 \
  --output results/eval_my_agent_r100x3_seed40000.json
```

The saved model produced the following result in this evaluation:

| Agent | Win rate | Average score |
| --- | ---: | ---: |
| **RivalGuard DQN** | **0.5400** | **5.0567 +/- 0.1447** |
| rule_based_agent_0 | 0.1467 | 2.3633 +/- 0.0551 |
| rule_based_agent_1 | 0.1933 | 2.6567 +/- 0.0987 |
| rule_based_agent_2 | 0.2233 | 2.8600 +/- 0.0436 |

The win rate counts both outright wins and ties for first place. Scores show
the mean and standard deviation over three runs. The game is stochastic, so a
new evaluation may give a different result.

## Curriculum training

Training is optional. The default curriculum is the `advanced` curriculum used
for the final model. Three `curriculum_agent` opponents independently sample a
new policy at the start of each round.

`MY_AGENT_STAGE` controls both the learner and the curriculum opponents. The
share of stronger `bomber` and `rule` opponents is 40% in P2, 70% in P3 and
90% in P4.

Start P1 from a new model:

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

Continue with P2:

```bash
MY_AGENT_STAGE=P2 \
MY_AGENT_CURRICULUM_SEED=12000 \
python main.py play --no-gui \
  --agents rivalguarddqn curriculum_agent curriculum_agent curriculum_agent \
  --train 1 \
  --scenario classic \
  --n-rounds 400
```

Continue with P3 and P4 in the same way:

```bash
MY_AGENT_STAGE=P3 \
MY_AGENT_CURRICULUM_SEED=13000 \
python main.py play --no-gui \
  --agents rivalguarddqn curriculum_agent curriculum_agent curriculum_agent \
  --train 1 \
  --scenario classic \
  --n-rounds 400

MY_AGENT_STAGE=P4 \
MY_AGENT_CURRICULUM_SEED=14000 \
python main.py play --no-gui \
  --agents rivalguarddqn curriculum_agent curriculum_agent curriculum_agent \
  --train 1 \
  --scenario classic \
  --n-rounds 400
```

Only P1 should use `MY_AGENT_FRESH=1`. Later stages start from the previous
stage's best `model.pt`.

During training, `training_state.pt` stores the latest policy and a compressed
replay reservoir with at most 6,000 transitions. In later stages, 25% of each
training batch comes from this older experience. This helps retain earlier
skills.

The agent checks its rolling performance over 100 rounds. When the current
stage improves, its best policy is saved directly to `model.pt`.
`training_state.pt` is only needed for training and is not required for the
final submission.
