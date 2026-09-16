# my_agent_planning

A short-horizon, model-based lookahead agent wrapped around **`my_agent3`'s already-trained Dueling DQN**. Where `my_agent`/`my_agent2`/`my_agent3` each learn (or hand-engineer-then-learn) a full policy, this agent adds nothing new to *learn* — it reuses `my_agent3`'s existing policy net verbatim, and adds a **rules-based safety net in front of it**, built by simulating the known game dynamics forward a couple of ticks.

## Why "model-based" here means simulation, not a learned world model

The Bomberman rules (how bombs count down, how explosions propagate, when crates are destroyed, which tiles are walkable) are **already known exactly** — they're just Python code in `environment.py`. Learning a neural world model to predict "what happens next" would mean approximately re-deriving something we already have for free, and worse. So the "model" in "model-based" here is the literal game-rule simulator, reused directly, not a learned approximation of it.

Concretely, this agent's simulation reuses `my_agent3.features.Feature._predict_next()` / `_trigger_explosion()` / `bomb_impact` (`agent_code/my_agent3/features.py`) — the same primitives `my_agent3` already uses internally for its own one-step "is this safe" checks (`_is_safe_to_drop`, `_find_safe_positions`, and the `_is_chance_to_kill()` we fixed two bugs in earlier this session). The only genuinely new code is `simulate_action()` in this folder's `callbacks.py`, which adds self-movement and bomb-placement on top — `_predict_next()` on its own only advances existing bombs/explosions, it doesn't move the agent.

## Architecture

```
game_state (dict)
    ↓
_q_values()                              ← ONE Feature() call + ONE forward pass
    my_agent3's own Q(s, a) for all 6 actions, on the CURRENT real state
    (identical to what my_agent3.act() itself would use to decide)
    ↓
for each action a1 in [UP, RIGHT, DOWN, LEFT, WAIT, BOMB]:
    simulate_action(state, a1)           ← known-rules simulation, 1 tick
        ├─ died this tick?               → veto (score = DEATH_PENALTY)
        ├─ every 2nd-ply follow-up dies? → veto: provably trapped (score = DEATH_PENALTY)
        └─ otherwise                     → score = q_values[a1]   (my_agent3's own estimate)
    ↓
argmax over surviving actions' scores
    ↓
action
```

## Files

| File | Purpose |
|---|---|
| `callbacks.py` | `setup()` (loads `my_agent3`'s `model.pt`, read-only) + `act()`; the simulation (`simulate_action`, `_step_died`, `_has_safe_followup`) and the veto+rank decision (`_choose_action`) |
| `README.md` | this file |

No `train.py`, no `model.py`, no `model.pt` of its own, deliberately. This agent doesn't learn — see "Relationship to `my_agent3`" below.

## The search, precisely

1. Compute `Q(s, ·)` for the current real state **once** — a single `Feature()` call + a single forward pass through `my_agent3`'s loaded network. This is exactly what `my_agent3.act()` computes and takes the argmax of on its own.
2. For each of the 6 actions, simulate one tick forward (`simulate_action`) using the known rules:
   - Movement legality mirrors the real engine's `tile_is_free()` check (free tile, no wall/crate, no other agent, no bomb sitting on it); an illegal move is treated the same way the real engine treats it — the agent just stays put.
   - Bomb countdown / explosion propagation / crate destruction reuse `Feature._predict_next()` verbatim.
   - Death is checked against the **pre-step** state (was the destination already on fire, or about to be hit by a bomb reaching timer 0 this exact tick) — this is the correct point to check, since a post-step explosion-map read (after `_predict_next`'s internal decrement) can under-count residual danger by one tick.
3. An action that dies this tick, or (at search depth 2) whose every possible follow-up action *also* dies, is vetoed with a large negative score — this is a provable, rules-only fact, no learning involved.
4. Among the actions that survive, pick the one with the highest `Q(s, a)` from step 1.

**Simplification, disclosed rather than hidden:** opponents are treated as static obstacles for the whole lookahead — their movement isn't simulated. This matches an assumption `my_agent2`/`my_agent3` already make in their own 1-step lookahead helpers, and keeps the search cheap enough to comfortably fit the 0.5s per-decision budget (empirically ~0.01–0.05s/decision; see "Performance" below). It also means the safety net is most reliable against static hazards (bombs, crates, explosions) and less precise about a fast-moving opponent closing in — a real limitation worth naming in the report rather than glossing over.

## An early bug this design caught (worth keeping in the report)

The first working version of this agent scored each candidate action `a1` by simulating forward and computing `max_a' Q(simulate(s, a1), a')` — the network's *own* best estimate for the state one step further out, rather than the network's direct `Q(s, a1)`. In testing (coin-heaven, no fixed seed) this made the agent walk into a wall and get stuck there for an entire 400-step round, because "bump into a wall and stay put" and "WAIT" produce an *identical* resulting state, and an imperfectly-trained network can rank that identical stay-put state above a state reached by actually moving toward a coin — there's no guarantee `Q(s,a)` and `max_a' Q(s', a')` agree for a network that hasn't fully converged to Bellman-consistency. Switching to reading `Q(s, a1)` directly (one forward pass on the real current state, indexed per action) fixed it immediately and also cut the per-decision cost roughly 10x, since it no longer needs a `Feature()` + forward pass per candidate action.

## Relationship to `my_agent3`

This agent has **no training loop and no separate `model.pt`** — it always loads `agent_code/my_agent3/model.pt` directly (read-only). Retraining `my_agent3` automatically improves this agent's decisions too, with zero extra work. Don't pass `--train` to `my_agent_planning` — there's nothing here to train.

Framed as an ablation: the difference between running `my_agent3` alone and running `my_agent_planning` is *exactly* the rules-based veto layer above — same network, same weights, same feature encoding, same action-ranking criterion. Any behavioral difference between the two is attributable to the search, which makes for a clean comparison in the report.

## Usage

```bash
# inference only - there is nothing to train
python main.py play --agents my_agent_planning rule_based_agent --scenario classic
```

Requires `agent_code/my_agent3/model.pt` to already exist (i.e., `my_agent3` must have been trained at least once).

## Evaluation vs. `my_agent3` alone (2026-09-06, single run each, `--train 0 --save-stats`)

| Scenario | Agent | Score | Suicides | Kills |
|---|---|---:|---:|---:|
| coin-heaven (solo, 50 rounds) | my_agent3 | 2500 (perfect) | 0 | – |
| coin-heaven (solo, 50 rounds) | **my_agent_planning** | 2500 (perfect) | 0 | – |
| loot-crate (solo, 20 rounds) | my_agent3 | 573 | 2 | – |
| loot-crate (solo, 20 rounds) | **my_agent_planning** | 513 | 5 | – |
| classic vs `peaceful_agent` (20 rounds) | my_agent3 | 147 | 3 | 8 |
| classic vs `peaceful_agent` (20 rounds) | **my_agent_planning** | 163 | 1 | 10 |
| classic vs `coin_collector_agent` (20 rounds) | my_agent3 | 125 | 2 | 5 |
| classic vs `coin_collector_agent` (20 rounds) | **my_agent_planning** | 126 | 1 | 6 |

**Honest read, not a cherry-picked one:** in the two scenarios with an opponent to fight, the safety net measurably helps — fewer suicides (1 vs 2–3) and more kills, and a modest score improvement. In coin-heaven it's identical (nothing to veto — no bombs anywhere in that scenario). In loot-crate, though, it's *worse* than `my_agent3` alone on this single run — both lower score and *more* suicides, which is the opposite of what a safety layer should do. This is a real, open result, not swept under the rug: candidate explanations to check before trusting it as a genuine regression (rather than run-to-run noise, since none of these runs used a fixed `--seed`) are (a) the 2-ply veto occasionally forcing a detour away from a self-placed bomb's blast radius that `my_agent3` alone would have judged (correctly, on the training distribution) as safe to stay near, and (b) plain variance — 20 rounds is not a lot of samples. Re-running loot-crate a few more times with different seeds before writing this into the report is the obvious next step.

## Known Issues

- Opponents are treated as static during the lookahead (see "Simplification" above) — the veto is precise against bomb/crate/explosion hazards but not against a moving enemy.
- The loot-crate regression above needs a few more seeded runs before it's reported as a real effect rather than noise.
- Search depth is hardcoded to 2 (`SEARCH_DEPTH` in `callbacks.py`); it was not tuned against depth 1 or depth 3 for a speed/safety tradeoff.
- Shares `my_agent3`'s own known issues by construction (see that agent's README) since it reuses its exact network and feature encoding — e.g. the CUDA→CPU unpickling workaround and the unpinned `torch` version in `requirements.txt`.
