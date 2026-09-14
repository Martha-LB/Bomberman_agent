"""
Custom per-round, per-agent stats harness.

--save-stats only gives:
  - by_round: per-round totals SUMMED ACROSS ALL AGENTS (coins/kills/suicides/steps only)
  - by_agent: per-agent totals SUMMED ACROSS ALL ROUNDS (lifetime_statistics + score)
Neither gives per-agent, per-round breakdowns, which is what's needed for
win_rate / mean_rank / tie_for_first_rate / survival_rate /
score_margin_to_best_opponent / mean_act_time_ms / per-round std of anything.

That data exists transiently inside the framework: environment.py's
GenericWorld.end_round() computes each agent's per-round stats into
agent.statistics (a per-round dict, reset only at the *next* round's
start_round()) and agent.score (this round's score), before folding them into
lifetime_statistics. We monkeypatch end_round to snapshot that per-agent data
every round for the one agent we care about (named via --my-agent), without
touching any checked-in framework file. Confirmed safe: this framework always
runs agents in-process (SequentialAgentBackend), so the monkeypatch applies
directly with no subprocess boundary to cross.
"""
import json
import random
import sys

import numpy as np
import torch

SEED = int(sys.argv[1])
TARGET_AGENT = sys.argv[2]         # e.g. "my_agent3"
OUT_PATH = sys.argv[3]             # where to dump the per-round records (JSON list)
MAIN_ARGV = sys.argv[4:]

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

import environment

_orig_end_round = environment.GenericWorld.end_round
_records = []

def _patched_end_round(self):
    _orig_end_round(self)
    agents = self.agents
    target = next((a for a in agents if a.name == TARGET_AGENT), None)
    if target is None:
        return
    others = [a for a in agents if a is not target]
    scores = {a.name: a.score for a in agents}
    max_score = max(scores.values())
    rank = 1 + sum(1 for a in agents if a.score > target.score)
    is_top = (target.score == max_score)
    tied_at_top = sum(1 for v in scores.values() if v == max_score) > 1
    tie_for_first = is_top and tied_at_top
    win = is_top and not tied_at_top
    stat = target.statistics
    steps = stat.get("steps", 0)
    think_time = stat.get("time", 0.0)
    opp_scores = [a.score for a in others]
    record = {
        "round_id": self.round_id,
        "score": target.score,
        "coins": stat.get("coins", 0),
        "kills": stat.get("kills", 0),
        "suicides": stat.get("suicides", 0),
        "crates": stat.get("crates", 0),
        "bombs": stat.get("bombs", 0),
        "invalid": stat.get("invalid", 0),
        "moves": stat.get("moves", 0),
        "steps": steps,
        "mean_act_time_ms": (think_time / steps * 1000.0) if steps > 0 else None,
        "dead": bool(target.dead),
        "survived": not bool(target.dead),
        "killed_by_enemy": bool(target.dead) and stat.get("suicides", 0) == 0,
        "rank": rank,
        "win": win,
        "tie_for_first": tie_for_first,
        "mean_opponent_score": (sum(opp_scores) / len(opp_scores)) if opp_scores else None,
        "best_opponent_score": max(opp_scores) if opp_scores else None,
        "score_margin_to_best_opponent": (target.score - max(opp_scores)) if opp_scores else None,
    }
    _records.append(record)

environment.GenericWorld.end_round = _patched_end_round

import main
try:
    main.main(MAIN_ARGV)
finally:
    with open(OUT_PATH, "w") as f:
        json.dump(_records, f, indent=2)
    print(f"Wrote {len(_records)} per-round records to {OUT_PATH}")
