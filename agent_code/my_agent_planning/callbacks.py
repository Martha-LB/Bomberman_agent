"""
my_agent_planning: short-horizon model-based lookahead wrapped around
my_agent3's trained Dueling DQN.

Design (full writeup in README.md):
  - "Model-based" part: at each decision point, simulate ~2 ticks ahead using
    the TRUE, known game rules (bomb countdown, explosion propagation, crate
    destruction, movement legality) instead of learning a world model — the
    rules are already known exactly, so learning them would just be
    reinventing something we already have for free. The bomb/explosion-advance
    primitives (_predict_next / _trigger_explosion / bomb_impact) are reused
    as-is from agent_code/my_agent3/features.py (the same ones my_agent3
    already uses for its own 1-step "is this safe" checks, and the ones whose
    kill-detection bug we fixed earlier this session); only the self-movement
    / bomb-placement step is new, since _predict_next doesn't cover it.
  - This search's ONLY job is to VETO actions that provably lead to death
    within the lookahead horizon — a fact fully decidable from the known
    rules, no learning required. Among actions that survive the veto, the
    choice is handed entirely to my_agent3's trained DQN (its own greedy
    Q-value) — this is the machine-learned component, and it's exactly the
    criterion my_agent3 already uses to act on its own.
  - Net effect: my_agent3's policy, with a rules-based safety net in front of
    it that catches the cases where the learned Q-function would otherwise
    walk into an avoidable death (e.g. states outside its training
    distribution — see my_agent2's README for a real example of that failure
    mode in a sibling agent).

This agent does not train its own weights; it always reads my_agent3's
model.pt read-only, so retraining my_agent3 automatically improves this
agent's leaf evaluation too. Don't pass --train to this agent.
"""
import io
import pickle
import random

import torch

import settings as s
from agent_code.my_agent3.features import Feature
from agent_code.my_agent3.model import DQN
from agent_code.my_agent3.callbacks import encode_feature, N_FEATS, N_ACTIONS, MODEL_PATH as MY_AGENT3_MODEL_PATH

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
DELTA   = {'UP': (0, -1), 'RIGHT': (1, 0), 'DOWN': (0, 1), 'LEFT': (-1, 0)}

SEARCH_DEPTH  = 2        # ticks of lookahead the safety veto checks
DEATH_PENALTY = -1000.0  # far below any realistic trained Q-value


def setup(self):
    self.device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    self.feature = Feature()  # reused simulation + feature-encoding primitives

    if MY_AGENT3_MODEL_PATH.exists():
        # Same CUDA->CPU unpickling workaround as my_agent3/callbacks.py (see
        # that agent's README for why this should eventually become a plain
        # torch.load(..., map_location='cpu') instead).
        original = torch.storage._load_from_bytes
        torch.storage._load_from_bytes = lambda b: torch.load(
            io.BytesIO(b), map_location='cpu', weights_only=False
        )
        try:
            with open(MY_AGENT3_MODEL_PATH, "rb") as f:
                self.policy_net = pickle.load(f).to(self.device)
            self.logger.info(f"Loaded my_agent3's trained model from {MY_AGENT3_MODEL_PATH}")
        finally:
            torch.storage._load_from_bytes = original
    else:
        self.logger.warning(
            "my_agent3 has no trained model.pt yet - leaf evaluation will be "
            "on an untrained network. Train my_agent3 first; this agent will "
            "automatically pick up the improved checkpoint."
        )
        self.policy_net = DQN(N_FEATS, N_ACTIONS).to(self.device)

    self.policy_net.eval()


def act(self, game_state: dict) -> str:
    if game_state is None:
        return random.choice(ACTIONS)
    return _choose_action(self.feature, self.policy_net, self.device, game_state, depth=SEARCH_DEPTH)


def _q_values(feat: Feature, model, device, state: dict):
    """my_agent3's own Q(s, ·) for the 6 actions, from a single forward pass on
    the CURRENT (real, not simulated) state - exactly what my_agent3.act()
    itself uses to decide. One Feature() call + one forward pass total,
    regardless of search depth."""
    feature = feat(state)
    encoded = encode_feature(feature)
    x = torch.tensor(encoded, dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        q = model(x)
    return q.squeeze(0).tolist()


# ── known-rules simulation (no learning involved) ───────────────────
def _step_died(feat: Feature, state: dict, new_pos) -> bool:
    """True if `new_pos` sits inside an already-active or newly-triggered
    explosion during the tick that just happened. Checked against the
    PRE-step state (before bombs/explosions advance), which is the state
    that actually determines whether standing there kills you this tick."""
    if state['explosion_map'][new_pos] > 0:
        return True
    nx, ny = new_pos
    target_idx = nx * s.ROWS + ny
    for (bx, by), timer in state['bombs']:
        if timer == 0 and feat.bomb_impact[bx * s.ROWS + by, target_idx]:
            return True
    return False


def simulate_action(feat: Feature, state: dict, action: str):
    """
    One full game tick forward for a hypothetical action taken by our agent.
    Bomb countdown / explosion propagation / crate destruction reuse
    Feature._predict_next() (which internally calls _trigger_explosion()) -
    the same, already-tested primitive my_agent3 uses for its own lookahead.
    Movement/bomb-placement legality is new here, since _predict_next only
    covers "drop a bomb in place", not directional movement.

    Returns (next_state, died). next_state is None for an illegal action
    (BOMB with no bomb left).

    Simplification (documented, not hidden): opponents are treated as static
    obstacles for the whole horizon - we don't simulate their movement. This
    matches what my_agent2/my_agent3's own 1-step lookahead already assumes
    elsewhere, and keeps the search cheap enough for the 0.5s per-step budget.
    """
    name, score, has_bomb, (x, y) = state['self']
    field     = state['field']
    other_pos = {o[-1] for o in state['others']}
    bomb_pos  = {pos for pos, _ in state['bombs']}

    bomb_drop = None
    if action == 'BOMB':
        if not has_bomb:
            return None, True  # illegal: nothing to drop
        new_pos, bomb_drop = (x, y), (x, y)
    elif action == 'WAIT':
        new_pos = (x, y)
    else:
        dx, dy = DELTA[action]
        cand = (x + dx, y + dy)
        passable = (0 <= cand[0] < s.COLS and 0 <= cand[1] < s.ROWS
                    and field[cand] == 0
                    and cand not in other_pos
                    and cand not in bomb_pos)
        new_pos = cand if passable else (x, y)  # bump into an obstacle -> stay, like the real engine

    died = _step_died(feat, state, new_pos)

    feat.game_state = state
    ns = feat._predict_next(bomb_drop=bomb_drop)
    ns['self'] = (name, score, (False if bomb_drop else has_bomb), new_pos)
    return ns, died


def _has_safe_followup(feat: Feature, state: dict) -> bool:
    """Is there at least one second-ply action that doesn't kill us? If not,
    this branch is a provable trap even though it survived the first tick."""
    for a2 in ACTIONS:
        s2, died2 = simulate_action(feat, state, a2)
        if s2 is not None and not died2:
            return True
    return False


# ── decision: known-rules veto + machine-learned ranking ────────────
def _choose_action(feat: Feature, model, device, state: dict, depth: int = 2) -> str:
    """
    Rank candidate actions by my_agent3's own Q(s, a) - the SAME criterion
    my_agent3.act() uses on its own - then veto any action that the known
    game rules prove leads to death within `depth` ticks. Pick the
    highest-ranked action that survives the veto.

    Earlier version of this function re-simulated each action and scored the
    RESULTING state with max_a' Q(s', a') instead of reading Q(s, a) directly.
    That's a subtly different quantity (a 1-ply bootstrap of the *next*
    state's value, not the network's own estimate for taking a1 from *here*),
    and for an imperfectly-trained network the two need not agree - in
    testing it caused the agent to get stuck walking into a wall forever,
    since "stay in place" and "attempt a blocked move" produce identical
    resulting states and therefore identical (but not necessarily *best*)
    child scores. Reading Q(s, ·) once, directly, matches my_agent3's own
    behavior exactly whenever nothing needs to be vetoed.
    """
    q_values = _q_values(feat, model, device, state)  # one Feature() + one forward pass, total

    best_action, best_value = None, float('-inf')
    for i, a1 in enumerate(ACTIONS):
        s1, died1 = simulate_action(feat, state, a1)
        if s1 is None:
            continue
        if died1:
            value = DEATH_PENALTY
        elif depth >= 2 and not _has_safe_followup(feat, s1):
            value = DEATH_PENALTY  # provably trapped within the lookahead horizon
        else:
            value = q_values[i]
        if value > best_value:
            best_value, best_action = value, a1
    return best_action or 'WAIT'
