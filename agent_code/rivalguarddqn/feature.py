from collections import deque
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import settings as s


DIRECTIONS = [(0, -1), (1, 0), (0, 1), (-1, 0)]  # UP, RIGHT, DOWN, LEFT
DIRECTIONS_WAIT = DIRECTIONS + [(0, 0)]

HORIZON = 8
SAFE_SPACE_CLIP = 10
MOTION_HISTORY_LENGTH = 4
MAX_DISTANCE = s.COLS + s.ROWS
FUTURE_CRATE_DISCOUNT = 0.95
COIN_BUDGET = 9
TWO_STEP_THREAT_RADIUS = s.BOMB_POWER + 2

TILE_TYPES = (
    "free",
    "wall",
    "crate",
    "bomb",
    "enemy",
    "coin",
    "explosion",
)
TILE_INDEX = {name: index for index, name in enumerate(TILE_TYPES)}

# Feature V16 appends coin competition to the stable V15 103-value layout.
# Every encoded input is a direct state measurement or a one-action
# consequence; multi-step Beam Search is deliberately excluded.
#   tiles       5 categorical values -> 5 * 7 one-hot = 35
#   valid       6 action flags                         =  6
#   danger_eta  5 normalized time-to-danger values    =  5
#   safe_space  5 normalized reachable-space values   =  5
#   pessimistic_safety 6 worst-case escape values      =  6
#   coin_delta  5 per-action distance changes          =  5
#   crate_delta 5 per-action distance changes           =  5
#   crate_yield 5 per-position expected coin yields     =  5
#   enemy_delta 5 per-action distance changes           =  5
#   bomb        5 immediate bomb opportunity values     =  5
#   global      8 strategic state values               =  8
#   recent_visit 5 recency values for move/WAIT         =  5
#   motion_history 4 * (dx, dy) actual displacements   =  8
#   coin_contest 5 own-vs-enemy visible-coin margins    =  5
N_FEATS = 108
SCORE_SCALE = 20.0
COIN_CONTEST_MARGIN_CLIP = 8.0


class Feature:
    """Build a Feature V16 representation."""

    def __init__(self, use_opponent_safety_filter=True):
        self.use_opponent_safety_filter = bool(use_opponent_safety_filter)
        walls = np.zeros((s.COLS, s.ROWS), dtype=np.int8)
        walls[:1, :] = walls[-1:, :] = -1
        walls[:, :1] = walls[:, -1:] = -1
        for x in range(s.COLS):
            for y in range(s.ROWS):
                if (x + 1) * (y + 1) % 2 == 1:
                    walls[x, y] = -1
        # This environment lets blasts pass through crates; only walls stop
        # them. Precompute the resulting static blast relation once.
        n_cells = s.COLS * s.ROWS
        self.bomb_impact = np.zeros((n_cells, n_cells), dtype=np.int8)
        self.blast_cells: Dict[Tuple[int, int], Tuple[Tuple[int, int], ...]] = {}
        for x in range(s.COLS):
            for y in range(s.ROWS):
                if walls[x, y] == -1:
                    self.blast_cells[(x, y)] = tuple()
                    continue

                cells = [(x, y)]
                origin_index = x * s.ROWS + y
                self.bomb_impact[origin_index, origin_index] = 1
                for dx, dy in DIRECTIONS:
                    for distance in range(1, s.BOMB_POWER + 1):
                        nx, ny = x + dx * distance, y + dy * distance
                        if not (0 <= nx < s.COLS and 0 <= ny < s.ROWS):
                            break
                        if walls[nx, ny] == -1:
                            break
                        cells.append((nx, ny))
                        self.bomb_impact[
                            origin_index,
                            nx * s.ROWS + ny,
                        ] = 1
                self.blast_cells[(x, y)] = tuple(cells)

        self._cache_key: Optional[Tuple] = None
        self._cache_value: Optional[Dict[str, np.ndarray]] = None
        self._tracked_round: Optional[int] = None
        self._own_bomb_position: Optional[Tuple[int, int]] = None
        self._recent_positions = deque(maxlen=MOTION_HISTORY_LENGTH)
        self._motion_history = deque(maxlen=MOTION_HISTORY_LENGTH)
        self._pending_action: Optional[str] = None
        self._pending_position: Optional[Tuple[int, int]] = None
        self._observed_scores: Dict[str, int] = {}
        self._coins_collected_estimate = 0
        self._max_observed_opponent_score = 0.0

    def _reset_temporal_state(self, round_number: int) -> None:
        self._tracked_round = round_number
        self._own_bomb_position = None
        self._recent_positions.clear()
        self._motion_history.clear()
        self._pending_action = None
        self._pending_position = None
        self._observed_scores.clear()
        self._coins_collected_estimate = 0
        self._max_observed_opponent_score = 0.0

    def _refresh_round_memory(self, state: Dict) -> None:
        """Track public score changes to estimate how many coins remain.

        In this environment a player can collect at most one coin per step,
        while every kill is worth exactly five points.  Consecutive public
        score observations therefore identify a coin gain as ``delta % 5``,
        including a coin and one or more kills occurring in the same step.
        Remembering the largest observed opponent score also prevents a dead
        high-scoring opponent from disappearing from the score comparison.
        """
        round_number = int(state["round"])
        if self._tracked_round != round_number:
            self._reset_temporal_state(round_number)

        own_name = str(state["self"][0])
        for agent in [state["self"], *state["others"]]:
            name = str(agent[0])
            score = int(round(float(agent[1])))
            previous = self._observed_scores.get(name)
            if previous is not None and score > previous:
                score_delta = score - previous
                self._coins_collected_estimate += score_delta % s.REWARD_KILL
            self._observed_scores[name] = max(score, previous or 0)
            if name != own_name:
                self._max_observed_opponent_score = max(
                    self._max_observed_opponent_score,
                    float(score),
                )

        self._coins_collected_estimate = min(
            self._coins_collected_estimate,
            COIN_BUDGET,
        )

    def note_action(self, state: Dict, action: str) -> None:
        """Remember the location of our bomb without changing the game API.

        The framework exposes bomb positions and timers, but not bomb owners.
        Each agent may own only one active bomb, so remembering a successful
        BOMB command is sufficient to recover ownership in later states.
        """
        if not state:
            return
        round_number = int(state["round"])
        if self._tracked_round != round_number:
            self._reset_temporal_state(round_number)
        position = tuple(state["self"][-1])
        self._pending_action = action
        self._pending_position = position
        self._recent_positions.append(position)
        if action == "BOMB" and bool(state["self"][2]):
            self._own_bomb_position = position

    def _refresh_bomb_tracking(self, state: Dict) -> None:
        round_number = int(state["round"])
        if self._tracked_round != round_number:
            self._reset_temporal_state(round_number)
            return

        if self._pending_action is not None and self._pending_position is not None:
            current = tuple(state["self"][-1])
            dx = current[0] - self._pending_position[0]
            dy = current[1] - self._pending_position[1]
            motion = (dx, dy) if (dx, dy) in DIRECTIONS else (0, 0)
            self._motion_history.append(motion)
            self._pending_action = None
            self._pending_position = None

        bomb_positions = {tuple(position) for position, _ in state["bombs"]}
        if (
            self._own_bomb_position is not None
            and self._own_bomb_position not in bomb_positions
        ):
            self._own_bomb_position = None

    def _temporal_features(
        self,
        position: Tuple[int, int],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Encode exact short motion history and candidate-cell recency.

        The finite motion history is part of the augmented agent state.  It is
        therefore available to the policy whenever recency-dependent shaping
        is used, rather than hiding reward-relevant history from the learner.
        """
        recent_visit = np.zeros(5, dtype=np.float32)
        history = list(self._recent_positions)
        for action_index, candidate in enumerate(self._action_positions(position)):
            for age, visited in enumerate(reversed(history), start=1):
                if candidate == visited:
                    recent_visit[action_index] = (
                        MOTION_HISTORY_LENGTH - age + 1
                    ) / MOTION_HISTORY_LENGTH
                    break

        motion_history = np.zeros((MOTION_HISTORY_LENGTH, 2), dtype=np.float32)
        motions = list(self._motion_history)
        offset = MOTION_HISTORY_LENGTH - len(motions)
        for index, motion in enumerate(motions, start=offset):
            motion_history[index] = motion
        return recent_visit, motion_history.ravel()

    @staticmethod
    def _state_cache_key(state: Dict) -> Tuple:
        """Return a collision-safe signature of every feature-relevant value.

        The framework gives the state before an action and the state after the
        action the same (round, step) pair. Using only those two counters would
        therefore make the training callback reuse the old feature as the next
        feature. The arrays are tiny (17 x 17), so including their bytes is much
        cheaper than recomputing the BFS-based feature while remaining exact.
        """
        self_state = state["self"]
        self_key = (
            self_state[0],
            self_state[1],
            bool(self_state[2]),
            tuple(self_state[3]),
        )
        others_key = tuple(
            (other[0], other[1], bool(other[2]), tuple(other[3]))
            for other in state["others"]
        )
        bombs_key = tuple(
            (tuple(position), int(timer))
            for position, timer in state["bombs"]
        )
        coins_key = tuple(tuple(position) for position in state["coins"])

        return (
            int(state["round"]),
            int(state["step"]),
            self_key,
            others_key,
            bombs_key,
            coins_key,
            state["field"].tobytes(),
            state["explosion_map"].tobytes(),
        )

    # ── Time-indexed danger model ─────────────────────────
    def _hazard_schedule(self, state: Dict, horizon: int = HORIZON) -> np.ndarray:
        """Return hazard[t, x, y] for t=0..horizon."""
        hazard = np.zeros(
            (horizon + 1, s.COLS, s.ROWS),
            dtype=bool,
        )

        explosion_map = state["explosion_map"]
        for x, y in np.argwhere(explosion_map > 0):
            remaining = int(np.ceil(explosion_map[x, y]))
            hazard[0, x, y] = True
            hazard[1:min(horizon, remaining) + 1, x, y] = True

        for bomb_position, timer in state["bombs"]:
            explosion_step = int(timer) + 1
            first = max(1, explosion_step)
            last = min(horizon, explosion_step + s.EXPLOSION_TIMER - 1)
            if first > last:
                continue
            for x, y in self.blast_cells[bomb_position]:
                hazard[first:last + 1, x, y] = True

        return hazard

    def _crate_clear_times(self, state: Dict, horizon: int) -> np.ndarray:
        clear_time = np.full((s.COLS, s.ROWS), horizon + 1, dtype=np.int16)
        field = state["field"]
        for bomb_position, timer in state["bombs"]:
            explosion_step = int(timer) + 1
            if explosion_step > horizon:
                continue
            for x, y in self.blast_cells[bomb_position]:
                if field[x, y] == 1:
                    clear_time[x, y] = min(clear_time[x, y], explosion_step)
        return clear_time

    # ── Safe movement search ─────────────────────────────
    def _survival_search(
        self,
        state: Dict,
        max_steps: int = HORIZON,
        hazard: Optional[np.ndarray] = None,
    ) -> Tuple[List[int], np.ndarray]:
        """Return safe endpoints and longest survival for every first action."""
        field = state["field"]
        bombs = state["bombs"]
        start_x, start_y = state["self"][-1]

        if hazard is None:
            hazard = self._hazard_schedule(state, max_steps)
        clear_time = self._crate_clear_times(state, max_steps)
        bomb_release = {
            position: int(timer) + 1
            for position, timer in bombs
        }
        enemy_now = {
            tuple(other[-1]) for other in state.get("others", ())
        }

        safe_endpoints = [set() for _ in range(5)]
        longest = np.zeros(5, dtype=np.int16)
        queue = deque([(start_x, start_y, 0, -1)])
        visited = {(start_x, start_y, 0, -1)}

        while queue:
            x, y, time_step, first_action = queue.popleft()
            if time_step == max_steps:
                if first_action >= 0:
                    safe_endpoints[first_action].add((x, y))
                continue

            next_time = time_step + 1
            for action_index, (dx, dy) in enumerate(DIRECTIONS_WAIT):
                nx, ny = x + dx, y + dy
                if not (0 <= nx < s.COLS and 0 <= ny < s.ROWS):
                    continue

                tile = field[nx, ny]
                if tile == -1:
                    continue
                if tile == 1 and clear_time[nx, ny] > next_time:
                    continue

                # A player may wait on the bomb underneath them, but cannot
                # re-enter a bomb tile after leaving it.
                release = bomb_release.get((nx, ny))
                if release is not None and next_time < release and (nx, ny) != (x, y):
                    continue
                if hazard[next_time, nx, ny]:
                    continue
                # Current occupancy is certain for the first move.  Ignoring
                # it here made the survival BFS count routes through an enemy
                # even though the physical action mask rejected that move.
                # Do not block later times: predicting where an enemy will be
                # several turns from now would be an unjustified hard prune.
                if next_time == 1 and (nx, ny) in enemy_now:
                    continue

                first = action_index if first_action < 0 else first_action
                if next_time > longest[first]:
                    longest[first] = next_time
                search_state = (nx, ny, next_time, first)
                if search_state in visited:
                    continue
                visited.add(search_state)
                queue.append(search_state)

        return [len(endpoints) for endpoints in safe_endpoints], longest

    def _survival_from_starts(
        self,
        state: Dict,
        starts: Sequence[Tuple[int, int]],
        enabled: Sequence[bool],
        max_steps: int = HORIZON,
    ) -> np.ndarray:
        """Return whether each labelled start has an eight-step safe path.

        This is the second ply of the local adversarial check.  A single
        enemy destination creates one hazard schedule, so all five possible
        positions reached by our first action share a vectorized reachability
        search instead of repeating an identical BFS five times.  Only binary
        survival is needed here: capacity is still measured by the ordinary
        safety feature, while hard pruning asks solely whether death is forced.
        """
        field = state["field"]
        hazard = self._hazard_schedule(state, max_steps)
        clear_time = self._crate_clear_times(state, max_steps)
        bomb_release = {
            tuple(position): int(timer) + 1
            for position, timer in state["bombs"]
        }
        reachable = np.zeros(
            (len(starts), s.COLS, s.ROWS),
            dtype=bool,
        )
        for label, start in enumerate(starts):
            if enabled[label] and not hazard[0, start[0], start[1]]:
                reachable[label, start[0], start[1]] = True

        for next_time in range(1, max_steps + 1):
            walkable = (field != -1) & (
                (field != 1) | (clear_time <= next_time)
            )
            safe_destination = walkable & ~hazard[next_time]

            # WAIT may remain on the bomb underneath the agent, whereas a
            # movement action cannot enter any unreleased bomb tile.
            next_reachable = reachable & safe_destination[None, :, :]
            movable_destination = safe_destination.copy()
            for position, release in bomb_release.items():
                if next_time < release:
                    movable_destination[position] = False

            moved = np.zeros_like(reachable)
            moved[:, 1:, :] |= reachable[:, :-1, :]
            moved[:, :-1, :] |= reachable[:, 1:, :]
            moved[:, :, 1:] |= reachable[:, :, :-1]
            moved[:, :, :-1] |= reachable[:, :, 1:]
            next_reachable |= moved & movable_destination[None, :, :]
            reachable = next_reachable

        return reachable.reshape(len(starts), -1).any(axis=1)

    # ── Static strategic maps ─────────────────────────────
    @staticmethod
    def _distance_map(field: np.ndarray, targets: Iterable[Tuple[int, int]]) -> np.ndarray:
        unreachable = s.COLS * s.ROWS
        distance = np.full(field.shape, unreachable, dtype=np.int16)
        queue = deque()

        for x, y in set(targets):
            if field[x, y] == 0:
                distance[x, y] = 0
                queue.append((x, y))

        while queue:
            x, y = queue.popleft()
            next_distance = distance[x, y] + 1
            for dx, dy in DIRECTIONS:
                nx, ny = x + dx, y + dy
                if not (0 <= nx < field.shape[0] and 0 <= ny < field.shape[1]):
                    continue
                if field[nx, ny] != 0 or next_distance >= distance[nx, ny]:
                    continue
                distance[nx, ny] = next_distance
                queue.append((nx, ny))
        return distance

    def _crate_counts(
        self,
        field: np.ndarray,
        excluded: Iterable[Tuple[int, int]] = (),
    ) -> np.ndarray:
        """Count crates hit from each free bomb tile, excluding prior claims."""
        crate_vector = (field == 1).astype(np.int16).flatten()
        for position in excluded:
            crate_vector[position[0] * s.ROWS + position[1]] = 0
        counts = self.bomb_impact.dot(crate_vector)
        return np.where(
            field == 0,
            counts.reshape(s.COLS, s.ROWS),
            0,
        )

    def _scheduled_crate_claims(
        self,
        state: Dict,
    ) -> Dict[Tuple[int, int], Tuple[int, int, Tuple[int, int]]]:
        """Map every doomed crate to its first (time, order, bomb position).

        The engine processes equal-timer bombs in list order. Including that
        order avoids crediting two bombs for the same crate.
        """
        field = state["field"]
        claims: Dict[Tuple[int, int], Tuple[int, int, Tuple[int, int]]] = {}
        for order, (bomb_position, timer) in enumerate(state["bombs"]):
            position = tuple(bomb_position)
            explosion_step = int(timer) + 1
            candidate_prefix = (explosion_step, order)
            for cell in self.blast_cells[position]:
                if field[cell] != 1:
                    continue
                candidate = (*candidate_prefix, position)
                if cell not in claims or candidate[:2] < claims[cell][:2]:
                    claims[cell] = candidate
        return claims

    def _owned_future_crate_value(
        self,
        claims: Dict[Tuple[int, int], Tuple[int, int, Tuple[int, int]]],
        hidden_coin_probability: float,
    ) -> float:
        """Estimate coins revealed by the tracked bomb, discounted by timer."""
        if self._own_bomb_position is None:
            return 0.0
        discounted_crates = sum(
            FUTURE_CRATE_DISCOUNT ** explosion_step
            for explosion_step, _, owner_position in claims.values()
            if owner_position == self._own_bomb_position
        )
        expected_coins = hidden_coin_probability * discounted_crates
        return float(1.0 - np.exp(-expected_coins))

    def _prospective_crate_yield(
        self,
        field: np.ndarray,
        positions: Sequence[Tuple[int, int]],
        physical_valid: np.ndarray,
        claims: Dict[Tuple[int, int], Tuple[int, int, Tuple[int, int]]],
        hidden_coin_probability: float,
    ) -> np.ndarray:
        """Estimate coin yield at the bomb location after each first action.

        Crates already scheduled to be destroyed by an active bomb are not
        counted again.  A crate is valuable only through its current estimated
        probability of hiding a still-unrevealed coin.
        """
        values = np.zeros(5, dtype=np.float32)
        for action_index, position in enumerate(positions):
            if not physical_valid[action_index] or field[position] != 0:
                continue
            yield_count = sum(
                field[cell] == 1 and cell not in claims
                for cell in self.blast_cells[position]
            )
            expected_coins = hidden_coin_probability * yield_count
            values[action_index] = 1.0 - np.exp(-expected_coins)
        return values

    @staticmethod
    def _normalized_distance(distance: np.ndarray, position: Tuple[int, int]) -> float:
        value = int(distance[position])
        if value >= s.COLS * s.ROWS:
            return 1.0
        return min(value / MAX_DISTANCE, 1.0)

    @staticmethod
    def _distance_deltas(
        distance: np.ndarray,
        positions: Sequence[Tuple[int, int]],
        valid: np.ndarray,
    ) -> np.ndarray:
        """Return signed distance improvement for move/WAIT actions.

        Positive values move closer to the target set, negative values move
        farther away, and zero means no change or no reachable target. The
        network, rather than this feature extractor, decides how to trade off
        coins, crates, enemies, and safety.
        """
        unreachable = distance.shape[0] * distance.shape[1]
        current = int(distance[positions[4]])
        deltas = np.zeros(5, dtype=np.float32)

        for action_index, position in enumerate(positions):
            if not valid[action_index]:
                continue
            after = int(distance[position])
            if current >= unreachable and after >= unreachable:
                continue
            if current >= unreachable:
                deltas[action_index] = 1.0
            elif after >= unreachable:
                deltas[action_index] = -1.0
            else:
                deltas[action_index] = np.clip(
                    (current - after) / MAX_DISTANCE,
                    -1.0,
                    1.0,
                )
        return deltas

    def _coin_contest(
        self,
        field: np.ndarray,
        coins: Sequence[Tuple[int, int]],
        enemy_positions: Sequence[Tuple[int, int]],
        enemy_distance: np.ndarray,
        action_positions: Sequence[Tuple[int, int]],
        physical_valid: np.ndarray,
    ) -> np.ndarray:
        """Return per-action arrival margins for the same visible coin.

        Each movement/WAIT action targets its nearest reachable visible coin.
        The value is ``own arrival - nearest enemy arrival``: negative means
        that we are favoured to collect it, while positive means that an enemy
        is likely to arrive first. Equal-distance coins are resolved in favour
        of the least contested option. The already executed first action is
        included in our arrival time, so simultaneous one-step races are ties.
        """
        values = np.zeros(5, dtype=np.float32)
        coin_positions = [tuple(coin) for coin in coins]
        if not coin_positions:
            return values

        unreachable = field.shape[0] * field.shape[1]
        has_enemies = bool(enemy_positions)
        for action_index, action_position in enumerate(action_positions):
            if not physical_valid[action_index]:
                continue
            own_distance = self._distance_map(field, [action_position])
            reachable = [
                coin for coin in coin_positions
                if int(own_distance[coin]) < unreachable
            ]
            if not reachable:
                continue
            own_after = min(int(own_distance[coin]) for coin in reachable)
            nearest = [
                coin for coin in reachable
                if int(own_distance[coin]) == own_after
            ]
            target = max(
                nearest,
                key=lambda coin: int(enemy_distance[coin]),
            )
            enemy_steps = int(enemy_distance[target])
            if not has_enemies or enemy_steps >= unreachable:
                values[action_index] = -1.0
                continue
            own_steps = 1 + own_after
            values[action_index] = np.clip(
                (own_steps - enemy_steps) / COIN_CONTEST_MARGIN_CLIP,
                -1.0,
                1.0,
            )
        return values

    # ── Lightweight one-step simulation ──────────────────────
    def _predict_next(
        self,
        state: Dict,
        bomb_drop: Optional[Tuple[int, int]] = None,
    ) -> Dict:
        next_state = dict(state)
        next_state["field"] = state["field"].copy()
        next_state["explosion_map"] = np.maximum(
            state["explosion_map"] - 1,
            0,
        )

        remaining_bombs = []
        for bomb_position, timer in state["bombs"]:
            if timer <= 0:
                for x, y in self.blast_cells[bomb_position]:
                    next_state["explosion_map"][x, y] = max(
                        next_state["explosion_map"][x, y],
                        s.EXPLOSION_TIMER - 1,
                    )
                    if next_state["field"][x, y] == 1:
                        next_state["field"][x, y] = 0
            else:
                remaining_bombs.append((bomb_position, int(timer) - 1))

        if bomb_drop is not None:
            remaining_bombs.append((bomb_drop, s.BOMB_TIMER - 1))
        next_state["bombs"] = remaining_bombs
        return next_state

    @staticmethod
    def _state_from_enemy_perspective(state: Dict, enemy: Sequence) -> Dict:
        enemy_state = dict(state)
        enemy_state["self"] = enemy
        enemy_state["others"] = [
            state["self"],
            *[other for other in state["others"] if other[0] != enemy[0]],
        ]
        return enemy_state

    def _trapped_enemy_count(self, state: Dict, post_drop_state: Dict) -> int:
        bomb_position = state["self"][-1]
        blast = set(self.blast_cells[bomb_position])
        trapped = 0

        for enemy in state["others"]:
            if enemy[-1] not in blast:
                continue
            before = self._state_from_enemy_perspective(state, enemy)
            after = self._state_from_enemy_perspective(post_drop_state, enemy)
            if (
                sum(self._survival_search(before)[0]) > 0
                and sum(self._survival_search(after)[0]) == 0
            ):
                trapped += 1
        return trapped

    # ── Feature groups ───────────────────────────────────
    @staticmethod
    def _action_positions(position: Tuple[int, int]) -> List[Tuple[int, int]]:
        x, y = position
        return [(x + dx, y + dy) for dx, dy in DIRECTIONS_WAIT]

    def _tile_features(self, state: Dict) -> np.ndarray:
        field = state["field"]
        explosion_map = state["explosion_map"]
        bomb_positions = {position for position, _ in state["bombs"]}
        enemy_positions = {other[-1] for other in state["others"]}
        coin_positions = set(state["coins"])

        tiles = []
        for x, y in self._action_positions(state["self"][-1]):
            if field[x, y] == -1:
                tile = "wall"
            elif field[x, y] == 1:
                tile = "crate"
            elif explosion_map[x, y] > 0:
                tile = "explosion"
            elif (x, y) in bomb_positions:
                tile = "bomb"
            elif (x, y) in enemy_positions:
                tile = "enemy"
            elif (x, y) in coin_positions:
                tile = "coin"
            else:
                tile = "free"
            tiles.append(TILE_INDEX[tile])
        return np.asarray(tiles, dtype=np.int8)

    def _valid_actions(self, state: Dict) -> np.ndarray:
        field = state["field"]
        # Treat current agent occupancy as blocked. Although execution order is
        # random and an opponent may move away first, relying on that race was
        # a major source of invalid actions and repeated collision attempts.
        blocked = {tuple(position) for position, _ in state["bombs"]}
        blocked.update(tuple(other[-1]) for other in state["others"])
        valid = np.zeros(6, dtype=np.float32)
        for action_index, position in enumerate(
            self._action_positions(state["self"][-1])[:4]
        ):
            valid[action_index] = float(
                field[position] == 0 and position not in blocked
            )
        valid[4] = 1.0  # WAIT is always syntactically valid.
        valid[5] = float(bool(state["self"][2]))
        return valid

    def _danger_eta(
        self,
        state: Dict,
        hazard: np.ndarray,
    ) -> np.ndarray:
        eta = np.full(5, -1.0, dtype=np.float32)
        for action_index, (x, y) in enumerate(
            self._action_positions(state["self"][-1])
        ):
            dangerous_steps = np.flatnonzero(hazard[1:, x, y])
            if dangerous_steps.size:
                eta[action_index] = (int(dangerous_steps[0]) + 1) / HORIZON
        return eta

    def _bomb_features(
        self,
        state: Dict,
        crate_counts: np.ndarray,
        hazard: np.ndarray,
        claims: Dict[Tuple[int, int], Tuple[int, int, Tuple[int, int]]],
        hidden_coin_probability: float,
        extra_bombs: Sequence[Tuple[Tuple[int, int], int]] = (),
        escape_details: Optional[Dict[str, int]] = None,
    ) -> np.ndarray:
        position = state["self"][-1]
        available = bool(state["self"][2])
        blast = set(self.blast_cells[position])
        enemies_hit = sum(other[-1] in blast for other in state["others"])

        safe_to_drop = False
        escape_options = 0
        robust_escape_options = 0
        trapped_enemies = 0
        if available and not hazard[1, position[0], position[1]]:
            post_drop_state = self._predict_next(state, bomb_drop=position)
            occupied = {tuple(pos) for pos, _ in post_drop_state["bombs"]}
            post_drop_state["bombs"] = list(post_drop_state["bombs"])
            post_drop_state["bombs"].extend(
                (tuple(pos), int(timer))
                for pos, timer in extra_bombs
                if tuple(pos) not in occupied
            )
            safe_after_drop, _ = self._survival_search(post_drop_state)
            escape_options = sum(count > 0 for count in safe_after_drop)
            post_drop_physical = self._valid_actions(post_drop_state)
            contested_after_drop = self._contested_lethal_moves(
                post_drop_state,
                post_drop_physical,
                safe_after_drop,
            )
            robust_escape_options = sum(
                count > 0 and not contested_after_drop[action_index]
                for action_index, count in enumerate(safe_after_drop)
            )
            safe_to_drop = escape_options > 0
            if safe_to_drop and enemies_hit:
                trapped_enemies = self._trapped_enemy_count(state, post_drop_state)
        if escape_details is not None:
            escape_details.update({
                "raw_actions": int(escape_options),
                "robust_actions": int(robust_escape_options),
            })

        max_enemies = max(1, s.MAX_AGENTS - 1)
        # Availability is already represented by valid[BOMB], while the old
        # safe_to_drop flag was exactly equivalent to escape_options > 0.
        # Entries 1 and 4 describe expected coin revelation rather than raw
        # crate destruction.  This keeps bombing instrumental to real score.
        immediate_expected_coins = (
            hidden_coin_probability * float(crate_counts[position])
        )
        return np.asarray(
            [
                escape_options / 5.0,
                1.0 - np.exp(-immediate_expected_coins),
                min(enemies_hit / max_enemies, 1.0),
                min(trapped_enemies / max_enemies, 1.0),
                self._owned_future_crate_value(
                    claims,
                    hidden_coin_probability,
                ),
            ],
            dtype=np.float32,
        )

    def _pessimistic_enemy_bombs(
        self,
        state: Dict,
    ) -> List[Tuple[Tuple[int, int], int]]:
        """Return one concurrent immediate-drop hypothesis per opponent."""
        occupied = {tuple(position) for position, _ in state["bombs"]}
        return [
            (tuple(other[-1]), s.BOMB_TIMER)
            for other in state["others"]
            if bool(other[2]) and tuple(other[-1]) not in occupied
        ]

    def _enemy_first_positions(
        self,
        state: Dict,
        enemy: Sequence,
    ) -> List[Tuple[int, int]]:
        """Enumerate physical move/WAIT destinations for one enemy."""
        field = state["field"]
        origin = tuple(enemy[-1])
        blocked = {tuple(position) for position, _ in state["bombs"]}
        blocked.add(tuple(state["self"][-1]))
        blocked.update(
            tuple(other[-1])
            for other in state["others"]
            if other[0] != enemy[0]
        )
        positions = []
        for dx, dy in DIRECTIONS_WAIT:
            nx, ny = origin[0] + dx, origin[1] + dy
            position = (nx, ny)
            if not (0 <= nx < s.COLS and 0 <= ny < s.ROWS):
                continue
            if field[position] != 0:
                continue
            if position != origin and position in blocked:
                continue
            positions.append(position)
        return positions

    def _two_step_enemy_threats(
        self,
        state: Dict,
        physical_valid: np.ndarray,
        base_safe_counts: Sequence[int],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Find nearby single-enemy bomb plans that force a dead end.

        For every current movement/WAIT action, enumerate one bomb-ready
        enemy's physically possible first move followed by BOMB.  The plan is
        considered forcing only if *all* of our physical second responses have
        zero eight-step escape capacity.  Immediate concurrent enemy bombs are
        checked as well.  Enemies are considered one at a time to avoid the
        unrealistic coordinated worst case that previously caused jitter.
        """
        worst_capacity = np.asarray(base_safe_counts, dtype=np.int32).copy()
        forced_death = np.zeros(5, dtype=bool)
        own_position = tuple(state["self"][-1])
        nearby_enemies = [
            enemy
            for enemy in state["others"]
            if bool(enemy[2])
            and abs(enemy[-1][0] - own_position[0])
            + abs(enemy[-1][1] - own_position[1])
            <= TWO_STEP_THREAT_RADIUS
        ]
        if not nearby_enemies:
            return worst_capacity, forced_death

        occupied_bombs = {tuple(position) for position, _ in state["bombs"]}

        # Enemy BOMB on the current turn.  Timer BOMB_TIMER is correct before
        # the concurrent action; the environment decrements it after actions.
        for enemy in nearby_enemies:
            enemy_position = tuple(enemy[-1])
            if enemy_position in occupied_bombs:
                continue
            scenario = dict(state)
            scenario["bombs"] = [
                *state["bombs"],
                (enemy_position, s.BOMB_TIMER),
            ]
            counts = np.asarray(
                self._survival_search(scenario)[0], dtype=np.int32
            )
            worst_capacity = np.minimum(worst_capacity, counts)
            forced_death |= (
                np.asarray(physical_valid[:5], dtype=bool)
                & (counts == 0)
            )

        # Enemy MOVE/WAIT now and BOMB on the following turn.  For one enemy
        # bomb destination, a labelled multi-start BFS evaluates every one of
        # our current first actions together.  A plan is forcing only when no
        # subsequent movement/WAIT sequence reaches a safe horizon endpoint.
        next_base = self._predict_next(state)
        next_bombs = {tuple(position) for position, _ in next_base["bombs"]}
        self_positions = self._action_positions(own_position)
        base_enabled = np.asarray(physical_valid[:5], dtype=bool)
        base_enabled &= np.asarray(
            [
                next_base["field"][position] == 0
                and next_base["explosion_map"][position] <= 0
                for position in self_positions
            ],
            dtype=bool,
        )
        for enemy in nearby_enemies:
            for enemy_position in self._enemy_first_positions(state, enemy):
                if enemy_position in next_bombs:
                    continue
                if next_base["explosion_map"][enemy_position] > 0:
                    continue
                threatened = np.asarray(
                    [
                        position in self.blast_cells[enemy_position]
                        and position != enemy_position
                        for position in self_positions
                    ],
                    dtype=bool,
                )
                enabled = base_enabled & threatened
                if not enabled.any():
                    continue
                scenario = dict(next_base)
                scenario["bombs"] = [
                    *next_base["bombs"],
                    (enemy_position, s.BOMB_TIMER),
                ]
                survives = self._survival_from_starts(
                    scenario,
                    self_positions,
                    enabled,
                )
                forced = enabled & ~survives
                worst_capacity[forced] = 0
                forced_death |= forced

        return worst_capacity, forced_death

    def _contested_lethal_moves(
        self,
        state: Dict,
        physical_valid: np.ndarray,
        safe_counts: Sequence[int],
    ) -> np.ndarray:
        """Flag escape moves whose execution failure removes every escape.

        Agents choose simultaneously but move in a random execution order. If
        an opponent can enter the same destination first, our movement may be
        converted into an implicit WAIT.  Such a contested move is not a
        reliable escape when the WAIT branch has no full-horizon safe path.
        Unlike checking only the next hazard frame, this detects collisions
        that leave us alive for a turn or two but too late to clear a blast.
        """
        contested_lethal = np.zeros(5, dtype=bool)
        origin = tuple(state["self"][-1])
        if int(safe_counts[4]) > 0:
            return contested_lethal

        enemy_destinations = {
            position
            for enemy in state["others"]
            for position in self._enemy_first_positions(state, enemy)
        }
        for action_index, destination in enumerate(
            self._action_positions(origin)[:4]
        ):
            contested_lethal[action_index] = bool(
                physical_valid[action_index]
                and destination in enemy_destinations
            )
        return contested_lethal

    @staticmethod
    def _safety_mask(
        physical_valid: np.ndarray,
        safe_counts: Sequence[int],
        bomb_escape_options: float,
    ) -> np.ndarray:
        mask = np.asarray(physical_valid, dtype=np.float32).copy()
        mask[:5] *= np.asarray(safe_counts, dtype=np.float32) > 0
        mask[5] *= float(bomb_escape_options > 0.0)
        return mask

    @staticmethod
    def _select_action_mask(
        physical_valid: np.ndarray,
        base_valid: np.ndarray,
        robust_valid: Optional[np.ndarray] = None,
        survival_steps: Optional[Sequence[int]] = None,
        immediate_safe: Optional[Sequence[bool]] = None,
    ) -> np.ndarray:
        """Prefer robust actions and use a non-bomb longest-life fallback."""
        if robust_valid is not None and robust_valid.any():
            return np.asarray(robust_valid, dtype=np.float32)

        if base_valid.any():
            # When all known-safe actions are opponent-contested, retain the
            # movement choices because no guaranteed alternative exists.  A
            # BOMB rejected by the robustness check must not be reintroduced.
            fallback = np.asarray(base_valid, dtype=np.float32).copy()
            if robust_valid is not None and not robust_valid[5]:
                fallback[5] = 0.0
            if fallback.any():
                return fallback
            return np.asarray(base_valid, dtype=np.float32)

        # No action reaches the full deterministic horizon.  Never add a new
        # bomb in this state; retain only the physical move/WAIT actions with
        # the longest known-safe prefix.
        fallback = np.zeros(6, dtype=np.float32)
        physical_movement = np.asarray(physical_valid[:5], dtype=bool)
        if immediate_safe is not None:
            hazard_safe = np.asarray(immediate_safe, dtype=bool)
            if np.any(physical_movement & hazard_safe):
                physical_movement &= hazard_safe
        if survival_steps is None:
            fallback[:5] = physical_movement
        else:
            depths = np.asarray(survival_steps, dtype=np.int16)
            best_depth = int(np.max(depths[physical_movement]))
            fallback[:5] = physical_movement & (depths == best_depth)
        if not fallback.any():
            fallback[4] = 1.0
        return fallback

    # ── Public API ────────────────────────────────────────
    def __call__(self, game_state: Dict) -> Dict[str, np.ndarray]:
        if game_state is None:
            return {}

        self._refresh_round_memory(game_state)
        cache_key = self._state_cache_key(game_state)
        if cache_key == self._cache_key and self._cache_value is not None:
            return self._cache_value

        self._refresh_bomb_tracking(game_state)

        field = game_state["field"]
        position = game_state["self"][-1]
        hazard = self._hazard_schedule(game_state)
        safe_counts, survival_steps = self._survival_search(
            game_state,
            hazard=hazard,
        )
        safe_space = np.minimum(safe_counts, SAFE_SPACE_CLIP).astype(np.float32)
        safe_space /= SAFE_SPACE_CLIP

        physical_valid = self._valid_actions(game_state)
        claims = self._scheduled_crate_claims(game_state)
        crate_counts = self._crate_counts(field, excluded=claims)
        crate_targets = [tuple(cell) for cell in np.argwhere(crate_counts > 0)]
        remaining_coin_count = max(
            0,
            COIN_BUDGET - self._coins_collected_estimate,
        )
        hidden_coin_count = max(
            0,
            remaining_coin_count - len(game_state["coins"]),
        )
        remaining_crate_count = int(np.count_nonzero(field == 1))
        hidden_coin_probability = (
            min(hidden_coin_count / remaining_crate_count, 1.0)
            if remaining_crate_count > 0 else 0.0
        )
        remaining_coin_fraction = remaining_coin_count / COIN_BUDGET

        coin_distance = self._distance_map(field, game_state["coins"])
        crate_distance = self._distance_map(field, crate_targets)
        enemy_distance = self._distance_map(
            field,
            [other[-1] for other in game_state["others"]],
        )

        possible_enemy_bombs = self._pessimistic_enemy_bombs(game_state)

        bomb_escape_details: Dict[str, int] = {}
        bomb = self._bomb_features(
            game_state,
            crate_counts,
            hazard,
            claims,
            hidden_coin_probability,
            escape_details=bomb_escape_details,
        )
        # Opponent hypotheses remain soft critic/DQN inputs unless a local
        # two-step plan proves one first action forced-dead while a robust
        # alternative exists.  The opponent-aware safety filter is not a
        # blanket worst-case mask over every possible enemy action.
        pessimistic_safe_counts, enemy_forced_death = self._two_step_enemy_threats(
            game_state,
            physical_valid,
            safe_counts,
        )
        pessimistic_bomb_escape = float(bomb[0])
        pessimistic_robust_bomb_actions = bomb_escape_details.get(
            "robust_actions", 0
        )
        for enemy_bomb in possible_enemy_bombs:
            enemy_position, enemy_timer = enemy_bomb
            enemy_bomb_escape_details: Dict[str, int] = {}
            enemy_bomb_outcome = self._bomb_features(
                game_state,
                crate_counts,
                hazard,
                claims,
                hidden_coin_probability,
                # _bomb_features first advances our own new bomb one step.
                extra_bombs=[(enemy_position, enemy_timer - 1)],
                escape_details=enemy_bomb_escape_details,
            )
            pessimistic_bomb_escape = min(
                pessimistic_bomb_escape,
                float(enemy_bomb_outcome[0]),
            )
            pessimistic_robust_bomb_actions = min(
                pessimistic_robust_bomb_actions,
                enemy_bomb_escape_details.get("robust_actions", 0),
            )

        base_valid = self._safety_mask(
            physical_valid,
            safe_counts,
            bomb[0],
        )
        action_positions = self._action_positions(position)
        immediate_safe = np.asarray([
            not hazard[0, action_position[0], action_position[1]]
            and not hazard[1, action_position[0], action_position[1]]
            for action_position in action_positions
        ], dtype=bool)
        if self.use_opponent_safety_filter:
            robust_valid = base_valid.copy()
            robust_valid[:5] *= ~enemy_forced_death
            contested_lethal = self._contested_lethal_moves(
                game_state,
                physical_valid,
                safe_counts,
            )
            robust_valid[:5] *= ~contested_lethal

            nearby_bomb_ready_enemy = any(
                bool(enemy[2])
                and abs(enemy[-1][0] - position[0])
                + abs(enemy[-1][1] - position[1])
                <= TWO_STEP_THREAT_RADIUS
                for enemy in game_state["others"]
            )
            robust_bomb = pessimistic_robust_bomb_actions > 0
            if nearby_bomb_ready_enemy:
                robust_bomb &= pessimistic_robust_bomb_actions >= 2
            robust_valid[5] *= float(robust_bomb)

            valid = self._select_action_mask(
                physical_valid,
                base_valid,
                robust_valid,
                survival_steps,
                immediate_safe,
            )
        else:
            # Training deliberately exposes the DQN to the full deterministic
            # known-hazard action set. Deployment adds opponent-aware pruning.
            valid = (
                base_valid
                if base_valid.any()
                else self._select_action_mask(
                    physical_valid,
                    base_valid,
                    survival_steps=survival_steps,
                    immediate_safe=immediate_safe,
                )
            )

        # Even if every action is threatened and the hard mask must recover,
        # the critic still receives the worst response capacity as a feature.
        pessimistic_safe_space = np.minimum(
            pessimistic_safe_counts,
            SAFE_SPACE_CLIP,
        ).astype(np.float32)
        pessimistic_safe_space /= SAFE_SPACE_CLIP
        pessimistic_safety = np.concatenate([
            pessimistic_safe_space,
            np.asarray([pessimistic_bomb_escape], dtype=np.float32),
        ])

        action_positions = self._action_positions(position)
        # Directional deltas describe physical consequences. Do not erase
        # them merely because the separate safety shield rejects an action.
        physical_movement = physical_valid[:5]
        coin_delta = self._distance_deltas(
            coin_distance, action_positions, physical_movement
        )
        crate_delta = self._distance_deltas(
            crate_distance, action_positions, physical_movement
        )
        crate_yield = self._prospective_crate_yield(
            field,
            action_positions,
            physical_movement,
            claims,
            hidden_coin_probability,
        )
        enemy_delta = self._distance_deltas(
            enemy_distance, action_positions, physical_movement
        )
        coin_contest = self._coin_contest(
            field,
            game_state["coins"],
            [other[-1] for other in game_state["others"]],
            enemy_distance,
            action_positions,
            physical_movement,
        )
        recent_visit, motion_history = self._temporal_features(position)

        own_score = float(game_state["self"][1])
        strongest_opponent = max(
            self._max_observed_opponent_score,
            max(
                (float(other[1]) for other in game_state["others"]),
                default=own_score,
            ),
        )
        score_margin = own_score - strongest_opponent
        feature = {
            "tiles": self._tile_features(game_state),
            "valid": valid,
            "physical_valid": physical_valid,
            "danger_eta": self._danger_eta(game_state, hazard),
            "safe_space": safe_space,
            "pessimistic_safety": pessimistic_safety,
            "coin_delta": coin_delta,
            "crate_delta": crate_delta,
            "crate_yield": crate_yield,
            "enemy_delta": enemy_delta,
            "coin_contest": coin_contest,
            "bomb": bomb,
            "global": np.asarray(
                [
                    self._normalized_distance(coin_distance, position),
                    self._normalized_distance(crate_distance, position),
                    self._normalized_distance(enemy_distance, position),
                    min(
                        len(game_state["others"]) / max(1, s.MAX_AGENTS - 1),
                        1.0,
                    ),
                    max(
                        0.0,
                        1.0 - game_state["step"] / max(1, s.MAX_STEPS),
                    ),
                    np.clip(
                        score_margin / SCORE_SCALE,
                        -1.0,
                        1.0,
                    ),
                    remaining_coin_fraction,
                    float(bool(game_state["coins"])),
                ],
                dtype=np.float32,
            ),
            "recent_visit": recent_visit,
            "motion_history": motion_history,
        }

        self._cache_key = cache_key
        self._cache_value = feature
        return feature
