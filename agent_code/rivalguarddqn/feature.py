from collections import deque
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import settings as s

from .safe import safety_filter, contested_lethal_moves


DIRECTIONS = [(0, -1), (1, 0), (0, 1), (-1, 0)] # UP, RIGHT, DOWN, LEFT
DIRECTIONS_WAIT = DIRECTIONS + [(0, 0)] # Add WAIT

# Some Parameters
HORIZON = 8
SAFE_SPACE_CLIP = 10
MOTION_HISTORY_LENGTH = 4
MAX_DISTANCE = s.COLS + s.ROWS
FUTURE_CRATE_DISCOUNT = 0.95
COIN_BUDGET = 9
TWO_STEP_THREAT_RADIUS = s.BOMB_POWER + 2

TILE_TYPES = (
    "free", # normal empty place
    "wall",
    "crate",
    "bomb",
    "enemy",
    "coin",
    "explosion",
)
TILE_INDEX = {name: index for index, name in enumerate(TILE_TYPES)}

# The model uses 108 values from these features :
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
#   motion_history 4 * (dx, dy) actual displacements   =  8  dont think it works
#   coin_contest 5 own-vs-enemy visible-coin margins    =  5


N_FEATS = 108
SCORE_SCALE = 20.0
COIN_CONTEST_MARGIN_CLIP = 8.0


class Feature:
    """Build the state features used by the DQN."""

    def __init__(self, use_opponent_safety_filter=True):
        self.use_opponent_safety_filter = bool(use_opponent_safety_filter)

        # Build the map
        walls = np.zeros((s.COLS, s.ROWS), dtype=np.int8)
        walls[:1, :] = walls[-1:, :] = -1
        walls[:, :1] = walls[:, -1:] = -1
        for x in range(s.COLS):
            for y in range(s.ROWS):
                if x % 2 == 0 and y % 2 == 0:
                    walls[x, y] = -1
        # Blast areas depend only on fixed stone walls, so compute them once here
        # For each bomb, which cells would be influenced
        n_cells = s.COLS * s.ROWS
        self.bomb_impact = np.zeros((n_cells, n_cells), dtype=np.int8)
        self.blast_cells: Dict[Tuple[int, int], Tuple[Tuple[int, int], ...]] = {}
        for x in range(s.COLS):
            for y in range(s.ROWS):
                if walls[x, y] == -1:
                    self.blast_cells[(x, y)] = tuple()
                    continue

                cells = [(x, y)]
                # Compute 1d index
                origin_index = x * s.ROWS + y
                # Bomb impact Map
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
        # The scores of all agents at last step
        self._observed_scores: Dict[str, int] = {}
        # The total coins already obtained for now
        self._coins_collected_estimate = 0
        # The score of the best opponent
        self._max_observed_opponent_score = 0.0

    def _reset_state(self, round_number: int) -> None:
        # Forget old round
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
        # Start clean for a new round
        round_number = int(state["round"])
        if self._tracked_round != round_number:
            self._reset_state(round_number)

        own_name = str(state["self"][0])
        for agent in [state["self"], *state["others"]]:
            name = str(agent[0])
            score = int(agent[1])
            previous = self._observed_scores.get(name)
            if previous is not None and score > previous:
                # Five points is a kill, rest is coins
                score_delta = score - previous
                self._coins_collected_estimate += score_delta % s.REWARD_KILL
            self._observed_scores[name] = score
            if name != own_name:
                self._max_observed_opponent_score = max(
                    self._max_observed_opponent_score,
                    score,
                )

        # In case too large
        self._coins_collected_estimate = min(
            self._coins_collected_estimate,
            COIN_BUDGET,
        )

    def note_action(self, state: Dict, action: str) -> None:
        if not state:
            return
        round_number = int(state["round"])
        if self._tracked_round != round_number:
            self._reset_state(round_number)
        position = tuple(state["self"][-1])
        # Save this before the game runs the action
        self._pending_action = action
        self._pending_position = position
        self._recent_positions.append(position)

        if action == "BOMB" and bool(state["self"][2]):
            # Only one own bomb can exist
            self._own_bomb_position = position

    def _refresh_bomb_tracking(self, state: Dict) -> None:
        """Record the last movement and remove finished own bombs."""
        round_number = int(state["round"])
        if self._tracked_round != round_number:
            self._reset_state(round_number)
            return

        if self._pending_action is not None and self._pending_position is not None:
            # Compare wanted move with real move
            current = tuple(state["self"][-1])
            dx = current[0] - self._pending_position[0]
            dy = current[1] - self._pending_position[1]
            motion = (dx, dy) if (dx, dy) in DIRECTIONS else (0, 0)
            self._motion_history.append(motion)

            self._pending_action = None
            self._pending_position = None

        bomb_positions = {tuple(position) for position, _ in state["bombs"]}
        # Bomb gone means it already exploded
        if (
            self._own_bomb_position is not None
            and self._own_bomb_position not in bomb_positions
        ):
            self._own_bomb_position = None

    def _temporal_features(
        self,
        position: Tuple[int, int],
    ) -> Tuple[np.ndarray, np.ndarray]:
        # Return recent visits and the last four movements
        # How recently each next tile was visited
        recent_visit = np.zeros(5, dtype=np.float32)
        history = list(self._recent_positions)
        for action_index, candidate in enumerate(self._action_positions(position)):
            for age, visited in enumerate(reversed(history), start=1):
                if candidate == visited:
                    recent_visit[action_index] = (
                        MOTION_HISTORY_LENGTH - age + 1
                    ) / MOTION_HISTORY_LENGTH
                    break

        # Put newest movement at the end
        motion_history = np.zeros((MOTION_HISTORY_LENGTH, 2), dtype=np.float32)
        motions = list(self._motion_history)
        offset = MOTION_HISTORY_LENGTH - len(motions)
        for index, motion in enumerate(motions, start=offset):
            motion_history[index] = motion
        return recent_visit, motion_history.ravel()

    @staticmethod
    def _state_cache_key(state: Dict) -> Tuple:
        # Build an exact cache key for one game state
        # Round and step alone are not enough during training
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

        # Bytes also catch crate and flame changes
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

    # Time-indexed danger model
    def _hazard_schedule(self, state: Dict, horizon: int = HORIZON) -> np.ndarray:
        """Mark dangerous tiles for every future step."""
        # One danger map for each future step
        hazard = np.zeros(
            (horizon + 1, s.COLS, s.ROWS),
            dtype=bool,
        )

        # Flames already on the board
        explosion_map = state["explosion_map"]
        for x, y in np.argwhere(explosion_map > 0):
            remaining = int(np.ceil(explosion_map[x, y]))
            hazard[0, x, y] = True
            hazard[1:min(horizon, remaining) + 1, x, y] = True

        # Future bomb flames
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
        # Return the step when each crate will be destroyed
        # Horizon + 1 means not destroyed in our search
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

    # Safe movement search
    def _survival_search(
        self,
        state: Dict,
        max_steps: int = HORIZON,
        hazard: Optional[np.ndarray] = None,
    ) -> Tuple[List[int], np.ndarray]:
        # Count safe endpoints and survival steps for each first action
        # Prepare all changes during the next eight steps
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

        # Keep results separate for UP RIGHT DOWN LEFT WAIT
        safe_endpoints = [set() for _ in range(5)]
        longest = np.zeros(5, dtype=np.int16)
        queue = deque([(start_x, start_y, 0, -1)])
        visited = {(start_x, start_y, 0, -1)}

        # Search position and time together
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

                # The agent may wait on its bomb but cannot re-enter it.
                release = bomb_release.get((nx, ny))
                if release is not None and next_time < release and (nx, ny) != (x, y):
                    continue
                if hazard[next_time, nx, ny]:
                    continue
                # Only current enemy positions are known with certainty.
                if next_time == 1 and (nx, ny) in enemy_now:
                    continue

                # Remember which first action reached this path
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
        # Check several start positions in one escape search
        # Each label is one possible first action
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

        # Move all labels one step at the same time
        for next_time in range(1, max_steps + 1):
            walkable = (field != -1) & (
                (field != 1) | (clear_time <= next_time)
            )
            safe_destination = walkable & ~hazard[next_time]

            # Waiting may stay on a bomb. movement may not enter one.
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

    # Static strategic maps
    @staticmethod
    def _distance_map(field: np.ndarray, targets: Iterable[Tuple[int, int]]) -> np.ndarray:
        # Find the shortest walking distance to the nearest target
        unreachable = s.COLS * s.ROWS
        distance = np.full(field.shape, unreachable, dtype=np.int16)
        queue = deque()

        # Search backwards from every target
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
        # Count new crates hit by a bomb on each free tile
        # Matrix gives all blast cells for every bomb position
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
        # Assign each doomed crate to the bomb that reaches it first
        field = state["field"]
        claims: Dict[Tuple[int, int], Tuple[int, int, Tuple[int, int]]] = {}
        # Earlier bomb gets the crate
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
        # Estimate the future coin value of our active bomb
        if self._own_bomb_position is None:
            return 0.0
        # Late crates have less value now
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
        # Estimate the crate value after each move or WAIT
        # Check bomb value at every next position
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
        # Scale one distance to the range from zero to one
        # One means far away or unreachable
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
        # Measure whether each move gets closer to the target
        # Positive means closer, negative means farther
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
        """Compare our arrival time to the nearest enemy's arrival time."""
        values = np.zeros(5, dtype=np.float32)
        coin_positions = [tuple(coin) for coin in coins]
        if not coin_positions:
            return values

        unreachable = field.shape[0] * field.shape[1]
        has_enemies = bool(enemy_positions)
        # Each action may lead to a different best coin
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
            # If same distance, prefer the coin less useful to enemies
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

    # Lightweight one-step simulation
    def _predict_next(
        self,
        state: Dict,
        bomb_drop: Optional[Tuple[int, int]] = None,
    ) -> Dict:
        # Advance bombs and explosions by one step
        # Copy arrays because old state must stay unchanged
        next_state = dict(state)
        next_state["field"] = state["field"].copy()
        next_state["explosion_map"] = np.maximum(
            state["explosion_map"] - 1,
            0,
        )

        # Explode ready bombs and tick the others
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

        # Add a possible new bomb
        if bomb_drop is not None:
            remaining_bombs.append((bomb_drop, s.BOMB_TIMER - 1))
        next_state["bombs"] = remaining_bombs
        return next_state

    @staticmethod
    def _state_from_enemy_perspective(state: Dict, enemy: Sequence) -> Dict:
        # View the same state from one enemy's position
        # Make this enemy the self agent
        enemy_state = dict(state)
        enemy_state["self"] = enemy
        enemy_state["others"] = [
            state["self"],
            *[other for other in state["others"] if other[0] != enemy[0]],
        ]
        return enemy_state

    def _trapped_enemy_count(self, state: Dict, post_drop_state: Dict) -> int:
        # Count enemies that lose every escape after our bomb is added
        bomb_position = state["self"][-1]
        blast = set(self.blast_cells[bomb_position])
        trapped = 0

        # Only enemies inside this blast can be trapped by it
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

    # Feature groups
    @staticmethod
    def _action_positions(position: Tuple[int, int]) -> List[Tuple[int, int]]:
        # Return the four move destinations and the current tile
        # Order must match the action list
        x, y = position
        return [(x + dx, y + dy) for dx, dy in DIRECTIONS_WAIT]

    def _tile_features(self, state: Dict) -> np.ndarray:
        # Describe the four nearby tiles and the current tile
        field = state["field"]
        explosion_map = state["explosion_map"]
        bomb_positions = {position for position, _ in state["bombs"]}
        enemy_positions = {other[-1] for other in state["others"]}
        coin_positions = set(state["coins"])

        # Use one tile type for each position
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
        # Mark actions that are physically possible now.
        field = state["field"]
        # Current bombs and enemies block movement.
        blocked = {tuple(position) for position, _ in state["bombs"]}
        blocked.update(tuple(other[-1]) for other in state["others"])
        # Movement needs a free unblocked tile
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
        # Return when each nearby tile first becomes dangerous
        # Minus one means no danger in our horizon
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
        # Describe the value and safety of dropping a bomb now
        # What this bomb can hit
        position = state["self"][-1]
        available = bool(state["self"][2])
        blast = set(self.blast_cells[position])
        enemies_hit = sum(other[-1] in blast for other in state["others"])

        safe_to_drop = False
        escape_options = 0
        robust_escape_options = 0
        trapped_enemies = 0
        # Simulate the bomb only when dropping is possible
        if available and not hazard[1, position[0], position[1]]:
            post_drop_state = self._predict_next(state, bomb_drop=position)
            occupied = {tuple(pos) for pos, _ in post_drop_state["bombs"]}
            post_drop_state["bombs"] = list(post_drop_state["bombs"])
            post_drop_state["bombs"].extend(
                (tuple(pos), int(timer))
                for pos, timer in extra_bombs
                if tuple(pos) not in occupied
            )
            # Count ways to leave after dropping
            safe_after_drop, _ = self._survival_search(post_drop_state)
            escape_options = sum(count > 0 for count in safe_after_drop)
            post_drop_physical = self._valid_actions(post_drop_state)
            post_drop_enemy_destinations = (
                {
                    enemy_position
                    for enemy in post_drop_state["others"]
                    for enemy_position in self._enemy_first_positions(
                        post_drop_state, enemy
                    )
                }
                if int(safe_after_drop[4]) == 0
                else set()
            )
            contested_after_drop = contested_lethal_moves(
                post_drop_physical,
                safe_after_drop,
                self._action_positions(tuple(post_drop_state["self"][-1])),
                post_drop_enemy_destinations,
            )
            # Remove escape moves an enemy can block
            robust_escape_options = sum(
                count > 0 and not contested_after_drop[action_index]
                for action_index, count in enumerate(safe_after_drop)
            )
            safe_to_drop = escape_options > 0
            if safe_to_drop and enemies_hit:
                trapped_enemies = self._trapped_enemy_count(state, post_drop_state)
        # OASF also needs the raw action counts
        if escape_details is not None:
            escape_details.update({
                "raw_actions": int(escape_options),
                "robust_actions": int(robust_escape_options),
            })

        max_enemies = max(1, s.MAX_AGENTS - 1)
        # Crate value is based on expected hidden coins.
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
        # Assume each ready enemy may drop a bomb now
        # One simple worst case for every enemy
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
        # Return the tiles an enemy can occupy next
        # Bombs and agents block the enemy too
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
        # Check whether one nearby enemy can force a dead end
        # Start with safety from known bombs
        worst_capacity = np.asarray(base_safe_counts, dtype=np.int32).copy()
        forced_death = np.zeros(5, dtype=bool)
        own_position = tuple(state["self"][-1])
        # Far enemies cannot affect this short search
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

        # First check an enemy bomb dropped now.
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

        # Then check an enemy move followed by a bomb next turn.
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

    def __call__(self, game_state: Dict) -> Dict[str, np.ndarray]:
        """Build every feature group for one game state."""
        # No state before and after a round
        if game_state is None:
            return {}

        # Update score memory
        self._refresh_round_memory(game_state)

        # Same state gives same features
        cache_key = self._state_cache_key(game_state)
        if cache_key == self._cache_key and self._cache_value is not None:
            return self._cache_value

        # Record real movement and own bomb
        self._refresh_bomb_tracking(game_state)

        # Build danger map
        field = game_state["field"]
        position = game_state["self"][-1]
        hazard = self._hazard_schedule(game_state)

        # Search eight steps from every first action
        safe_counts, survival_steps = self._survival_search(
            game_state,
            hazard=hazard,
        )

        # Clip large counts before giving them to DQN
        safe_space = np.minimum(safe_counts, SAFE_SPACE_CLIP).astype(np.float32)
        safe_space /= SAFE_SPACE_CLIP

        # Actions possible without safety check
        physical_valid = self._valid_actions(game_state)

        # Do not count crates already claimed by another bomb
        claims = self._scheduled_crate_claims(game_state)
        crate_counts = self._crate_counts(field, excluded=claims)
        crate_targets = [tuple(cell) for cell in np.argwhere(crate_counts > 0)]

        # Estimate how many coins still exist
        remaining_coin_count = max(
            0,
            COIN_BUDGET - self._coins_collected_estimate,
        )
        hidden_coin_count = max(
            0,
            remaining_coin_count - len(game_state["coins"]),
        )

        # Chance that one crate hides a coin
        remaining_crate_count = int(np.count_nonzero(field == 1))
        hidden_coin_probability = (
            min(hidden_coin_count / remaining_crate_count, 1.0)
            if remaining_crate_count > 0 else 0.0
        )
        remaining_coin_fraction = remaining_coin_count / COIN_BUDGET

        # Shortest distance to coins, useful crates and enemies
        coin_distance = self._distance_map(field, game_state["coins"])
        crate_distance = self._distance_map(field, crate_targets)
        enemy_distance = self._distance_map(
            field,
            [other[-1] for other in game_state["others"]],
        )

        # Enemy may drop a bomb now
        possible_enemy_bombs = self._pessimistic_enemy_bombs(game_state)

        # Value and escape of our new bomb
        bomb_escape_details: Dict[str, int] = {}
        bomb = self._bomb_features(
            game_state,
            crate_counts,
            hazard,
            claims,
            hidden_coin_probability,
            escape_details=bomb_escape_details,
        )
        # Enemy bomb now or after one move
        pessimistic_safe_counts, enemy_forced_death = self._two_step_enemy_threats(
            game_state,
            physical_valid,
            safe_counts,
        )
        pessimistic_bomb_escape = float(bomb[0])
        pessimistic_robust_bomb_actions = bomb_escape_details.get(
            "robust_actions", 0
        )
        # Keep the worst bomb escape result
        for enemy_bomb in possible_enemy_bombs:
            enemy_position, enemy_timer = enemy_bomb
            enemy_bomb_escape_details: Dict[str, int] = {}
            enemy_bomb_outcome = self._bomb_features(
                game_state,
                crate_counts,
                hazard,
                claims,
                hidden_coin_probability,
                # Simulated state already one step ahead
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

        # Position after UP RIGHT DOWN LEFT WAIT
        action_positions = self._action_positions(position)

        # Safe now and in next step
        immediate_safe = np.asarray([
            not hazard[0, action_position[0], action_position[1]]
            and not hazard[1, action_position[0], action_position[1]]
            for action_position in action_positions
        ], dtype=bool)
        # Enemy collision matters when WAIT has no escape
        enemy_destinations = (
            {
                enemy_position
                for enemy in game_state["others"]
                for enemy_position in self._enemy_first_positions(
                    game_state, enemy
                )
            }
            if self.use_opponent_safety_filter and int(safe_counts[4]) == 0
            else set()
        )
        # Nearby ready enemy makes our bomb escape less reliable
        nearby_bomb_ready_enemy = self.use_opponent_safety_filter and any(
            bool(enemy[2])
            and abs(enemy[-1][0] - position[0])
            + abs(enemy[-1][1] - position[1])
            <= TWO_STEP_THREAT_RADIUS
            for enemy in game_state["others"]
        )
        # Final actions allowed for DQN
        valid = safety_filter(
            enabled=self.use_opponent_safety_filter,
            physical_valid=physical_valid,
            safe_counts=safe_counts,
            survival_steps=survival_steps,
            immediate_safe=immediate_safe,
            bomb_escape_options=float(bomb[0]),
            enemy_forced_death=enemy_forced_death,
            action_positions=action_positions,
            enemy_destinations=enemy_destinations,
            pessimistic_bomb_actions=pessimistic_robust_bomb_actions,
            nearby_bomb_ready_enemy=nearby_bomb_ready_enemy,
        )
        # Give worst-case safety to DQN too
        pessimistic_safe_space = np.minimum(
            pessimistic_safe_counts,
            SAFE_SPACE_CLIP,
        ).astype(np.float32)
        pessimistic_safe_space /= SAFE_SPACE_CLIP
        pessimistic_safety = np.concatenate([
            pessimistic_safe_space,
            np.asarray([pessimistic_bomb_escape], dtype=np.float32),
        ])

        # Directional features use physical moves
        # OASF should not erase information from DQN
        action_positions = self._action_positions(position)
        physical_movement = physical_valid[:5]

        # Progress after every move or WAIT
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
        # Can we reach the coin before an enemy
        coin_contest = self._coin_contest(
            field,
            game_state["coins"],
            [other[-1] for other in game_state["others"]],
            enemy_distance,
            action_positions,
            physical_movement,
        )
        # Short movement memory
        recent_visit, motion_history = self._temporal_features(position)

        # Keep score of dead opponents too
        own_score = float(game_state["self"][1])
        strongest_opponent = max(
            self._max_observed_opponent_score,
            max(
                (float(other[1]) for other in game_state["others"]),
                default=own_score,
            ),
        )
        score_margin = own_score - strongest_opponent
        # Put every feature group together
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
                    # Coin distance
                    self._normalized_distance(coin_distance, position),
                    # Useful crate distance
                    self._normalized_distance(crate_distance, position),
                    # Enemy distance
                    self._normalized_distance(enemy_distance, position),
                    # Enemies still alive
                    min(
                        len(game_state["others"]) / max(1, s.MAX_AGENTS - 1),
                        1.0,
                    ),
                    # Time left
                    max(
                        0.0,
                        1.0 - game_state["step"] / max(1, s.MAX_STEPS),
                    ),
                    # Lead over strongest opponent
                    np.clip(
                        score_margin / SCORE_SCALE,
                        -1.0,
                        1.0,
                    ),
                    # Coins still not collected
                    remaining_coin_fraction,
                    # Any visible coin
                    float(bool(game_state["coins"])),
                ],
                dtype=np.float32,
            ),
            "recent_visit": recent_visit,
            "motion_history": motion_history,
        }

        # Save because training asks for same state more than once
        self._cache_key = cache_key
        self._cache_value = feature
        return feature
