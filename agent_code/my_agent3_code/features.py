import copy
from collections import deque, defaultdict
from typing import Dict, List

import numpy as np
import settings as s


DIRECTIONS      = [(0, -1), (1, 0), (0, 1), (-1, 0)]  # UP, RIGHT, DOWN, LEFT
DIRECTIONS_WAIT = DIRECTIONS + [(0, 0)]


class Feature:
    """
    Builds a 21-element feature list per game step:
      [0:5]   tile tokens, one per direction [up, right, down, left, center]
              values: 'block' / 'free' / 'coin' / 'enemy' / 'dead'
      [5]     bomb slot token: 'True' / 'False' / 'dead'
      [6:11]  coin_score per direction  [up, right, down, left, center]
      [11:16] crate_score per direction [up, right, down, left, center]
      [16:21] enemy_score per direction [up, right, down, left, center]
at here i had a BFS + hand-tuned-weights rule pick "the best
    direction" directly and fed that answer in as a feature. 

    """

    def __init__(self):
        # Static wall mask
        walls = np.zeros((s.COLS, s.ROWS), int)
        walls[:1, :]  = walls[-1:, :] = -1
        walls[:, :1]  = walls[:, -1:] = -1
        for x in range(s.COLS):
            for y in range(s.ROWS):
                if (x + 1) * (y + 1) % 2 == 1:
                    walls[x, y] = -1
        self.walls = walls

        # Precompute which cells a bomb at cell i affects
        n = s.COLS * s.ROWS
        self.bomb_impact = np.zeros((n, n), dtype=int)
        for x in range(s.COLS):
            for y in range(s.ROWS):
                if walls[x, y] == -1:
                    continue
                idx = x * s.ROWS + y
                self.bomb_impact[idx, idx] = 1
                for dx, dy in DIRECTIONS:
                    for step in range(1, s.BOMB_POWER + 1):
                        nx, ny = x + dx * step, y + dy * step
                        if not (0 <= nx < s.COLS and 0 <= ny < s.ROWS):
                            break
                        if walls[nx, ny] == -1:
                            break
                        self.bomb_impact[idx, nx * s.ROWS + ny] = 1

        self.game_state: Dict = {}
        self.main_enemy: str  = None

    # ── danger map ──────────────────────────────────────────
    def _danger_map(self, bombs) -> np.ndarray:
        bmap = np.zeros((s.COLS, s.ROWS), int)
        for (bx, by), timer in bombs:
            bmap[bx, by] = s.BOMB_TIMER - timer
        return (self.bomb_impact * bmap.flatten()).max(1).reshape(s.COLS, s.ROWS)


    def _opponent_danger_zone(self, others) -> set:
        zone = {o[-1] for o in others}
        for (ox, oy) in list(zone):
            for dx, dy in DIRECTIONS:
                nx, ny = ox + dx, oy + dy
                if 0 <= nx < s.COLS and 0 <= ny < s.ROWS:
                    zone.add((nx, ny))
        return zone

    # ── safe positions reachable per first-step direction ──
    def _find_safe_positions(self, state, max_steps=5) -> List[int]:
        field, expl, bombs, others, pos = (
            state['field'], state['explosion_map'],
            state['bombs'], state['others'], state['self'][-1],
        )
        danger      = self._danger_map(bombs)
        other_zone  = self._opponent_danger_zone(others)
        bomb_dict   = {(bx, by): t for (bx, by), t in bombs}

        first_step_safes: defaultdict = defaultdict(list)
        q = deque([(pos[0], pos[1], 0, None)])

        while q:
            x, y, steps, first = q.popleft()
            in_danger = (danger[x][y] > 0 and
                         s.BOMB_TIMER - danger[x][y] < steps
                         <= s.BOMB_TIMER - danger[x][y] + s.EXPLOSION_TIMER)
            if in_danger or (expl[x][y] >= steps > 0):
                continue
            if steps == max_steps:
                if (x, y) not in first_step_safes[first]:
                    first_step_safes[first].append((x, y))
                continue
            for i, (dx, dy) in enumerate(DIRECTIONS_WAIT):
                nx, ny = x + dx, y + dy
                if not (0 <= nx < s.COLS and 0 <= ny < s.ROWS):
                    continue
                passable = (field[nx, ny] == 0 or
                            (field[nx, ny] == 1 and
                             steps > s.BOMB_TIMER - danger[nx][ny] + s.EXPLOSION_TIMER
                             and danger[nx][ny] > 0))
                if not passable:
                    continue
                if (nx, ny) in other_zone and steps <= 1:
                    continue
                if (nx, ny) in bomb_dict and bomb_dict[(nx, ny)] >= steps:
                    continue
                new_first = i if first is None else first
                q.append((nx, ny, steps + 1, new_first))

        return [len(first_step_safes.get(i, [])) for i in range(5)]

    # ── simulate bomb explosion ────────────────────────────
    def _trigger_explosion(self, pos, state):
        x, y = pos
        state['explosion_map'][x][y] = s.EXPLOSION_TIMER
        for dx, dy in DIRECTIONS:
            for step in range(1, s.BOMB_POWER + 1):
                nx, ny = x + dx * step, y + dy * step
                if not (0 <= nx < s.COLS and 0 <= ny < s.ROWS):
                    break
                if state['field'][nx][ny] == -1:
                    break
                state['explosion_map'][nx][ny] = s.EXPLOSION_TIMER
                if state['field'][nx][ny] == 1:
                    state['field'][nx][ny] = 0

    def _predict_next(self, bomb_drop=None):
        ns = copy.deepcopy(self.game_state)
        new_bombs = []
        for (bx, by), timer in ns['bombs']:
            if timer == 0:
                self._trigger_explosion((bx, by), ns)
            else:
                new_bombs.append(((bx, by), timer - 1))
        if bomb_drop:
            new_bombs.append((bomb_drop, s.BOMB_TIMER - 1))
            ns['self'] = (*ns['self'][:-1], bomb_drop)
        ns['bombs'] = new_bombs
        ns['explosion_map'] -= (ns['explosion_map'] > 0).astype(int)
        return ns

    def _is_safe_to_drop(self, pos) -> int:
        """Returns # of safe first-step directions after dropping bomb here."""
        ns = self._predict_next(pos)
        return sum(1 for v in self._find_safe_positions(ns) if v != 0)

    # ── BFS distances ──────────────────────────────────────
    def _bfs(self, state, sx, sy) -> np.ndarray:
        m, n = state['field'].shape
        # init with manhattan distance + buffer (for unreachable cells)
        dist = (np.abs(np.arange(m)[:, None] - sx) +
                np.abs(np.arange(n) - sy) +
                2 * (s.BOMB_POWER + s.EXPLOSION_TIMER))
        field, expl, bombs, others = (
            state['field'], state['explosion_map'],
            state['bombs'], state['others'],
        )
        danger     = self._danger_map(bombs)
        other_zone = self._opponent_danger_zone(others)
        bomb_pos   = {(bx, by): t for (bx, by), t in bombs}
        max_wait  = (max((t for _, t in bombs), default=0)
                     + s.EXPLOSION_TIMER + 1
                     if bombs else expl.max() + 1)

        visited = {(sx, sy)}
        q = deque([(sx, sy, 0)])

        while q:
            x, y, steps = q.popleft()
            in_danger = (danger[x][y] > 0 and
                         s.BOMB_TIMER - danger[x][y] < steps
                         <= s.BOMB_TIMER - danger[x][y] + s.EXPLOSION_TIMER)
            if in_danger or (expl[x][y] >= steps > 0):
                continue
            dist[x][y] = min(dist[x][y], steps)
            for dx, dy in DIRECTIONS_WAIT:
                nx, ny = x + dx, y + dy
                if not (0 <= nx < m and 0 <= ny < n):
                    continue
                if steps >= max_wait and (nx, ny) in visited:
                    continue
                passable = (field[nx, ny] == 0 or
                            (field[nx, ny] == 1 and
                             steps > s.BOMB_TIMER - danger[nx][ny] + s.EXPLOSION_TIMER
                             and danger[nx][ny] > 0))
                if not passable:
                    continue
                if (nx, ny) in other_zone and steps <= 1:
                    continue
                bt = bomb_pos.get((nx, ny))
                if bt is not None and bt >= steps:
                    continue
                visited.add((nx, ny))
                q.append((nx, ny, steps + 1))
        return dist

    # ── crate count per cell (# crates a bomb here would hit) ──
    def _crate_neighbors(self, field) -> np.ndarray:
        crate_vec = (field == 1).astype(int).flatten()
        counts    = self.bomb_impact.dot(crate_vec)
        return np.where(field == 0, counts.reshape(s.COLS, s.ROWS), 0)

    # ── find main enemy (closest opponent) ─────────────────
    def _find_main_enemy(self):
        if not self.game_state['others']:
            return None
        sx, sy = self.game_state['self'][-1]
        return min(
            self.game_state['others'],
            key=lambda o: abs(sx - o[-1][0]) + abs(sy - o[-1][1])
        )[0]


    def _direction_scores(self, safe: List[int]) -> np.ndarray:
        """Returns (5, 3) array of [coin_score, crate_score, enemy_score]
        per direction (UP, RIGHT, DOWN, LEFT, WAIT)."""
        scores = np.zeros((5, 3), dtype=np.float32)
        if not any(safe):
            return scores

        ns     = self._predict_next()
        field  = ns['field']
        coins  = ns['coins']
        others = ns['others']
        x, y   = ns['self'][-1]

        crate_map = self._crate_neighbors(field)
        for ci, cj in zip(*np.where(crate_map > 0)):
            if not self._is_safe_to_drop((ci, cj)):
                crate_map[ci, cj] = 0

        coin_arr   = np.array(coins) if coins else np.empty((0, 2), int)
        enemy_pos  = [o[-1] for o in others]
        directions = [(x + dx, y + dy) for dx, dy in DIRECTIONS_WAIT]

        for idx in range(5):
            if not safe[idx]:
                continue
            nx, ny = directions[idx]
            dist = self._bfs(ns, nx, ny) + 1  # +1 to avoid div/0

            scores[idx, 0] = (1.0 / dist[coin_arr[:, 0], coin_arr[:, 1]]).sum() if coin_arr.size else 0.0
            scores[idx, 1] = (crate_map / dist).sum()
            scores[idx, 2] = sum(1.0 / dist[ex, ey] for ex, ey in enemy_pos)

        return scores

    # ── main call ──────────────────────────────────────────
    def __call__(self, game_state: Dict) -> List[str]:
        if game_state is None:
            return []

        self.game_state = game_state
        self.main_enemy = self._find_main_enemy() if game_state['others'] else None

        field, expl, bombs, coins, others, (x, y) = (
            game_state['field'], game_state['explosion_map'],
            game_state['bombs'], game_state['coins'],
            game_state['others'], game_state['self'][-1],
        )

        # ── step 1: build tile tokens for 5 directions ────
        features   = []
        directions = [(x + d[0], y + d[1]) for d in DIRECTIONS_WAIT]
        field_lbl  = {1: 'block', -1: 'block', 0: 'free'}

        for i, j in directions:
            tile = field_lbl[field[i, j]]
            if tile == 'free':
                if expl[i, j] > 0:
                    tile = 'dead'
                elif any(op[-1] == (i, j) for op in others):
                    tile = 'enemy'
                elif any((b[0] == i and b[1] == j) for b, _ in bombs):
                    tile = 'block'
                elif any((c[0] == i and c[1] == j) for c in coins):
                    tile = 'coin'
            features.append(tile)

        # ── step 2: append has_bomb token ─────────────────
        features.append(str(game_state['self'][2]))   # 'True' / 'False'

        # ── step 3: mark unsafe directions as 'dead' ──────
        safe = self._find_safe_positions(game_state)
        for i in range(len(safe)):
            if safe[i] == 0 and features[i] in ['coin', 'free']:
                features[i] = 'dead'
        if not self._is_safe_to_drop((x, y)):
            features[-1] = 'dead'

        # ── step 4: this one is the objective per-direction distance scores ───

        scores = self._direction_scores(safe)
        for idx in range(5):
            features.append(float(scores[idx, 0]))  # coin_score
        for idx in range(5):
            features.append(float(scores[idx, 1]))  # crate_score
        for idx in range(5):
            features.append(float(scores[idx, 2]))  # enemy_score

        return features
