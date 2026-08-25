import copy
from collections import deque, defaultdict
from typing import Dict, List

import numpy as np
import settings as s

# direction order: UP RIGHT DOWN LEFT WAIT(center)
DIRECTIONS      = [(0, -1), (1, 0), (0, 1), (-1, 0)]
DIRECTIONS_WAIT = DIRECTIONS + [(0, 0)]


class Feature:
    """
    Produces a 57-dim feature vector per game step.

    Layout per direction slot (5 slots: UP RIGHT DOWN LEFT CENTER):
      [0..4]  one-hot tile type: crate / wall / enemy / bomb / coin
      [5]     crate score  (BFS-weighted sum of reachable crates)
      [6]     coin score   (BFS-weighted sum of reachable coins)
      [7..9]  enemy score  (BFS distance to each of 3 enemies)
      [10]    safe count   (# safe positions reachable via this first step)
    → 5 × 11 = 55 dims

    Global (2 dims):
      [55]  bomb_valid  — can drop bomb here and still escape?
      [56]  kill        — dropping bomb here would trap an enemy?

    Total: 57 dims
    """

    CATEGORIES = ['crate', 'wall', 'enemy', 'bomb', 'coin']
    CAT_IDX    = {c: i for i, c in enumerate(CATEGORIES)}

    def __init__(self):
        # static wall mask (doesn't change between games)
        walls = np.zeros((s.COLS, s.ROWS), int)
        walls[:1, :] = walls[-1:, :] = -1
        walls[:, :1] = walls[:, -1:] = -1
        for x in range(s.COLS):
            for y in range(s.ROWS):
                if (x + 1) * (y + 1) % 2 == 1:
                    walls[x, y] = -1
        self.walls = walls

        # precompute which cells a bomb at cell i affects
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

    # ── danger map ──────────────────────────────────────────────────
    def _danger_map(self, bombs) -> np.ndarray:
        bmap = np.zeros((s.COLS, s.ROWS), int)
        for (bx, by), timer in bombs:
            bmap[bx, by] = s.BOMB_TIMER - timer
        return (self.bomb_impact * bmap.flatten()).max(1).reshape(s.COLS, s.ROWS)

    # ── BFS: shortest safe paths from (x,y) ─────────────────────────
    def _bfs(self, state, x, y) -> np.ndarray:
        field, expl, bombs, others = (
            state['field'], state['explosion_map'],
            state['bombs'], state['others'],
        )
        danger        = self._danger_map(bombs)
        other_pos     = {o[-1] for o in others}
        bomb_pos      = {(bx, by): t for (bx, by), t in bombs}
        max_wait      = (max((t for _, t in bombs), default=0)
                         + s.EXPLOSION_TIMER + 1)

        dist = np.full((s.COLS, s.ROWS), np.inf)
        visited = {(x, y)}
        q = deque([(x, y, 0)])

        while q:
            cx, cy, steps = q.popleft()
            # skip if passing through active explosion
            in_danger = (danger[cx][cy] > 0 and
                         s.BOMB_TIMER - danger[cx][cy] < steps
                         <= s.BOMB_TIMER - danger[cx][cy] + s.EXPLOSION_TIMER)
            if in_danger or (expl[cx][cy] >= steps > 0):
                continue
            dist[cx][cy] = min(dist[cx][cy], steps)
            for dx, dy in DIRECTIONS_WAIT:
                nx, ny = cx + dx, cy + dy
                if not (0 <= nx < s.COLS and 0 <= ny < s.ROWS):
                    continue
                if steps >= max_wait and (nx, ny) in visited:
                    continue
                passable = (field[nx, ny] == 0 or
                            (field[nx, ny] == 1 and
                             steps > s.BOMB_TIMER - danger[nx][ny] + s.EXPLOSION_TIMER
                             and danger[nx][ny] > 0))
                if not passable:
                    continue
                if (nx, ny) in other_pos and steps == 0:
                    continue
                bt = bomb_pos.get((nx, ny))
                if bt is not None and bt >= steps:
                    continue
                visited.add((nx, ny))
                q.append((nx, ny, steps + 1))
        return dist

    # ── safe position count per first-step direction ─────────────────
    def _safe_counts(self, state, max_steps=5) -> List[int]:
        field, expl, bombs, others, pos = (
            state['field'], state['explosion_map'],
            state['bombs'], state['others'], state['self'][-1],
        )
        danger    = self._danger_map(bombs)
        other_pos = {o[-1] for o in others}
        bomb_pos  = {(bx, by): t for (bx, by), t in bombs}

        first_step_safes: defaultdict = defaultdict(list)
        q = deque([(pos[0], pos[1], 0, None)])

        while q:
            cx, cy, steps, first = q.popleft()
            in_danger = (danger[cx][cy] > 0 and
                         s.BOMB_TIMER - danger[cx][cy] < steps
                         <= s.BOMB_TIMER - danger[cx][cy] + s.EXPLOSION_TIMER)
            if in_danger or (expl[cx][cy] >= steps > 0):
                continue
            if steps == max_steps:
                if (cx, cy) not in first_step_safes[first]:
                    first_step_safes[first].append((cx, cy))
                continue
            for i, (dx, dy) in enumerate(DIRECTIONS_WAIT):
                nx, ny = cx + dx, cy + dy
                if not (0 <= nx < s.COLS and 0 <= ny < s.ROWS):
                    continue
                passable = (field[nx, ny] == 0 or
                            (field[nx, ny] == 1 and
                             steps > s.BOMB_TIMER - danger[nx][ny] + s.EXPLOSION_TIMER
                             and danger[nx][ny] > 0))
                if not passable:
                    continue
                if (nx, ny) in other_pos and steps == 0:
                    continue
                bt = bomb_pos.get((nx, ny))
                if bt is not None and bt >= steps:
                    continue
                new_first = i if first is None else first
                q.append((nx, ny, steps + 1, new_first))

        return [len(first_step_safes.get(i, [])) for i in range(5)]

    # ── simulate bomb explosion ──────────────────────────────────────
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

    def _safe_to_drop(self, pos) -> bool:
        ns = self._predict_next(pos)
        return any(self._safe_counts(ns))

    # ── crate neighbour counts (how many crates a bomb at each cell hits) ─
    def _crate_scores(self, field) -> np.ndarray:
        crate_vec    = (field == 1).astype(int).flatten()
        counts       = self.bomb_impact.dot(crate_vec)
        counts_mat   = counts.reshape(s.COLS, s.ROWS)
        return np.where(field == 0, counts_mat, 0)

    # ── per-direction score matrix ───────────────────────────────────
    def _score_matrix(self, safe: List[int]) -> np.ndarray:
        """Returns (5,5) matrix: [crate_score, coin_score, enemy0, enemy1, enemy2]"""
        scores = np.zeros((5, 5))
        if not any(safe):
            return scores

        ns     = self._predict_next()
        field  = ns['field']
        coins  = ns['coins']
        others = ns['others']
        x, y   = ns['self'][-1]

        crate_map  = self._crate_scores(field)
        # mask positions where dropping bomb is unsafe
        for ci, cj in zip(*np.where(crate_map > 0)):
            if not self._safe_to_drop((ci, cj)):
                crate_map[ci, cj] = 0

        coin_arr   = np.array(coins) if coins else np.empty((0, 2), int)
        enemy_pos  = [o[-1] for o in others]
        directions = [(x + dx, y + dy) for dx, dy in DIRECTIONS_WAIT]

        for idx in range(5):
            if not safe[idx]:
                continue
            nx, ny = directions[idx]
            dist = self._bfs(ns, nx, ny) + 1        # +1 to avoid div/0

            # crate score
            scores[idx, 0] = (crate_map / dist).sum()
            # coin score
            if coin_arr.size:
                scores[idx, 1] = (1.0 / dist[coin_arr[:, 0], coin_arr[:, 1]]).sum()
            # enemy scores
            for ei, (ex, ey) in enumerate(enemy_pos[:3]):
                scores[idx, 2 + ei] = 1.0 / dist[ex, ey]

        return scores

    def _kill_chance(self) -> bool:
        """True if dropping a bomb here would trap at least one enemy."""
        s0 = copy.deepcopy(self.game_state)
        s1 = self._predict_next(self.game_state['self'][-1])
        s1['bombs'] = copy.deepcopy(self.game_state['bombs'])
        s1['bombs'].append((self.game_state['self'][-1], s.BOMB_TIMER - 1))

        for i in range(len(s1['others'])):
            s0['self'], s0['others'][i] = s0['others'][i], s0['self']
            s1['self'], s1['others'][i] = s1['others'][i], s1['self']
            enemy_safe_before = sum(self._safe_counts(s0)) != 0
            enemy_safe_after  = sum(self._safe_counts(s1)) == 0
            we_can_escape     = self._safe_counts(self.game_state)[-1]  # WAIT slot
            if enemy_safe_before and enemy_safe_after and we_can_escape:
                return True
        return False

    # ── main call ───────────────────────────────────────────────────
    def __call__(self, game_state: Dict) -> np.ndarray:
        if game_state is None:
            return np.zeros(57, dtype=np.float32)

        self.game_state = game_state
        field, bombs, coins, others, (x, y) = (
            game_state['field'], game_state['bombs'],
            game_state['coins'], game_state['others'],
            game_state['self'][-1],
        )

        # ── tile one-hot for 5 neighbour cells ──────────────────────
        other_pos = {o[-1] for o in others}
        bomb_pos  = {(bx, by) for (bx, by), _ in bombs}
        coin_pos  = set(coins)

        tile_matrix = np.zeros((5, 5), dtype=np.float32)  # 5 directions × 5 categories
        directions  = [(x + dx, y + dy) for dx, dy in DIRECTIONS_WAIT]
        for i, (nx, ny) in enumerate(directions):
            raw = field[nx, ny]
            if (nx, ny) in other_pos:
                cat = 'enemy'
            elif (nx, ny) in bomb_pos:
                cat = 'bomb'
            elif (nx, ny) in coin_pos:
                cat = 'coin'
            elif raw == 1:
                cat = 'crate'
            elif raw == -1:
                cat = 'wall'
            else:
                cat = None   # free tile — no one-hot needed
            if cat is not None:
                tile_matrix[i, self.CAT_IDX[cat]] = 1.0

        # ── safe counts + score matrix ───────────────────────────────
        #scores = np.zeros((5, 5), dtype=np.float32)   # 先用零替代
        #kill   = 0.0
        safe    = self._safe_counts(game_state)
        scores  = self._score_matrix(safe)                 # (5,5)
        safe_col = np.array(safe, dtype=np.float32).reshape(5, 1)

        # combine: [one_hot(5) | scores(5) | safe(1)] per direction → (5,11)
        per_dir = np.hstack([tile_matrix, scores, safe_col])   # (5,11)

        # ── global features ──────────────────────────────────────────
        bomb_valid = float(self._safe_to_drop((x, y)) and game_state['self'][2])
        kill       = float(self._kill_chance())

        features = np.concatenate([per_dir.flatten(), [bomb_valid, kill]])

        return features.astype(np.float32)
