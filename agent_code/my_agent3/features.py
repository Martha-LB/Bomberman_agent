import copy
from collections import deque, defaultdict
from typing import Dict, List

import numpy as np
import settings as s


DIRECTIONS      = [(0, -1), (1, 0), (0, 1), (-1, 0)]  # UP, RIGHT, DOWN, LEFT
DIRECTIONS_WAIT = DIRECTIONS + [(0, 0)]


class Feature:
    """
    Produces a 7-token string feature per game step:
      [up, right, down, left, center, has_bomb, bomb_status]

    Token values:
      - Directions: 'block' / 'free' / 'coin' / 'enemy' / 'dead' / 'target'
      - has_bomb  : 'True' / 'False'
      - bomb slot : 'True' / 'False' / 'dead' / 'target' / 'KILL!'

    The 'target' token marks the best direction as determined by look_for_target(),
    which uses BFS + weighted scoring. The 'KILL!' token marks a bomb placement
    that would trap an enemy.
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

    # ── safe positions reachable per first-step direction ──
    def _find_safe_positions(self, state, max_steps=5) -> List[int]:
        field, expl, bombs, others, pos = (
            state['field'], state['explosion_map'],
            state['bombs'], state['others'], state['self'][-1],
        )
        danger    = self._danger_map(bombs)
        other_pos = {o[-1] for o in others}
        bomb_dict = {(bx, by): t for (bx, by), t in bombs}

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
                if (nx, ny) in other_pos and steps == 0:
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
        danger    = self._danger_map(bombs)
        other_pos = {o[-1] for o in others}
        bomb_pos  = {(bx, by): t for (bx, by), t in bombs}
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
                if (nx, ny) in other_pos and steps == 0:
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

    # ── the star: rule-based target picker ─────────────────
    def _look_for_target(self, safe: List[int], has_bomb: bool) -> int:
        """
        Returns index of best action (0-4 for move dirs, 5 for BOMB), or -1 if none safe.
        Uses BFS + weighted scoring; switches weights when in danger.
        """
        if sum(safe) == 0:
            # 优先放炸弹（如果有）打破僵局
            x, y = self.game_state['self'][-1]
            if has_bomb and self._is_safe_to_drop((x, y)):
                return 5  # BOMB
            return -1

        ns    = self._predict_next()
        field = ns['field']
        bombs = ns['bombs']
        coins = ns['coins']
        others = ns['others']
        x, y  = ns['self'][-1]

        danger = self._danger_map(bombs)
        directions = [(x + dx, y + dy) for dx, dy in DIRECTIONS_WAIT]

        # dynamic weights: aggressive when safe, defensive when in danger
        if danger[x, y] == 0:
            w = {"crates": 1, "coins": 300, "enemy0": 30,
                 "enemies": -5, "escape": 0.1}
        else:
            w = {"crates": 1, "coins": 300, "enemy0": 0,
                 "enemies": -300, "escape": 10}

        # build target maps
        coin_map = np.zeros_like(field, dtype=float)
        if coins:
            cp = np.array(coins)
            coin_map[cp[:, 0], cp[:, 1]] = 1

        enemies_map = np.zeros_like(field, dtype=float)
        if others:
            op = np.array([o[-1] for o in others])
            enemies_map[op[:, 0], op[:, 1]] = 1

        enemy0_map = np.zeros_like(field, dtype=float)
        if self.main_enemy:
            ep = {o[0]: o[-1] for o in others}
            e0 = ep.get(self.main_enemy)
            if e0:
                enemy0_map[e0[0], e0[1]] = 1

        crate_map = self._crate_neighbors(field)
        # only count crates within BOMB_POWER (avoid wasting BFS)
        m_dist = (np.abs(np.arange(crate_map.shape[0])[:, None] - x) +
                  np.abs(np.arange(crate_map.shape[1]) - y))
        indices = np.argwhere((crate_map > 0) & (m_dist <= s.BOMB_POWER))
        for i, j in indices:
            crate_map[i, j] *= self._is_safe_to_drop((i, j))

        # score each safe direction
        scores = {i: 0.0 for i in range(5) if safe[i]}
        for i in scores:
            nx, ny = directions[i]
            d      = self._bfs(ns, nx, ny) + 1  # avoid div/0

            scores[i] += w['coins']   * (coin_map    / d).sum()
            scores[i] += w['enemy0']  * (enemy0_map  / d).sum()
            scores[i] += w['enemies'] * (enemies_map / d).sum()
            scores[i] += w['crates']  * (crate_map   / d).sum()
            scores[i] += safe[i] * w['escape']

        max_score    = max(scores.values())
        best_actions = [idx for idx, v in scores.items() if v == max_score]
        target       = np.random.choice(best_actions)

        # if WAIT is best: consider dropping bomb instead
        if target == 4:
            if target == 4 and has_bomb:
                # 检查是否值得放炸弹攻击敌人
                if self._is_safe_to_drop((x, y)):
                    # 看看爆炸范围内有没有敌人
                    for other in others:
                        ox, oy = other[-1]
                        if abs(ox - x) + abs(oy - y) <= s.BOMB_POWER:
                            return 5  # BOMB
            
            if crate_map[x, y] > 0 and self._is_safe_to_drop((x, y)) and has_bomb:
                return 5  # BOMB
            # else pick 2nd-best (excluding WAIT)
            scores[4] = -np.inf
            if not any(v > -np.inf for v in scores.values()):
                return 4
            max_score    = max(scores.values())
            best_actions = [idx for idx, v in scores.items() if v == max_score]
            target       = np.random.choice(best_actions)

        return int(target)

    # ── detect kill opportunity ────────────────────────────
    def _is_chance_to_kill(self) -> bool:
        my_pos = self.game_state['self'][-1]

        # Escape check must run on the state AFTER our hypothetical bomb exists,
        # not on the current board (which has no bomb on it yet). _is_safe_to_drop
        # already does this correctly, so reuse it instead of re-deriving it here.
        # This is independent of which opponent we're checking, so compute once.
        if not self._is_safe_to_drop(my_pos):
            return False

        for i in range(len(self.game_state['others'])):
            # Fresh copies every iteration: s0/s1 get a self<->other[i] swap below,
            # and reusing copies across iterations would leave iteration i-1's swap
            # in place, corrupting the check for every opponent after the first.
            s0 = copy.deepcopy(self.game_state)
            s1 = self._predict_next(my_pos)
            s1['bombs'] = copy.deepcopy(self.game_state['bombs'])
            s1['bombs'].append((my_pos, s.BOMB_TIMER - 1))

            s0['self'], s0['others'][i] = s0['others'][i], s0['self']
            s1['self'], s1['others'][i] = s1['others'][i], s1['self']
            if (sum(self._find_safe_positions(s0)) != 0 and
                sum(self._find_safe_positions(s1)) == 0):
                return True
        return False

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

        # ── step 4: mark best target direction ────────────
        target = self._look_for_target(safe, game_state['self'][2])
        if target >= 0:
            features[target] = 'target'
        else:
            # 没有安全方向：尝试放炸弹，或者随便走一个非死亡方向
            if game_state['self'][2] and features[-1] != 'dead':
                features[-1] = 'target'   # BOMB
            else:
                # 找一个不是 'dead' 的方向
                for i in range(4):
                    if features[i] != 'dead':
                        features[i] = 'target'
                        break
                else:
                    features[-2] = 'target'   # if nothing safe, target the WAIT slot? use has_bomb slot as fallback

        # ── step 5: mark KILL opportunity ─────────────────
        if self._is_chance_to_kill() and features[-1] not in ['dead', 'False']:
            features[-1] = 'KILL!'

        return features
