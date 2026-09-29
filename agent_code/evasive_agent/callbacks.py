import numpy as np

import settings as s


MOVES = {
    "UP": (0, -1),
    "RIGHT": (1, 0),
    "DOWN": (0, 1),
    "LEFT": (-1, 0),
    "WAIT": (0, 0),
}


def setup(self):
    # Prepare random tie-breaking
    self.rng = np.random.default_rng()


def _danger_times(game_state):
    # Return when each tile will next be hit by an explosion
    field = game_state["field"]
    danger = np.full(field.shape, s.BOMB_TIMER + 2, dtype=np.int16)
    danger[game_state["explosion_map"] > 0] = 0

    for origin, timer in game_state["bombs"]:
        x, y = origin
        danger[x, y] = min(danger[x, y], int(timer) + 1)
        for dx, dy in ((0, -1), (1, 0), (0, 1), (-1, 0)):
            for distance in range(1, s.BOMB_POWER + 1):
                position = (x + dx * distance, y + dy * distance)
                tile = field[position]
                if tile == -1:
                    break
                danger[position] = min(danger[position], int(timer) + 1)
                # In this environment crates are destroyed but do not stop
                # the blast. Only stone walls end a bomb ray.
    return danger


def _legal_actions(game_state):
    # Return moves that are physically possible now
    field = game_state["field"]
    x, y = game_state["self"][-1]
    bombs = {tuple(position) for position, _ in game_state["bombs"]}
    opponents = {tuple(agent[-1]) for agent in game_state["others"]}
    actions = []

    for action, (dx, dy) in MOVES.items():
        destination = (x + dx, y + dy)
        if field[destination] != 0:
            continue
        if destination in bombs or destination in opponents:
            continue
        if game_state["explosion_map"][destination] > 0:
            continue
        actions.append(action)
    return actions or ["WAIT"]


def _free_neighbors(game_state, position):
    # Count open exits around one tile
    field = game_state["field"]
    bombs = {tuple(cell) for cell, _ in game_state["bombs"]}
    opponents = {tuple(agent[-1]) for agent in game_state["others"]}
    count = 0
    for dx, dy in tuple(MOVES.values())[:4]:
        neighbor = (position[0] + dx, position[1] + dy)
        if field[neighbor] == 0 and neighbor not in bombs and neighbor not in opponents:
            count += 1
    return count


def act(self, game_state):
    # Evasive bombs without ever dropping one
    danger = _danger_times(game_state)
    x, y = game_state["self"][-1]
    coins = [tuple(coin) for coin in game_state["coins"]]
    choices = []

    for action in _legal_actions(game_state):
        dx, dy = MOVES[action]
        destination = (x + dx, y + dy)
        coin_distance = min(
            (
                abs(destination[0] - coin[0]) + abs(destination[1] - coin[1])
                for coin in coins
            ),
            default=0,
        )
        score = (
            int(danger[destination]),
            _free_neighbors(game_state, destination),
            -coin_distance,
            float(self.rng.random()),
        )
        choices.append((score, action))

    return max(choices)[1]
