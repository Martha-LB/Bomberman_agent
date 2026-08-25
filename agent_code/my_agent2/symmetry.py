import numpy as np

# how each action maps under rotation
ROTATION_MAP = {
    0:   {'UP':'UP',    'RIGHT':'RIGHT', 'DOWN':'DOWN',  'LEFT':'LEFT',  'WAIT':'WAIT', 'BOMB':'BOMB'},
    90:  {'UP':'RIGHT', 'RIGHT':'DOWN',  'DOWN':'LEFT',  'LEFT':'UP',    'WAIT':'WAIT', 'BOMB':'BOMB'},
    180: {'UP':'DOWN',  'RIGHT':'LEFT',  'DOWN':'UP',    'LEFT':'RIGHT', 'WAIT':'WAIT', 'BOMB':'BOMB'},
    270: {'UP':'LEFT',  'RIGHT':'UP',    'DOWN':'RIGHT', 'LEFT':'DOWN',  'WAIT':'WAIT', 'BOMB':'BOMB'},
}
HFLIP_MAP = {'UP':'UP',   'RIGHT':'LEFT', 'DOWN':'DOWN', 'LEFT':'RIGHT', 'WAIT':'WAIT', 'BOMB':'BOMB'}
VFLIP_MAP = {'UP':'DOWN', 'RIGHT':'RIGHT','DOWN':'UP',   'LEFT':'LEFT',  'WAIT':'WAIT', 'BOMB':'BOMB'}

DIR_IDX = {'UP': 0, 'RIGHT': 1, 'DOWN': 2, 'LEFT': 3, 'CENTER': 4}


def adjust_action(action: str, rotation=0, flip_h=False, flip_v=False) -> str:
    action = ROTATION_MAP[rotation][action]
    if flip_h:
        action = HFLIP_MAP[action]
    if flip_v:
        action = VFLIP_MAP[action]
    return action


def adjust_features(features: np.ndarray, rotation=0, flip_h=False, flip_v=False) -> np.ndarray:
    """
    Rotate/flip the 57-dim feature vector.
    The first 55 dims are (5 directions × 11 features); last 2 are global.
    We permute the direction rows to match the new orientation.
    """
    if features is None or features.size == 0:
        return features

    matrix = features[:55].reshape(5, 11).copy()   # (UP RIGHT DOWN LEFT CENTER) × 11
    global_ = features[55:]                         # bomb_valid, kill

    dirs = ['UP', 'RIGHT', 'DOWN', 'LEFT']
    n_rot = (rotation // 90) % 4
    dirs  = dirs[-n_rot:] + dirs[:-n_rot]           # rotate direction labels

    if flip_h:
        dirs[1], dirs[3] = dirs[3], dirs[1]         # swap RIGHT ↔ LEFT
    if flip_v:
        dirs[0], dirs[2] = dirs[2], dirs[0]         # swap UP ↔ DOWN

    new_matrix = np.empty_like(matrix)
    for i, d in enumerate(dirs):
        new_matrix[i] = matrix[DIR_IDX[d]]
    new_matrix[4] = matrix[DIR_IDX['CENTER']]       # CENTER is always last

    return np.concatenate([new_matrix.flatten(), global_])


def all_symmetries(features: np.ndarray, action: str):
    """
    Yield (aug_features, aug_action) for all 12 symmetries
    (4 rotations × 3 flip combinations).
    """
    for rot in [0, 90, 180, 270]:
        for fh, fv in [(False, False), (True, False), (False, True)]:
            yield (
                adjust_features(features, rot, fh, fv),
                adjust_action(action, rot, fh, fv),
            )
