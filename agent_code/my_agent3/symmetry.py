from typing import List


# how each action maps under rotation
ROTATION_MAP = {
    0:   {'UP':'UP',    'RIGHT':'RIGHT', 'DOWN':'DOWN',  'LEFT':'LEFT',  'WAIT':'WAIT', 'BOMB':'BOMB'},
    90:  {'UP':'RIGHT', 'RIGHT':'DOWN',  'DOWN':'LEFT',  'LEFT':'UP',    'WAIT':'WAIT', 'BOMB':'BOMB'},
    180: {'UP':'DOWN',  'RIGHT':'LEFT',  'DOWN':'UP',    'LEFT':'RIGHT', 'WAIT':'WAIT', 'BOMB':'BOMB'},
    270: {'UP':'LEFT',  'RIGHT':'UP',    'DOWN':'RIGHT', 'LEFT':'DOWN',  'WAIT':'WAIT', 'BOMB':'BOMB'},
}
HFLIP_MAP = {'UP':'UP',   'RIGHT':'LEFT', 'DOWN':'DOWN', 'LEFT':'RIGHT', 'WAIT':'WAIT', 'BOMB':'BOMB'}
VFLIP_MAP = {'UP':'DOWN', 'RIGHT':'RIGHT','DOWN':'UP',   'LEFT':'LEFT',  'WAIT':'WAIT', 'BOMB':'BOMB'}

DIR_IDX = {'UP': 0, 'RIGHT': 1, 'DOWN': 2, 'LEFT': 3, 'WAIT': 4}


def adjust_action(action: str, rotation=0, flip_h=False, flip_v=False) -> str:
    action = ROTATION_MAP[rotation][action]
    if flip_h:
        action = HFLIP_MAP[action]
    if flip_v:
        action = VFLIP_MAP[action]
    return action


def adjust_state(feature: List[str], rotation=0, flip_h=False, flip_v=False) -> List[str]:
    """
    Feature layout: [up, right, down, left, center, bomb_status]  (6 tokens)
    Permute the first 4 slots according to rotation/flip. Center + bomb stay.
    """
    if not feature:
        return feature

    dirs = ['UP', 'RIGHT', 'DOWN', 'LEFT']
    n_rot = (rotation // 90) % 4
    dirs  = dirs[-n_rot:] + dirs[:-n_rot]

    if flip_h:
        dirs[1], dirs[3] = dirs[3], dirs[1]
    if flip_v:
        dirs[0], dirs[2] = dirs[2], dirs[0]

    new_feat = [feature[DIR_IDX[d]] for d in dirs]
    new_feat.append(feature[4])   # center (WAIT slot)
    new_feat.append(feature[5])   # bomb_status
    return new_feat


def all_symmetries(feature, action):
    """Yield (aug_feature, aug_action) for all 12 symmetries."""
    for rot in [0, 90, 180, 270]:
        for fh, fv in [(False, False), (True, False), (False, True)]:
            yield (
                adjust_state(feature, rot, fh, fv),
                adjust_action(action, rot, fh, fv),
            )