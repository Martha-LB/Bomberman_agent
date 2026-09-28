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


def _permuted_dir_order(rotation=0, flip_h=False, flip_v=False) -> List[str]:
    dirs = ['UP', 'RIGHT', 'DOWN', 'LEFT']
    n_rot = (rotation // 90) % 4
    dirs  = dirs[-n_rot:] + dirs[:-n_rot]

    if flip_h:
        dirs[1], dirs[3] = dirs[3], dirs[1]
    if flip_v:
        dirs[0], dirs[2] = dirs[2], dirs[0]
    return dirs


def adjust_state(feature: List, rotation=0, flip_h=False, flip_v=False) -> List:
    """
    Feature layout (21 elements):
      [0:5]   tile tokens   [up, right, down, left, center]
      [5]     bomb_status
      [6:11]  coin_score    [up, right, down, left, center]
      [11:16] crate_score   [up, right, down, left, center]
      [16:21] enemy_score   [up, right, down, left, center]


    """
    if not feature:
        return feature

    dirs = _permuted_dir_order(rotation, flip_h, flip_v)

    def permute_group(offset):
        return [feature[offset + DIR_IDX[d]] for d in dirs] + [feature[offset + 4]]

    new_feat = []
    new_feat += permute_group(0)     # tile tokens (0:5)
    new_feat.append(feature[5])      # bomb_status - unchanged
    new_feat += permute_group(6)     # coin_score (6:11)
    new_feat += permute_group(11)    # crate_score (11:16)
    new_feat += permute_group(16)    # enemy_score (16:21)
    return new_feat


def all_symmetries(feature, action):
    """Yield (aug_feature, aug_action) for all 12 symmetries."""
    for rot in [0, 90, 180, 270]:
        for fh, fv in [(False, False), (True, False), (False, True)]:
            yield (
                adjust_state(feature, rot, fh, fv),
                adjust_action(action, rot, fh, fv),
            )