# how each action maps under rotation
ROTATION_MAP = {
    0:   {'UP':'UP',    'RIGHT':'RIGHT', 'DOWN':'DOWN',  'LEFT':'LEFT',  'WAIT':'WAIT', 'BOMB':'BOMB'},
    90:  {'UP':'RIGHT', 'RIGHT':'DOWN',  'DOWN':'LEFT',  'LEFT':'UP',    'WAIT':'WAIT', 'BOMB':'BOMB'},
    180: {'UP':'DOWN',  'RIGHT':'LEFT',  'DOWN':'UP',    'LEFT':'RIGHT', 'WAIT':'WAIT', 'BOMB':'BOMB'},
    270: {'UP':'LEFT',  'RIGHT':'UP',    'DOWN':'RIGHT', 'LEFT':'DOWN',  'WAIT':'WAIT', 'BOMB':'BOMB'},
}
HFLIP_MAP = {'UP':'UP',   'RIGHT':'LEFT', 'DOWN':'DOWN', 'LEFT':'RIGHT', 'WAIT':'WAIT', 'BOMB':'BOMB'}

# ── D_4 dihedral group: 8 unique symmetries ──
# (rotation, hflip) — vflip is redundant, equals (rot+180, hflip)
D4_SYMMETRIES = [
    (0,   False), (90,  False), (180, False), (270, False),
    (0,   True),  (90,  True),  (180, True),  (270, True),
]


def adjust_action(action: str, rotation=0, flip_h=False) -> str:
    """Apply rotation then optional horizontal flip to an action."""
    action = ROTATION_MAP[rotation][action]
    if flip_h:
        action = HFLIP_MAP[action]
    return action
