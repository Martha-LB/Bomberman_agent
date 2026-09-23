# Action names after rotating the board.
ROTATION_MAP = {
    0:   {'UP':'UP',    'RIGHT':'RIGHT', 'DOWN':'DOWN',  'LEFT':'LEFT',  'WAIT':'WAIT', 'BOMB':'BOMB'},
    90:  {'UP':'RIGHT', 'RIGHT':'DOWN',  'DOWN':'LEFT',  'LEFT':'UP',    'WAIT':'WAIT', 'BOMB':'BOMB'},
    180: {'UP':'DOWN',  'RIGHT':'LEFT',  'DOWN':'UP',    'LEFT':'RIGHT', 'WAIT':'WAIT', 'BOMB':'BOMB'},
    270: {'UP':'LEFT',  'RIGHT':'UP',    'DOWN':'RIGHT', 'LEFT':'DOWN',  'WAIT':'WAIT', 'BOMB':'BOMB'},
}
HFLIP_MAP = {
    'UP': 'UP',
    'RIGHT': 'LEFT',
    'DOWN': 'DOWN',
    'LEFT': 'RIGHT',
    'WAIT': 'WAIT',
    'BOMB': 'BOMB',
}

# Four rotations, with and without a horizontal flip.
D4_SYMMETRIES = [
    (0,   False), (90,  False), (180, False), (270, False),
    (0,   True),  (90,  True),  (180, True),  (270, True),
]


def adjust_action(action: str, rotation=0, flip_h=False) -> str:
    """Rotate and optionally flip an action name."""
    # Rotate first, flip second
    action = ROTATION_MAP[rotation][action]
    if flip_h:
        action = HFLIP_MAP[action]
    return action
