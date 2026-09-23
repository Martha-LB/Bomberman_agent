from typing import Iterable, Optional, Sequence, Tuple

import numpy as np


def contested_lethal_moves(
    physical_valid: np.ndarray,
    safe_counts: Sequence[int],
    action_positions: Sequence[Tuple[int, int]],
    enemy_destinations: Iterable[Tuple[int, int]],
) -> np.ndarray:
    """Find moves that become fatal if an enemy blocks the destination."""
    # Waiting is still safe, so collision is not fatal
    contested = np.zeros(5, dtype=bool)
    if int(safe_counts[4]) > 0:
        return contested

    # Check if enemy can take the same tile first
    enemy_destinations = set(enemy_destinations)
    for action_index, destination in enumerate(action_positions[:4]):
        contested[action_index] = bool(
            physical_valid[action_index]
            and destination in enemy_destinations
        )
    return contested


def safety_mask(
    physical_valid: np.ndarray,
    safe_counts: Sequence[int],
    bomb_escape_options: float,
) -> np.ndarray:
    """Keep actions that have a complete escape path."""
    # First five are movement and WAIT
    mask = np.asarray(physical_valid, dtype=np.float32).copy()
    mask[:5] *= np.asarray(safe_counts, dtype=np.float32) > 0
    mask[5] *= float(bomb_escape_options > 0.0)
    return mask


def select_action_mask(
    physical_valid: np.ndarray,
    base_valid: np.ndarray,
    robust_valid: Optional[np.ndarray] = None,
    survival_steps: Optional[Sequence[int]] = None,
    immediate_safe: Optional[Sequence[bool]] = None,
) -> np.ndarray:
    """Choose the safest non-empty action set."""
    # Best case, use opponent-safe actions
    if robust_valid is not None and robust_valid.any():
        return np.asarray(robust_valid, dtype=np.float32)

    # If OASF removed all, go back to known-hazard safety
    if base_valid.any():
        fallback = np.asarray(base_valid, dtype=np.float32).copy()
        if robust_valid is not None and not robust_valid[5]:
            fallback[5] = 0.0
        if fallback.any():
            return fallback
        return np.asarray(base_valid, dtype=np.float32)

    # No full escape, keep moves that live longest
    fallback = np.zeros(6, dtype=np.float32)
    physical_movement = np.asarray(physical_valid[:5], dtype=bool)
    if immediate_safe is not None:
        hazard_safe = np.asarray(immediate_safe, dtype=bool)
        if np.any(physical_movement & hazard_safe):
            physical_movement &= hazard_safe
    if survival_steps is None:
        fallback[:5] = physical_movement
    else:
        depths = np.asarray(survival_steps, dtype=np.int16)
        best_depth = int(np.max(depths[physical_movement]))
        fallback[:5] = physical_movement & (depths == best_depth)
    # WAIT is final emergency choice
    if not fallback.any():
        fallback[4] = 1.0
    return fallback


def safety_filter(
    *,
    enabled: bool,
    physical_valid: np.ndarray,
    safe_counts: Sequence[int],
    survival_steps: Sequence[int],
    immediate_safe: Sequence[bool],
    bomb_escape_options: float,
    enemy_forced_death: np.ndarray,
    action_positions: Sequence[Tuple[int, int]],
    enemy_destinations: Iterable[Tuple[int, int]],
    pessimistic_bomb_actions: int,
    nearby_bomb_ready_enemy: bool,
) -> np.ndarray:
    """Return the final action mask."""
    # First remove walls, bombs and known deadly paths
    base_valid = safety_mask(
        physical_valid,
        safe_counts,
        bomb_escape_options,
    )
    # Training can stop here without opponent guesses
    if not enabled:
        if base_valid.any():
            return base_valid
        return select_action_mask(
            physical_valid,
            base_valid,
            survival_steps=survival_steps,
            immediate_safe=immediate_safe,
        )

    # Evaluation also checks possible enemy actions
    robust_valid = base_valid.copy()
    robust_valid[:5] *= ~np.asarray(enemy_forced_death, dtype=bool)
    contested = contested_lethal_moves(
        physical_valid,
        safe_counts,
        action_positions,
        enemy_destinations,
    )
    robust_valid[:5] *= ~contested

    # Need more than one escape near a ready enemy
    robust_bomb = pessimistic_bomb_actions > 0
    if nearby_bomb_ready_enemy:
        robust_bomb &= pessimistic_bomb_actions >= 2
    robust_valid[5] *= float(robust_bomb)

    return select_action_mask(
        physical_valid,
        base_valid,
        robust_valid,
        survival_steps,
        immediate_safe,
    )
