"""Read-only short-horizon safety filter for an existing greedy Snake policy."""

from copy import deepcopy
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SafetyDecision:
    action: int
    greedy_action: int
    overridden: bool
    no_viable_action: bool


def _survival_exists(env, steps):
    """Is there a sequence of `steps` moves without task failure?"""
    if steps == 0:
        return True

    for action in range(3):
        if env.would_collide(action):
            continue
        next_env = deepcopy(env)
        _, _, terminated, truncated, info = next_env.step(action)
        if terminated or truncated:
            if info["ending_reason"] == "win":
                return True
            continue
        if _survival_exists(next_env, steps - 1):
            return True
    return False


def _viable(env, action, future_steps):
    if env.would_collide(action):
        return False
    next_env = deepcopy(env)
    _, _, terminated, truncated, info = next_env.step(action)
    if terminated or truncated:
        return info["ending_reason"] == "win"
    return _survival_exists(next_env, future_steps)


def select_action(env, q_values, future_steps=0):
    """Choose the highest-Q action that admits a safe short continuation.

    future_steps=0: unchanged greedy DQN (no safety filtering).
    future_steps=1: proposed move plus one additional survivable move.
    future_steps=2: proposed move plus two additional survivable moves.

    Search simulates the moving body, tail release, food growth, and game rules.
    It does not train the network or change the real environment/RNG.
    """
    if type(future_steps) is not int or future_steps not in (0, 1, 2):
        raise ValueError("future_steps must be 0, 1, or 2")
    q = np.asarray(q_values)
    if q.shape != (3,) or not np.isfinite(q).all():
        raise ValueError("q_values must be three finite numbers")
    if env.ending_reason is not None:
        raise ValueError("cannot choose an action after episode termination")

    greedy = int(np.argmax(q))
    if future_steps == 0 or _viable(env, greedy, future_steps):
        return SafetyDecision(greedy, greedy, False, False)

    # Preserve the network's ranking; only exclude unsafe alternatives.
    ranked = sorted(range(3), key=lambda a: (-float(q[a]), a))
    for action in ranked:
        if action != greedy and _viable(env, action, future_steps):
            return SafetyDecision(action, greedy, True, False)

    # If the horizon is infeasible, avoid an immediate collision when possible.
    for action in ranked:
        if not env.would_collide(action):
            return SafetyDecision(action, greedy, action != greedy, True)
    return SafetyDecision(greedy, greedy, False, True)
