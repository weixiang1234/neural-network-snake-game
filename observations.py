"""Compact engineered features, separate from the raw board and renderer."""

import numpy as np

from snake_env import SnakeEnv


OBSERVATION_ORDER = (
    "danger_straight", "danger_right", "danger_left",
    "moving_left", "moving_right", "moving_up", "moving_down",
    "food_left", "food_right", "food_up", "food_down",
)

OBSERVATION_ORDER_13 = OBSERVATION_ORDER + ("food_dx_normalized", "food_dy_normalized")


def observation_order(input_size: int):
    """Use the exact feature ordering saved with a model checkpoint."""
    if input_size == 11:
        return OBSERVATION_ORDER
    if input_size == 13:
        return OBSERVATION_ORDER_13
    raise ValueError("supported observation sizes are 11 and 13")


def encode_observation(env: SnakeEnv, input_size: int = 11) -> np.ndarray:
    """Return a float32 feature vector matching the selected model input size.

    The 11 original features remain unchanged. V3 appends normalized food
    displacement (food minus head), preserving signs and approximate distance.
    The compact vector still omits distant body geometry and the no-food clock.
    """
    observation_order(input_size)  # Reject incompatible feature counts.
    x, y = env.snake[0]
    fx, fy = env.food if env.food is not None else (x, y)
    features = np.array([
        env.would_collide(0), env.would_collide(1), env.would_collide(2),
        env.direction == 2, env.direction == 0,
        env.direction == 3, env.direction == 1,
        fx < x, fx > x, fy < y, fy > y,
    ], dtype=np.float32)
    if input_size == 11:
        return features
    distance = np.array([
        (fx - x) / (env.config.width - 1),
        (fy - y) / (env.config.height - 1),
    ], dtype=np.float32)
    return np.concatenate((features, distance))
