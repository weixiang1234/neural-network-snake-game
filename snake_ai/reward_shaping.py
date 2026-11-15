"""Potential-based reward shaping for Snake training only."""


def food_potential(env):
    """Normalized proximity to food; terminal states have zero potential."""
    if env.ending_reason is not None or env.food is None:
        return 0.0
    x, y = env.snake[0]
    fx, fy = env.food
    max_distance = env.config.width + env.config.height - 2
    distance = abs(fx - x) + abs(fy - y)
    return 0.5 - distance / max_distance


def shaped_reward(task_reward, before, after, gamma, scale):
    """Keep evaluation/task rewards separate from learning rewards."""
    return task_reward + scale * (gamma * after - before)
