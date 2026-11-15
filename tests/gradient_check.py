"""Numerically check every parameter on a tiny, fixed float64 batch."""

import numpy as np

from snake_ai.network import MLP, NetworkConfig


def gradient_check():
    model = MLP(NetworkConfig(2, 3, 3), rng=np.random.default_rng(7), dtype=np.float64)
    # Explicit fixture: both active/inactive ReLUs, safely away from zero.
    model.params["W1"][...] = [[0.4, -0.3, 0.2], [-0.2, 0.5, 0.6]]
    model.params["b1"][...] = [0.3, -0.2, 0.1]
    x = np.array([[0.2, -0.4], [0.7, 0.3], [-0.5, 0.6]], dtype=np.float64)
    actions = np.array([0, 2, 1])
    targets = np.array([0.8, -0.3, 0.4], dtype=np.float64)
    x.flags.writeable = targets.flags.writeable = actions.flags.writeable = False
    epsilon = 1e-6
    margin = float(np.min(np.abs(model.activations(x)["pre_hidden"])))
    if margin <= 100 * epsilon:
        raise AssertionError("fixture is too close to a ReLU kink")
    loss_before, analytical = model.loss_and_gradients(x, actions, targets)
    numerical = {}
    for name, parameter in model.params.items():
        numerical[name] = np.empty_like(parameter)
        for index in np.ndindex(parameter.shape):
            original = parameter[index].copy()
            try:
                parameter[index] = original + epsilon
                plus = model.loss_and_gradients(x, actions, targets)[0]
                parameter[index] = original - epsilon
                minus = model.loss_and_gradients(x, actions, targets)[0]
            finally:
                parameter[index] = original
            numerical[name][index] = (plus - minus) / (2 * epsilon)
    exact = np.concatenate([value.ravel() for value in analytical.values()])
    estimated = np.concatenate([numerical[name].ravel() for name in analytical])
    max_error = float(np.max(np.abs(exact - estimated)))
    relative_error = float(np.linalg.norm(exact - estimated) / max(1e-12, np.linalg.norm(exact) + np.linalg.norm(estimated)))
    if max_error > 1e-7 or relative_error > 1e-6:
        raise AssertionError(f"gradient check failed: absolute={max_error}, relative={relative_error}")
    model.sgd_step(analytical, learning_rate=0.01)
    loss_after, _ = model.loss_and_gradients(x, actions, targets)
    if not loss_after < loss_before:
        raise AssertionError("small SGD update did not reduce fixed-batch loss")
    return {
        "parameters_checked": len(exact), "min_abs_pre_activation": margin,
        "max_absolute_error": max_error, "relative_l2_error": relative_error,
        "loss_before": loss_before, "loss_after": loss_after,
    }


def main():
    import json
    from snake_ai.observations import encode_observation
    from snake_ai.snake_env import SnakeEnv

    print("Fixed-target gradient check (float64, epsilon=1e-6):")
    print(json.dumps(gradient_check(), indent=2))
    env = SnakeEnv()
    env.reset(seed=0)
    model = MLP(rng=np.random.default_rng(0))
    x = encode_observation(env)[None, :]
    cache = model.activations(x)
    print(f"Default model: input {x.shape}, hidden {cache['hidden'].shape}, Q {cache['q_values'].shape}")
    print(f"Untrained Q values: {cache['q_values'][0].tolist()}")
    print(f"Greedy action: {int(np.argmax(cache['q_values'][0]))} (untrained; no skill implied)")
    print("PASS: finite differences agree and one SGD step reduces fixed-target loss.")


if __name__ == "__main__":
    main()
