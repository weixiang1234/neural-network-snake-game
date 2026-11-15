"""Detect exact greedy-policy cycles without changing the Snake environment."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from snake_ai.checkpoints import load_inference
from snake_ai.observations import encode_observation
from snake_ai.snake_env import SnakeEnv


def board_signature(env):
    """Exclude the no-food clock, which both feature sets omit."""
    return (tuple(env.snake), env.direction, env.food)


def inspect_episode(model, env, seed):
    env.reset(seed=seed)
    seen = {}
    last_score = env.score
    first_loop_step = None
    cycle_length = None
    without_food_at_loop = None

    while env.ending_reason is None:
        if env.score != last_score:
            seen.clear()  # A new food location starts a new search.
            last_score = env.score

        signature = board_signature(env)
        previous_step = seen.get(signature)
        if previous_step is None:
            seen[signature] = env.steps
        elif first_loop_step is None:
            first_loop_step = env.steps
            cycle_length = env.steps - previous_step
            without_food_at_loop = env.steps_without_food

        features = encode_observation(env, model.config.input_size)
        q_values = model.forward(features[None, :])[0]
        env.step(int(np.argmax(q_values)))

    return {
        "env_seed": seed,
        "score": env.score,
        "episode_length": env.steps,
        "ending_reason": env.ending_reason,
        "loop_detected": first_loop_step is not None,
        "first_loop_step": first_loop_step,
        "cycle_length": cycle_length,
        "steps_without_food_at_loop": without_food_at_loop,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("results/m5_learning_300/policy.npz"))
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed-start", type=int, default=20000)
    parser.add_argument("--output", type=Path, default=Path("results/m5_learning_300/loop_diagnosis.csv"))
    args = parser.parse_args()
    if args.episodes < 1 or args.seed_start < 0 or args.output.suffix != ".csv":
        parser.error("use positive episodes, nonnegative seeds and a .csv output")

    model, config = load_inference(args.checkpoint)
    env = SnakeEnv(config.environment)
    rows = [inspect_episode(model, env, args.seed_start + i) for i in range(args.episodes)]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    looped = sum(row["loop_detected"] for row in rows)
    no_food_ends = sum(row["ending_reason"] == "no_food_limit" for row in rows)
    print(json.dumps({
        "episodes": len(rows),
        "exact_repeated_states_detected": looped,
        "no_food_limit_endings": no_food_ends,
        "mean_score": sum(row["score"] for row in rows) / len(rows),
        "output": str(args.output),
    }, indent=2))
    print("\nFirst episodes:")
    for row in rows[:10]:
        print(f"seed={row['env_seed']} score={row['score']} end={row['ending_reason']} "
              f"loop={row['loop_detected']} period={row['cycle_length']} "
              f"first_loop_step={row['first_loop_step']}")


if __name__ == "__main__":
    main()
