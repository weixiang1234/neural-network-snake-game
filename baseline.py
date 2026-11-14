"""Headless uniform-random baseline with independent food and action randomness."""

import argparse
from collections import Counter
import csv
from dataclasses import asdict
import json
from numbers import Integral
from pathlib import Path
import platform

import numpy as np

from snake_env import SnakeConfig, SnakeEnv


def random_action(rng: np.random.Generator) -> int:
    """Uniform over all three actions; receives no board data or action mask."""
    return int(rng.integers(0, 3))


def evaluate_random(seeds, policy_seed=2026, config=None):
    """One episode per seed, with a separate per-episode action RNG.

    SeedSequence([policy_seed, env_seed]) makes an episode reproducible even
    when the list is reordered or earlier episodes consume different steps.
    """
    seeds = tuple(seeds)
    if not seeds:
        raise ValueError("at least one environment seed is required")
    for seed in (policy_seed, *seeds):
        if isinstance(seed, bool) or not isinstance(seed, Integral) or seed < 0:
            raise ValueError("seeds must be nonnegative integers")
    config = config or SnakeConfig()
    env = SnakeEnv(config)
    rows = []
    for seed in seeds:
        env.reset(seed=int(seed))
        action_rng = np.random.default_rng(np.random.SeedSequence([policy_seed, seed]))
        episode_return = 0.0
        while True:
            _, reward, terminated, truncated, info = env.step(random_action(action_rng))
            episode_return += reward
            if terminated or truncated:
                break
        rows.append({
            "policy": "uniform_random",
            "env_seed": int(seed),
            "policy_seed": int(policy_seed),
            "score": info["score"],
            "episode_length": info["steps"],
            "return": episode_return,
            "ending_reason": info["ending_reason"],
            "terminated": terminated,
            "truncated": truncated,
        })
    return rows


def summarize(rows):
    if not rows:
        raise ValueError("cannot summarize empty results")
    scores = np.array([row["score"] for row in rows])
    return {
        "episodes": len(rows),
        "score_mean": float(scores.mean()),
        "score_median": float(np.median(scores)),
        "score_std_population": float(scores.std()),
        "score_min": int(scores.min()),
        "score_max": int(scores.max()),
        "episode_length_mean": float(np.mean([row["episode_length"] for row in rows])),
        "return_mean": float(np.mean([row["return"] for row in rows])),
        "ending_reasons": dict(sorted(Counter(row["ending_reason"] for row in rows).items())),
    }


def save_results(path, rows, config):
    """Keep individual episodes in CSV and rules/summary in adjacent JSON."""
    path = Path(path)
    if path.suffix.lower() != ".csv":
        raise ValueError("output path must end in .csv")
    summary = summarize(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    metadata = {
        "policy": "uniform_random",
        "action_mapping": {"0": "straight", "1": "right", "2": "left"},
        "action_rng": "NumPy default_rng(SeedSequence([policy_seed, env_seed])) per episode",
        "environment_rng": "Python random.Random, reset with env_seed per episode",
        "environment_config": asdict(config),
        "effective_no_food_limit": config.food_limit,
        "rewards": {"food_including_win": 10, "collision": -10, "no_food_limit": -10, "otherwise": 0},
        "ending_semantics": "wall, body, win, no_food_limit are terminal; no truncations",
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "summary": summary,
    }
    path.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--policy-seed", type=int, default=2026)
    parser.add_argument("--output", type=Path, default=Path("results/random_baseline.csv"))
    args = parser.parse_args()
    if args.episodes < 1 or args.seed_start < 0 or args.policy_seed < 0:
        parser.error("episodes must be positive and seeds nonnegative")
    if args.output.suffix.lower() != ".csv":
        parser.error("output must end in .csv")
    config = SnakeConfig()
    rows = evaluate_random(range(args.seed_start, args.seed_start + args.episodes), args.policy_seed, config)
    save_results(args.output, rows, config)
    print(json.dumps(summarize(rows), indent=2))
    print(f"Saved episodes: {args.output}")
    print(f"Saved rules and summary: {args.output.with_suffix('.json')}")


if __name__ == "__main__":
    main()
