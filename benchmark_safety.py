"""Compare frozen greedy DQN with 1- and 2-step safety lookahead on matched seeds."""

import argparse
from collections import Counter
import csv
import json
from pathlib import Path
from statistics import mean, median, pstdev

import numpy as np

from checkpoints import load_inference
from observations import encode_observation
from safety_lookahead import select_action
from snake_env import SnakeEnv


def run_episode(model, environment_config, seed, future_steps):
    env = SnakeEnv(environment_config)
    env.reset(seed=seed)
    seen = {}
    last_score = 0
    loop_period = None
    overrides = no_viable = 0
    episode_return = 0.0

    while env.ending_reason is None:
        if env.score != last_score:
            seen.clear()
            last_score = env.score
        signature = (tuple(env.snake), env.direction, env.food)
        if loop_period is None and signature in seen:
            loop_period = env.steps - seen[signature]
        else:
            seen.setdefault(signature, env.steps)

        inputs = encode_observation(env, model.config.input_size)[None, :]
        q = model.forward(inputs)[0]
        decision = select_action(env, q, future_steps=future_steps)
        overrides += int(decision.overridden)
        no_viable += int(decision.no_viable_action)
        _, reward, _, _, info = env.step(decision.action)
        episode_return += reward

    return {
        "env_seed": seed,
        "score": env.score,
        "return": episode_return,
        "episode_length": env.steps,
        "ending_reason": info["ending_reason"],
        "loop_detected": loop_period is not None,
        "first_cycle_length": loop_period if loop_period is not None else "",
        "overrides": overrides,
        "no_viable_steps": no_viable,
    }


def summarize(rows):
    scores = [row["score"] for row in rows]
    counts = Counter(row["ending_reason"] for row in rows)
    decisions = sum(row["episode_length"] for row in rows)
    overrides = sum(row["overrides"] for row in rows)
    return {
        "episodes": len(rows),
        "score_mean": mean(scores),
        "score_median": median(scores),
        "score_std_population": pstdev(scores),
        "score_min": min(scores),
        "score_max": max(scores),
        "zero_score_episodes": sum(score == 0 for score in scores),
        "repeated_cycles": sum(row["loop_detected"] for row in rows),
        "ending_reasons": dict(sorted(counts.items())),
        "total_decisions": decisions,
        "overrides": overrides,
        "override_rate": overrides / decisions if decisions else 0.0,
        "no_viable_steps": sum(row["no_viable_steps"] for row in rows),
    }


def compare(model, config, seeds):
    results = {}
    for future_steps in (0, 1, 2):
        rows = [run_episode(model, config.environment, seed, future_steps) for seed in seeds]
        results[future_steps] = (rows, summarize(rows))
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed-start", type=int, default=70000)
    parser.add_argument("--output-dir", type=Path, default=Path("results/safety_lookahead"))
    args = parser.parse_args()
    if args.episodes < 1 or args.seed_start < 0:
        parser.error("episodes must be positive and seed-start nonnegative")
    model, config = load_inference(args.checkpoint)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    seeds = range(args.seed_start, args.seed_start + args.episodes)

    results = compare(model, config, seeds)
    report = {
        "checkpoint": str(args.checkpoint),
        "seed_start": args.seed_start,
        "episodes": args.episodes,
        "description": "0=pure DQN, 1=one extra survivable move, 2=two extra survivable moves",
        "policies": {},
    }
    for steps, (rows, summary) in results.items():
        name = ("pure_dqn", "lookahead_1", "lookahead_2")[steps]
        path = args.output_dir / f"{name}.csv"
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        report["policies"][name] = summary
        print(f"{name}: {json.dumps(summary)}", flush=True)

    (args.output_dir / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Saved comparison to {args.output_dir}")


if __name__ == "__main__":
    main()
