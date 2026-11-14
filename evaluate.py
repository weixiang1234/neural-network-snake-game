"""Read-only, headless greedy evaluation of an inference checkpoint."""

import argparse
import csv
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np

from baseline import summarize
from checkpoints import load_inference
from observations import encode_observation
from snake_env import SnakeEnv


def evaluate_policy(model, environment_config, seeds):
    env = SnakeEnv(environment_config)
    rows = []
    for seed in seeds:
        env.reset(seed=int(seed))
        episode_return = 0.0
        while True:
            q = model.forward(encode_observation(env, model.config.input_size)[None, :])[0]
            _, reward, terminated, truncated, info = env.step(int(np.argmax(q)))
            episode_return += reward
            if terminated or truncated:
                break
        rows.append({"env_seed": int(seed), "score": info["score"], "return": episode_return,
                     "episode_length": info["steps"], "ending_reason": info["ending_reason"]})
    return rows


def write_csv(path, rows, fieldnames):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed-start", type=int, default=100_000)
    parser.add_argument("--output", type=Path, default=Path("results/evaluation.csv"))
    args = parser.parse_args()
    if args.episodes < 1 or args.seed_start < 0 or args.output.suffix != ".csv":
        parser.error("use positive episodes, nonnegative seeds, and a .csv output")
    model, config = load_inference(args.checkpoint)
    rows = evaluate_policy(model, config.environment, range(args.seed_start, args.seed_start + args.episodes))
    write_csv(args.output, rows, list(rows[0]))
    report = {"checkpoint": str(args.checkpoint), "policy": "greedy_no_learning",
              "config": asdict(config), "summary": summarize(rows)}
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
