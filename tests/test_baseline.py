import csv
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from baseline import evaluate_random, random_action, save_results, summarize
from snake_env import SnakeConfig


class BaselineTests(unittest.TestCase):
    def test_repeatability_and_episode_rng_independence(self):
        seeds = [7, 2, 19, 5]
        first = evaluate_random(seeds, policy_seed=8)
        self.assertEqual(first, evaluate_random(seeds, policy_seed=8))
        self.assertEqual(first, list(reversed(evaluate_random(reversed(seeds), policy_seed=8))))

    def test_scores_returns_and_endings_are_consistent(self):
        rows = evaluate_random(range(30), config=SnakeConfig(5, 5, 8))
        for row in rows:
            self.assertGreater(row["episode_length"], 0)
            self.assertTrue(row["terminated"])
            self.assertFalse(row["truncated"])
            self.assertIn(row["ending_reason"], ("wall", "body", "no_food_limit", "win"))
            final_penalty = 0 if row["ending_reason"] == "win" else 10
            self.assertEqual(row["return"], 10 * row["score"] - final_penalty)

    def test_actions_include_all_choices(self):
        # No board or danger mask is passed to the random policy.
        rng = np.random.default_rng(0)
        actions = [random_action(rng) for _ in range(300)]
        self.assertEqual(set(actions), {0, 1, 2})
        self.assertTrue(all(type(action) is int for action in actions))

    def test_summary_and_saved_episode_records(self):
        config = SnakeConfig(5, 5, 8)
        rows = evaluate_random(range(5), config=config)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "random.csv"
            save_results(path, rows, config)
            with path.open(newline="") as stream:
                saved = list(csv.DictReader(stream))
            self.assertEqual([int(row["env_seed"]) for row in saved], list(range(5)))
            self.assertEqual([int(row["score"]) for row in saved], [row["score"] for row in rows])
            metadata = json.loads(path.with_suffix(".json").read_text())
            self.assertEqual(metadata["summary"], summarize(rows))
            self.assertEqual(metadata["effective_no_food_limit"], 8)
            self.assertEqual(metadata["environment_config"]["width"], 5)

    def test_summary_on_known_results(self):
        rows = [dict(score=score, episode_length=10, **{"return": score * 10 - 10}, ending_reason="wall")
                for score in (0, 1, 2)]
        summary = summarize(rows)
        self.assertEqual((summary["score_mean"], summary["score_median"], summary["score_max"]), (1, 1, 2))
        self.assertAlmostEqual(summary["score_std_population"], (2 / 3) ** 0.5)
        self.assertEqual(summary["ending_reasons"], {"wall": 3})

    def test_invalid_seed_inputs(self):
        for seeds, policy_seed in (([], 0), ([-1], 0), ([True], 0), ([0], -1)):
            with self.assertRaises(ValueError):
                evaluate_random(seeds, policy_seed)


if __name__ == "__main__":
    unittest.main()
