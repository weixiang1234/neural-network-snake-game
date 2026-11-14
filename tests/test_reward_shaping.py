"""Checks for optional potential-based reward shaping."""

import csv
import math
import tempfile
import unittest
from pathlib import Path

from dqn import DQNConfig
from reward_shaping import food_potential, shaped_reward
from snake_env import SnakeConfig, SnakeEnv
from train import Trainer, TrainingConfig


class RewardShapingTests(unittest.TestCase):
    def test_proximity_and_terminal_potential(self):
        env = SnakeEnv(SnakeConfig(5, 5))
        env.snake = [(2, 2), (1, 2), (0, 2)]
        env.food = (4, 2)
        self.assertAlmostEqual(food_potential(env), 0.25)
        env.snake[0] = (3, 2)
        self.assertAlmostEqual(food_potential(env), 0.375)
        env.ending_reason = "no_food_limit"
        self.assertEqual(food_potential(env), 0)

    def test_reward_change_and_discounted_telescoping(self):
        scale, gamma = 2.0, 0.95
        nearer = shaped_reward(0, 0.25, 0.375, gamma, scale)
        farther = shaped_reward(0, 0.25, 0.125, gamma, scale)
        self.assertGreater(nearer, 0)
        self.assertLess(farther, 0)
        potentials = [0.2, 0.5, 0.3, 0.0]
        discounted_shaping = sum(
            gamma ** t * (shaped_reward(0, potentials[t], potentials[t + 1], gamma, scale))
            for t in range(len(potentials) - 1)
        )
        self.assertAlmostEqual(discounted_shaping, -scale * potentials[0])
        self.assertAlmostEqual(shaped_reward(-10, .9, 0.0, gamma, scale), -11.8)

    def test_invalid_scale(self):
        for bad in (-1, float("inf"), float("nan"), True):
            with self.assertRaises(ValueError):
                TrainingConfig(reward_shaping_scale=bad)

    def test_shaped_rewards_and_resume_metadata(self):
        with tempfile.TemporaryDirectory() as folder:
            config = TrainingConfig(
                agent=DQNConfig(seed=123), episodes=2, evaluation_every=0,
                checkpoint_every=2, reward_shaping_scale=2.0,
            )
            trainer = Trainer(config)
            trainer.run(folder)
            self.assertTrue(any(
                abs(float(v)) > 1e-5 for v in trainer.agent.replay.rewards[:len(trainer.agent.replay)]
                if abs(v) < 10
            ))
            with (Path(folder) / "training.csv").open() as file:
                rows = list(csv.DictReader(file))
            self.assertEqual(len(rows), 2)
            self.assertIn("learning_return", rows[0])
            self.assertNotEqual(float(rows[0]["learning_return"]), float(rows[0]["return"]))
            resumed = Trainer.resume(Path(folder) / "training.npz", episodes=3)
            self.assertEqual(resumed.config.reward_shaping_scale, 2.0)
            resumed.run(folder)
            self.assertEqual(len(resumed.records), 3)


if __name__ == "__main__":
    unittest.main()
