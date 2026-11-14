"""Focused tests of the frozen-policy safety filter."""

import unittest

import numpy as np

from safety_lookahead import select_action
from snake_env import SnakeConfig, SnakeEnv


class SafetyLookaheadTests(unittest.TestCase):
    def trapped_food_state(self):
        env = SnakeEnv(SnakeConfig(5, 5))
        env.snake = [
            (1, 1), (1, 0), (2, 0), (3, 0),
            (3, 1), (3, 2), (2, 2), (2, 3),
        ]
        env.direction = 1  # facing down; left turn enters food at (2, 1)
        env.food = (2, 1)  # taking food grows into a cul-de-sac
        env.steps = env.steps_without_food = env.score = 0
        env.ending_reason = None
        return env

    def test_zero_horizon_exactly_matches_greedy(self):
        env = self.trapped_food_state()
        q = np.array([9.0, 0.0, 12.0])
        decision = select_action(env, q, future_steps=0)
        self.assertEqual(decision.action, 2)
        self.assertFalse(decision.overridden)

    def test_one_step_prevents_food_growth_trap(self):
        env = self.trapped_food_state()
        self.assertFalse(env.would_collide(0))
        self.assertFalse(env.would_collide(2))
        after = select_action(env, [9.0, 0.0, 12.0], future_steps=1)
        self.assertEqual(after.action, 0)
        self.assertTrue(after.overridden)
        self.assertFalse(after.no_viable_action)

    def test_two_step_also_recognizes_trap(self):
        env = self.trapped_food_state()
        decision = select_action(env, [9.0, 0.0, 12.0], future_steps=2)
        self.assertEqual(decision.action, 0)
        self.assertTrue(decision.overridden)

    def test_does_not_modify_actual_environment_or_rng(self):
        env = self.trapped_food_state()
        before_obs = env._observation()
        before_info = env._info()
        before_rng = env.rng.getstate()
        for horizon in (0, 1, 2):
            select_action(env, [9.0, 0.0, 12.0], future_steps=horizon)
        self.assertEqual(env._observation(), before_obs)
        self.assertEqual(env._info(), before_info)
        self.assertEqual(env.rng.getstate(), before_rng)

    def test_keeps_greedy_when_it_has_a_safe_continuation(self):
        env = SnakeEnv(SnakeConfig(20, 20))
        env.reset(seed=3)
        env.food = (18, 18)
        for horizon in (1, 2):
            decision = select_action(env, [5.0, 0.0, 0.0], horizon)
            self.assertEqual(decision.action, 0)
            self.assertFalse(decision.overridden)
            self.assertFalse(decision.no_viable_action)

    def test_pure_mode_matches_existing_evaluation(self):
        from evaluate import evaluate_policy
        from network import MLP, NetworkConfig
        from benchmark_safety import run_episode

        model = MLP(NetworkConfig(), rng=np.random.default_rng(47))
        config = SnakeConfig(5, 5)
        seeds = range(21, 26)
        reference = evaluate_policy(model, config, seeds)
        actual = [run_episode(model, config, seed, future_steps=0) for seed in seeds]
        for expected, got in zip(reference, actual):
            for key in ("env_seed", "score", "return", "episode_length", "ending_reason"):
                self.assertEqual(got[key], expected[key])
            self.assertEqual(got["overrides"], 0)

    def test_invalid_inputs_and_completed_episode(self):
        env = self.trapped_food_state()
        for bad in (-1, 3, True, 1.5):
            with self.assertRaises(ValueError):
                select_action(env, [1, 2, 3], future_steps=bad)
        for bad in ([1, 2], [1, np.inf, 0], [[1, 2, 3]]):
            with self.assertRaises(ValueError):
                select_action(env, bad, future_steps=1)
        env.ending_reason = "body"
        with self.assertRaises(ValueError):
            select_action(env, [1, 2, 3], future_steps=1)


if __name__ == "__main__":
    unittest.main()
