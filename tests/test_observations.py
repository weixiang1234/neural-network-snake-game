import copy
import unittest

import numpy as np

from snake_ai.observations import OBSERVATION_ORDER, encode_observation
from snake_ai.snake_env import SnakeConfig, SnakeEnv


class ObservationTests(unittest.TestCase):
    def setUp(self):
        self.env = SnakeEnv(SnakeConfig(5, 5))
        self.env.snake = [(2, 2), (1, 2), (0, 2)]
        self.env.food = (4, 1)

    def assert_vector(self, expected):
        vector = encode_observation(self.env)
        self.assertEqual(vector.shape, (11,))
        self.assertEqual(vector.dtype, np.float32)
        self.assertTrue(np.isfinite(vector).all())
        np.testing.assert_array_equal(vector, expected)

    def test_complete_hand_calculated_vector(self):
        self.assertEqual(len(OBSERVATION_ORDER), 11)
        self.assert_vector([0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0])

    def test_wall_danger_uses_relative_heading(self):
        self.env.snake = [(4, 2), (4, 3), (4, 4)]
        self.env.direction = 3  # up; right turn would hit right wall
        self.env.food = (0, 4)
        self.assert_vector([0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1])

    def test_body_danger_and_vacated_tail(self):
        self.env.snake = [(1, 1), (1, 2), (0, 2), (0, 1)]
        self.env.food = (2, 2)
        self.env.direction = 3  # up; left turn enters old tail legally
        self.assert_vector([0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1])
        self.env.direction = 1  # down; straight hits neck
        self.assert_vector([1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1])

    def test_all_heading_flags(self):
        expected = ([0, 1, 0, 0], [0, 0, 0, 1], [1, 0, 0, 0], [0, 0, 1, 0])
        for heading, flags in enumerate(expected):
            self.env.direction = heading
            np.testing.assert_array_equal(encode_observation(self.env)[3:7], flags)

    def test_food_signs_and_alignment(self):
        cases = [
            ((0, 2), [1, 0, 0, 0]), ((4, 2), [0, 1, 0, 0]),
            ((2, 0), [0, 0, 1, 0]), ((2, 4), [0, 0, 0, 1]),
            ((0, 0), [1, 0, 1, 0]), ((4, 4), [0, 1, 0, 1]),
        ]
        for food, expected in cases:
            self.env.food = food
            np.testing.assert_array_equal(encode_observation(self.env)[7:], expected)

    def test_full_board_terminal_food_flags(self):
        env = SnakeEnv(SnakeConfig(3, 3))
        env.snake = [(1, 0), (0, 0), (0, 1), (0, 2), (1, 2), (2, 2), (2, 1), (1, 1)]
        env.food = (2, 0)
        env.step(0)
        self.assertEqual(env.ending_reason, "win")
        np.testing.assert_array_equal(encode_observation(env)[7:], [0, 0, 0, 0])

    def test_same_features_can_have_different_outcomes(self):
        env = SnakeEnv(SnakeConfig(5, 5, 2))
        env.reset(seed=0)
        env.food = (0, 0)
        near_limit = copy.deepcopy(env)
        near_limit.steps_without_food = 1
        np.testing.assert_array_equal(encode_observation(env), encode_observation(near_limit))
        self.assertFalse(env.step(0)[2])
        self.assertTrue(near_limit.step(0)[2])

    def test_food_distance_is_omitted(self):
        self.env.food = (3, 1)
        near = encode_observation(self.env)
        self.env.food = (4, 0)
        np.testing.assert_array_equal(near, encode_observation(self.env))

    def test_fresh_vectors_do_not_change_environment(self):
        before = copy.deepcopy(self.env.__dict__)
        vector = encode_observation(self.env)
        vector[:] = 99
        self.assert_vector([0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0])
        self.assertEqual(self.env.snake, before["snake"])
        self.assertEqual(self.env.rng.getstate(), before["rng"].getstate())

    def test_danger_matches_actual_transitions(self):
        rng = np.random.default_rng(17)
        for seed in range(20):
            self.env.reset(seed=seed)
            for _ in range(40):
                features = encode_observation(self.env)
                for action in range(3):
                    probe = copy.deepcopy(self.env)
                    _, _, _, _, info = probe.step(action)
                    self.assertEqual(bool(features[action]), info["ending_reason"] in ("wall", "body"))
                if self.env.step(int(rng.integers(3)))[2]:
                    break


if __name__ == "__main__":
    unittest.main()
