"""Hand-built boundary cases and seeded rollout checks; no Pygame imports."""

import unittest

from snake_env import SnakeConfig, SnakeEnv


class SnakeRulesTests(unittest.TestCase):
    def test_reset_body_fits_supported_board_sizes(self):
        for width, height in ((3, 3), (4, 3), (3, 7), (20, 20)):
            env = SnakeEnv(SnakeConfig(width, height))
            obs, info = env.reset(seed=5)
            self.assertEqual(len(set(obs.snake)), 3)
            self.assertTrue(all(0 <= x < width and 0 <= y < height for x, y in obs.snake))
            self.assertNotIn(obs.food, obs.snake)
            self.assertEqual((info["score"], info["steps"]), (0, 0))

    def fixture(self, snake, food, direction=0, limit=None):
        env = SnakeEnv(SnakeConfig(3, 3, limit))
        env.snake = list(snake)
        env.food = food
        env.direction = direction
        return env

    def test_relative_turns_in_each_heading(self):
        expected = {
            0: [(3, 2), (2, 3), (2, 1)],
            1: [(2, 3), (1, 2), (3, 2)],
            2: [(1, 2), (2, 1), (2, 3)],
            3: [(2, 1), (3, 2), (1, 2)],
        }
        for direction in range(4):
            for action in range(3):
                with self.subTest(direction=direction, action=action):
                    env = SnakeEnv(SnakeConfig(5, 5))
                    env.snake = [(2, 2)]
                    env.food = (0, 0)
                    env.direction = direction
                    obs, reward, done, truncated, _ = env.step(action)
                    self.assertEqual(obs.snake[0], expected[direction][action])
                    self.assertEqual((reward, done, truncated), (0, False, False))

    def test_growth_and_clock_reset(self):
        env = self.fixture([(1, 1), (0, 1), (0, 2)], (2, 1), limit=1)
        before = tuple(env.snake)
        obs, reward, done, truncated, info = env.step(0)
        self.assertEqual(obs.snake, ((2, 1),) + before)
        self.assertEqual((reward, done, truncated, info["score"]), (10, False, False, 1))
        self.assertEqual(obs.steps_without_food, 0)
        self.assertNotIn(obs.food, obs.snake)

    def test_wall_collision_and_reset_required(self):
        env = self.fixture([(2, 1), (1, 1), (0, 1)], (0, 0))
        self.assertTrue(env.would_collide(0))
        obs, reward, done, truncated, info = env.step(0)
        self.assertEqual((reward, done, truncated, info["ending_reason"]), (-10, True, False, "wall"))
        self.assertEqual(obs.snake[0], (2, 1))
        with self.assertRaises(RuntimeError):
            env.step(0)
        env.reset(seed=1)
        self.assertIsNone(env.ending_reason)

    def test_body_collision(self):
        env = self.fixture([(1, 1), (1, 2), (0, 2), (0, 1)], (2, 2), direction=1)
        self.assertTrue(env.would_collide(0))
        _, reward, done, _, info = env.step(0)
        self.assertEqual((reward, done, info["ending_reason"]), (-10, True, "body"))

    def test_enter_vacated_tail(self):
        env = self.fixture([(1, 1), (1, 2), (0, 2), (0, 1)], (2, 2), direction=3)
        self.assertFalse(env.would_collide(2))
        obs, reward, done, _, _ = env.step(2)
        self.assertEqual(obs.snake, ((0, 1), (1, 1), (1, 2), (0, 2)))
        self.assertEqual((reward, done), (0, False))

    def test_full_board_win(self):
        env = self.fixture(
            [(1, 0), (0, 0), (0, 1), (0, 2), (1, 2), (2, 2), (2, 1), (1, 1)],
            (2, 0),
        )
        obs, reward, done, truncated, info = env.step(0)
        self.assertEqual(len(set(obs.snake)), 9)
        self.assertIsNone(obs.food)
        self.assertEqual((reward, done, truncated, info["ending_reason"]), (10, True, False, "win"))
        self.assertEqual(info["score"], 1)

    def test_only_free_cell_gets_food(self):
        env = self.fixture(
            [(1, 0), (0, 0), (0, 1), (0, 2), (1, 2), (2, 2), (2, 1), (1, 1)],
            (2, 0),
        )
        for seed in range(30):
            env.rng.seed(seed)
            self.assertEqual(env._spawn_food(), (2, 0))

    def test_no_food_limit_is_terminal(self):
        env = SnakeEnv(SnakeConfig(7, 7, 2))
        env.food = (0, 0)
        self.assertFalse(env.step(0)[2])
        _, reward, done, truncated, info = env.step(0)
        self.assertEqual((reward, done, truncated, info["ending_reason"]), (-10, True, False, "no_food_limit"))
        self.assertEqual((info["steps"], info["steps_without_food"]), (2, 2))
        self.assertEqual(SnakeConfig().food_limit, 400)

    def test_seeded_replay_and_snapshot_independence(self):
        env = SnakeEnv()
        def rollout():
            initial = env.reset(seed=42)
            results = [initial]
            for action in [0, 1, 0, 2, 0, 0, 1] * 10:
                transition = env.step(action)
                results.append(transition)
                self.assertNotIn(transition[0].food, transition[0].snake)
                if transition[2]:
                    break
            return results
        first = rollout()
        self.assertEqual(first, rollout())
        self.assertEqual(first[0][0].snake[0], (10, 10))
        self.assertEqual(len(first[0][0].snake), 3)

    def test_food_free_across_seeds(self):
        env = SnakeEnv()
        for seed in range(100):
            obs, _ = env.reset(seed=seed)
            self.assertNotIn(obs.food, obs.snake)
            x, y = obs.food
            self.assertTrue(0 <= x < 20 and 0 <= y < 20)

    def test_prediction_matches_movement(self):
        import random
        rng = random.Random(19)
        env = SnakeEnv(SnakeConfig(5, 5))
        for seed in range(25):
            env.reset(seed=seed)
            for _ in range(50):
                action = rng.randrange(3)
                collision = env.would_collide(action)
                _, _, done, _, info = env.step(action)
                self.assertEqual(collision, info["ending_reason"] in ("wall", "body"))
                if done:
                    break

    def test_invalid_input_does_not_advance(self):
        env = SnakeEnv()
        before = env._observation()
        for action in (-1, 3, 0.0, True, "left", None):
            with self.assertRaises(ValueError):
                env.step(action)
            self.assertEqual(env._observation(), before)
            self.assertEqual(env.steps, 0)
        for config in ({"width": 2}, {"height": 2.5}, {"max_steps_without_food": 0}):
            with self.assertRaises(ValueError):
                SnakeConfig(**config)


if __name__ == "__main__":
    unittest.main()
