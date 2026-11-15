import copy
from dataclasses import replace
import unittest

import numpy as np

from snake_ai.dqn import DQNAgent, DQNConfig, ReplayBatch, ReplayBuffer
from snake_ai.observations import encode_observation
from snake_ai.snake_env import SnakeConfig, SnakeEnv


class ReplayTests(unittest.TestCase):
    def test_overwrite_copies_and_sampling_without_replacement(self):
        replay = ReplayBuffer(3)
        for value in range(5):
            observation = np.full(11, value, dtype=np.float32)
            replay.add(observation, value % 3, value, observation + 1, False, value == 4)
            observation[:] = -100  # storage must own its copy
        self.assertEqual(len(replay), 3)
        batch = replay.sample(3, np.random.default_rng(0))
        self.assertEqual(set(batch.rewards.tolist()), {2, 3, 4})
        np.testing.assert_array_equal(batch.observations[:, 0], batch.rewards)
        np.testing.assert_array_equal(batch.next_observations[:, 0], batch.rewards + 1)
        np.testing.assert_array_equal(batch.truncated, batch.rewards == 4)
        np.testing.assert_array_equal(batch.actions, batch.rewards.astype(int) % 3)
        batch.observations[:] = -200
        self.assertTrue((replay.observations >= 2).all())
        with self.assertRaises(ValueError):
            replay.sample(4, np.random.default_rng(0))

    def test_invalid_entries_do_not_advance_replay(self):
        replay = ReplayBuffer(3)
        x = np.zeros(11)
        for entry in ((x[:10], 0, 0, x, False, False), (x, 3, 0, x, False, False),
                      (x, 0, np.nan, x, False, False), (x, 0, 0, x, 1, False)):
            with self.assertRaises(ValueError):
                replay.add(*entry)
            self.assertEqual(len(replay), 0)
        with self.assertRaises(ValueError):
            replay.sample(1, np.random.default_rng(0))


class DQNTests(unittest.TestCase):
    def agent(self, **overrides):
        config = replace(DQNConfig(), replay_capacity=8, batch_size=2, warmup_steps=2)
        return DQNAgent(replace(config, **overrides))

    def transition(self, agent, reward=1, terminated=False, truncated=False):
        return agent.observe(np.zeros(11), 0, reward, np.ones(11), terminated, truncated)

    def test_target_initial_copy_is_independent(self):
        agent = self.agent()
        for name in agent.online.params:
            np.testing.assert_array_equal(agent.online.params[name], agent.target.params[name])
            self.assertFalse(np.shares_memory(agent.online.params[name], agent.target.params[name]))
        agent.online.params["b2"][0] = 9
        self.assertEqual(agent.target.params["b2"][0], 0)

    def test_terminal_and_truncated_targets_hand_calculated(self):
        agent = self.agent()
        for p in agent.target.params.values():
            p[:] = 0
        agent.target.params["b2"][:] = [2, 5, 1]
        # Continuing, collision, external truncation, both flags, win.
        batch = ReplayBatch(np.zeros((5, 11)), np.zeros(5, dtype=int),
                            np.array([1, -10, 1, -10, 10], dtype=np.float32),
                            np.ones((5, 11)), np.array([False, True, False, True, True]),
                            np.array([False, False, True, True, False]))
        np.testing.assert_allclose(agent.td_targets(batch), [5.75, -10, 5.75, -10, 10])
        # Predictions come from target, regardless of online values.
        agent.online.params["b2"][:] = 100
        np.testing.assert_allclose(agent.td_targets(batch), [5.75, -10, 5.75, -10, 10])

    def test_terminal_next_states_are_not_evaluated(self):
        agent = self.agent()
        batch = ReplayBatch(np.zeros((1, 11)), np.array([0]), np.array([-10], dtype=np.float32),
                            np.full((1, 11), np.nan), np.array([True]), np.array([False]))
        np.testing.assert_array_equal(agent.td_targets(batch), [-10])

    def test_no_food_limit_from_real_environment_does_not_bootstrap(self):
        agent = self.agent(batch_size=1, warmup_steps=1)
        env = SnakeEnv(SnakeConfig(5, 5, 1))
        env.reset(seed=0)
        env.food = (0, 0)
        observation = encode_observation(env)
        _, reward, terminated, truncated, info = env.step(0)
        result = agent.observe(observation, 0, reward, encode_observation(env), terminated, truncated)
        self.assertEqual(info["ending_reason"], "no_food_limit")
        np.testing.assert_array_equal(result["targets"], [-10])

    def test_warmup_and_batch_requirements(self):
        agent = self.agent(warmup_steps=3)
        before = {name: p.copy() for name, p in agent.online.params.items()}
        self.assertIsNone(self.transition(agent))
        self.assertIsNone(self.transition(agent))
        self.assertEqual(agent.updates, 0)
        for name in before:
            np.testing.assert_array_equal(agent.online.params[name], before[name])
        self.assertIsNotNone(self.transition(agent))
        self.assertEqual((agent.environment_steps, agent.updates), (3, 1))
        agent = self.agent(warmup_steps=1, batch_size=3)
        self.assertIsNone(self.transition(agent))
        self.assertIsNone(self.transition(agent))
        self.assertIsNotNone(self.transition(agent))

    def test_target_frozen_until_optimizer_update_schedule(self):
        agent = self.agent(warmup_steps=3, target_sync_updates=2)
        target_before = {name: p.copy() for name, p in agent.target.params.items()}
        for _ in range(2):
            self.transition(agent)
        result = self.transition(agent)
        self.assertEqual(result["update"], 1)
        self.assertFalse(result["target_synchronized"])
        for name in target_before:
            np.testing.assert_array_equal(agent.target.params[name], target_before[name])
        self.assertTrue(any(not np.array_equal(agent.online.params[n], target_before[n]) for n in target_before))
        result = self.transition(agent)
        self.assertTrue(result["target_synchronized"])
        self.assertEqual((agent.environment_steps, agent.updates), (4, 2))
        for name in target_before:
            np.testing.assert_array_equal(agent.target.params[name], agent.online.params[name])

    def test_epsilon_schedule_and_greedy_evaluation_is_read_only(self):
        agent = self.agent(epsilon_decay_steps=10)
        self.assertEqual(agent.epsilon, 1)
        agent.environment_steps = 5
        self.assertAlmostEqual(agent.epsilon, 0.525)
        agent.environment_steps = 10
        self.assertAlmostEqual(agent.epsilon, 0.05)
        agent.environment_steps = 100
        self.assertAlmostEqual(agent.epsilon, 0.05)
        rng_before = copy.deepcopy(agent.action_rng.bit_generator.state)
        result = agent.act(np.zeros(11), explore=False)
        self.assertEqual(result.action, int(np.argmax(result.q_values)))
        self.assertEqual((result.epsilon, result.exploratory), (0, False))
        self.assertEqual(agent.action_rng.bit_generator.state, rng_before)
        self.assertEqual((agent.environment_steps, agent.updates, len(agent.replay)), (100, 0, 0))

    def test_exploration_samples_all_actions_even_with_all_danger_flags(self):
        agent = self.agent(epsilon_start=1, epsilon_end=1)
        observation = np.zeros(11)
        observation[:3] = 1
        decisions = [agent.act(observation) for _ in range(100)]
        self.assertEqual({d.action for d in decisions}, {0, 1, 2})
        self.assertTrue(all(d.exploratory for d in decisions))
        self.assertEqual(agent.environment_steps, 0)  # act alone never advances schedule

    def test_seeded_environment_action_replay_and_updates_repeat(self):
        def run():
            agent = self.agent(replay_capacity=16, batch_size=4, warmup_steps=4, target_sync_updates=3)
            env = SnakeEnv()
            env.reset(seed=42)
            records = []
            for _ in range(32):
                observation = encode_observation(env)
                decision = agent.act(observation)
                _, reward, terminated, truncated, _ = env.step(decision.action)
                result = agent.observe(observation, decision.action, reward, encode_observation(env), terminated, truncated)
                records.append((decision.action, None if result is None else result["loss"]))
                if terminated or truncated:
                    env.reset()
            return records, agent
        first, agent = run()
        second, repeated = run()
        self.assertEqual(first, second)
        self.assertEqual(agent.updates, 29)
        for name in agent.online.params:
            np.testing.assert_array_equal(agent.online.params[name], repeated.online.params[name])
            np.testing.assert_array_equal(agent.target.params[name], repeated.target.params[name])

    def test_invalid_configuration(self):
        for values in ({"gamma": 1.1}, {"learning_rate": 0}, {"batch_size": 0},
                       {"batch_size": 10001}, {"warmup_steps": 0}, {"seed": -1},
                       {"epsilon_end": 2}, {"target_sync_updates": 0}, {"epsilon_decay_steps": 0}):
            with self.assertRaises(ValueError):
                DQNConfig(**values)


if __name__ == "__main__":
    unittest.main()
