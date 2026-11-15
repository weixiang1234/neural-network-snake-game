"""Compatibility and feature checks for the 13-input Snake DQN."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from snake_ai.checkpoints import load_inference, load_training, save_inference, save_training
from snake_ai.dqn import DQNAgent, DQNConfig
from snake_ai.evaluate import evaluate_policy
from snake_ai.network import NetworkConfig
from snake_ai.observations import OBSERVATION_ORDER, OBSERVATION_ORDER_13, encode_observation, observation_order
from snake_ai.snake_env import SnakeConfig, SnakeEnv
from train import Trainer, TrainingConfig


class V3Tests(unittest.TestCase):
    def test_displacements_and_legacy_prefix(self):
        env = SnakeEnv(SnakeConfig(5, 5))
        env.snake = [(2, 2), (1, 2), (0, 2)]
        env.direction = 0
        env.food = (4, 1)
        original = encode_observation(env)
        expanded = encode_observation(env, 13)
        self.assertEqual(original.shape, (11,))
        self.assertEqual(expanded.shape, (13,))
        self.assertEqual(expanded.dtype, np.float32)
        np.testing.assert_array_equal(expanded[:11], original)
        np.testing.assert_allclose(expanded[-2:], [0.5, -0.25])
        self.assertEqual(observation_order(11), OBSERVATION_ORDER)
        self.assertEqual(observation_order(13), OBSERVATION_ORDER_13)
        with self.assertRaises(ValueError):
            encode_observation(env, 12)

    def test_missing_food_displacement_is_zero(self):
        env = SnakeEnv(SnakeConfig(3, 3))
        env.food = None
        np.testing.assert_array_equal(encode_observation(env, 13)[-2:], [0, 0])

    def test_both_network_sizes_can_run_and_load(self):
        with tempfile.TemporaryDirectory() as path:
            for size in (11, 13):
                with self.subTest(input_size=size):
                    config = DQNConfig(network=NetworkConfig(input_size=size))
                    agent = DQNAgent(config)
                    output = Path(path) / f"policy_{size}.npz"
                    save_inference(output, agent)
                    model, loaded = load_inference(output)
                    self.assertEqual(loaded, config)
                    with np.load(output, allow_pickle=False) as data:
                        meta = json.loads(str(data["metadata"].item()))
                    self.assertEqual(meta["observation_order"], list(observation_order(size)))
                    x = np.zeros((1, size), dtype=np.float32)
                    np.testing.assert_array_equal(model.forward(x), agent.online.forward(x))
                    self.assertEqual(len(evaluate_policy(model, config.environment, [201])), 1)

    def test_v3_training_checkpoint_round_trip(self):
        with tempfile.TemporaryDirectory() as path:
            config = TrainingConfig(
                agent=DQNConfig(network=NetworkConfig(input_size=13)),
                episodes=2, evaluation_every=0, checkpoint_every=1,
            )
            trainer = Trainer(config)
            trainer.run(path)
            loaded_agent, state = load_training(Path(path) / "training.npz")
            self.assertEqual(state["config"]["agent"]["network"]["input_size"], 13)
            self.assertEqual(loaded_agent.environment_steps, trainer.agent.environment_steps)
            self.assertEqual(len(loaded_agent.replay), len(trainer.agent.replay))
            for name in trainer.agent.online.params:
                np.testing.assert_array_equal(
                    loaded_agent.online.params[name], trainer.agent.online.params[name]
                )
            resumed = Trainer.resume(Path(path) / "training.npz", episodes=3)
            self.assertEqual(len(resumed.records), 2)
            resumed.run(path)
            self.assertEqual(len(resumed.records), 3)

    def test_metadata_order_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as path:
            output = Path(path) / "policy.npz"
            agent = DQNAgent(DQNConfig(network=NetworkConfig(input_size=13)))
            save_inference(output, agent)
            with np.load(output, allow_pickle=False) as data:
                arrays = {key: data[key].copy() for key in data.files}
            meta = json.loads(str(arrays["metadata"].item()))
            meta["observation_order"] = list(OBSERVATION_ORDER)
            arrays["metadata"] = np.array(json.dumps(meta))
            np.savez_compressed(output, **arrays)
            with self.assertRaisesRegex(ValueError, "observation_order"):
                load_inference(output)


if __name__ == "__main__":
    unittest.main()
