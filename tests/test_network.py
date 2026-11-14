import unittest

import numpy as np

from tests.gradient_check import gradient_check
from network import MLP, NetworkConfig


class NetworkTests(unittest.TestCase):
    def model(self, seed=3, dtype=np.float32):
        return MLP(rng=np.random.default_rng(seed), dtype=dtype)

    def test_default_shapes_dtype_and_seed(self):
        first, second = self.model(), self.model()
        shapes = {"W1": (11, 32), "b1": (32,), "W2": (32, 3), "b2": (3,)}
        self.assertEqual(sum(p.size for p in first.params.values()), 483)
        for name, shape in shapes.items():
            self.assertEqual(first.params[name].shape, shape)
            self.assertEqual(first.params[name].dtype, np.float32)
            np.testing.assert_array_equal(first.params[name], second.params[name])
        self.assertFalse(np.array_equal(first.params["W1"], self.model(seed=4).params["W1"]))
        cache = first.activations(np.ones((4, 11), dtype=np.float32))
        self.assertEqual(cache["hidden"].shape, (4, 32))
        self.assertEqual(cache["q_values"].shape, (4, 3))
        self.assertTrue((cache["hidden"] >= 0).all())

    def test_hand_calculated_forward_gradients_and_update(self):
        model = MLP(NetworkConfig(1, 1, 1), rng=np.random.default_rng(0), dtype=np.float64)
        for name, value in {"W1": 0.5, "b1": -0.25, "W2": 2.0, "b2": -0.1}.items():
            model.params[name][...] = value
        x, actions, targets = [[2.0]], [0], [1.0]
        self.assertAlmostEqual(float(model.forward(x)[0, 0]), 1.4)
        loss, gradients = model.loss_and_gradients(x, actions, targets)
        self.assertAlmostEqual(loss, 0.16)
        for name, expected in {"W1": 3.2, "b1": 1.6, "W2": 0.6, "b2": 0.8}.items():
            np.testing.assert_allclose(gradients[name], expected)
        model.sgd_step(gradients, 0.01)
        self.assertAlmostEqual(float(model.forward(x)[0, 0]), 1.22798)

    def test_finite_differences_all_parameters(self):
        result = gradient_check()
        self.assertEqual(result["parameters_checked"], 21)
        self.assertLess(result["max_absolute_error"], 1e-7)
        self.assertLess(result["relative_l2_error"], 1e-6)
        self.assertLess(result["loss_after"], result["loss_before"])

    def test_only_selected_outputs_and_inactive_relu_receive_expected_gradient(self):
        model = MLP(NetworkConfig(1, 2, 3), rng=np.random.default_rng(0), dtype=np.float64)
        model.params["W1"][...] = [[1, -1]]
        model.params["b1"][...] = 0
        model.params["W2"][...] = 1
        model.params["b2"][...] = 0
        # h=[2,0], Q=[2,2,2]; selected output 1 has error 1, gradient 2.
        loss, gradients = model.loss_and_gradients([[2]], [1], [1])
        self.assertEqual(loss, 1)
        np.testing.assert_array_equal(gradients["W2"], [[0, 4, 0], [0, 0, 0]])
        np.testing.assert_array_equal(gradients["b2"], [0, 2, 0])
        np.testing.assert_array_equal(gradients["W1"], [[4, 0]])
        np.testing.assert_array_equal(gradients["b1"], [2, 0])

    def test_duplicate_batch_preserves_mean_loss_and_gradients(self):
        model = self.model(dtype=np.float64)
        x = np.random.default_rng(2).normal(size=(3, 11))
        actions, targets = np.array([0, 2, 1]), np.array([1.0, -1.0, 0.5])
        loss, grads = model.loss_and_gradients(x, actions, targets)
        repeated_loss, repeated_grads = model.loss_and_gradients(
            np.tile(x, (2, 1)), np.tile(actions, 2), np.tile(targets, 2))
        self.assertAlmostEqual(loss, repeated_loss)
        for name in grads:
            np.testing.assert_allclose(grads[name], repeated_grads[name], atol=1e-14)

    def test_float32_update_reduces_loss_without_changing_fixed_data(self):
        model = self.model()
        x = np.random.default_rng(4).integers(0, 2, size=(4, 11)).astype(np.float32)
        actions, targets = np.array([0, 1, 2, 0]), np.array([1, -1, 0.5, 0], dtype=np.float32)
        original = x.copy(), actions.copy(), targets.copy()
        params_before = {name: p.copy() for name, p in model.params.items()}
        loss, grads = model.loss_and_gradients(x, actions, targets)
        for name in grads:
            self.assertEqual(grads[name].shape, model.params[name].shape)
            self.assertEqual(grads[name].dtype, np.float32)
            self.assertTrue(np.isfinite(grads[name]).all())
            np.testing.assert_array_equal(model.params[name], params_before[name])
        model.sgd_step(grads, 0.001)
        self.assertLess(model.loss_and_gradients(x, actions, targets)[0], loss)
        for current, before in zip((x, actions, targets), original):
            np.testing.assert_array_equal(current, before)

    def test_bad_input_shapes_and_nonfinite_values(self):
        model = self.model()
        for x in (np.zeros(11), np.zeros((0, 11)), np.zeros((2, 10)), np.full((1, 11), np.nan)):
            with self.assertRaises(ValueError):
                model.forward(x)
        x = np.zeros((2, 11))
        for actions, targets in (([0], [1, 1]), ([0.0, 1.0], [1, 1]),
                                 ([True, False], [1, 1]), ([0, 3], [1, 1]),
                                 ([0, -1], [1, 1]), ([0, 1], [[1], [1]]),
                                 ([0, 1], [np.inf, 1])):
            with self.assertRaises(ValueError):
                model.loss_and_gradients(x, actions, targets)
        with self.assertRaises(ValueError):
            NetworkConfig(hidden_size=0)

    def test_invalid_update_is_atomic(self):
        model = self.model()
        before = {name: p.copy() for name, p in model.params.items()}
        grads = {name: np.ones_like(p) for name, p in model.params.items()}
        grads["b2"][0] = np.nan  # last parameter fails after other proposals
        with self.assertRaises(ValueError):
            model.sgd_step(grads, 0.01)
        for name in before:
            np.testing.assert_array_equal(model.params[name], before[name])
        for rate in (0, -1, np.nan, np.inf, True):
            with self.assertRaises(ValueError):
                model.sgd_step(grads, rate)


if __name__ == "__main__":
    unittest.main()
