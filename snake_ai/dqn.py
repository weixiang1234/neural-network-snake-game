"""One-step DQN components. Training runs and checkpoints come in Milestone 5."""

from dataclasses import dataclass, field
from numbers import Integral

import numpy as np

from .network import MLP, NetworkConfig
from .snake_env import SnakeConfig


@dataclass(frozen=True)
class DQNConfig:
    network: NetworkConfig = field(default_factory=NetworkConfig)
    environment: SnakeConfig = field(default_factory=SnakeConfig)
    seed: int = 0
    gamma: float = 0.95
    learning_rate: float = 0.001
    replay_capacity: int = 10_000
    batch_size: int = 64
    warmup_steps: int = 1_000
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 50_000
    target_sync_updates: int = 250

    def __post_init__(self):
        for name in ("replay_capacity", "batch_size", "warmup_steps", "epsilon_decay_steps", "target_sync_updates"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if isinstance(self.seed, bool) or not isinstance(self.seed, Integral) or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        if self.batch_size > self.replay_capacity:
            raise ValueError("batch_size must fit in replay_capacity")
        if not np.isfinite(self.gamma) or not 0 <= self.gamma <= 1:
            raise ValueError("gamma must be in [0, 1]")
        if not np.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive and finite")
        if not (np.isfinite(self.epsilon_start) and np.isfinite(self.epsilon_end)
                and 0 <= self.epsilon_end <= self.epsilon_start <= 1):
            raise ValueError("require 0 <= epsilon_end <= epsilon_start <= 1")
        if self.network.input_size not in (11, 13) or self.network.output_size != 3:
            raise ValueError("Snake DQN requires 11 or 13 inputs and 3 outputs")


@dataclass(frozen=True)
class ReplayBatch:
    observations: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    next_observations: np.ndarray
    terminated: np.ndarray
    truncated: np.ndarray


class ReplayBuffer:
    def __init__(self, capacity, observation_size=11):
        for value in (capacity, observation_size):
            if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
                raise ValueError("capacity and observation_size must be positive integers")
        self.capacity = capacity
        self.observation_size = observation_size
        self.observations = np.empty((capacity, observation_size), dtype=np.float32)
        self.next_observations = np.empty_like(self.observations)
        self.actions = np.empty(capacity, dtype=np.int64)
        self.rewards = np.empty(capacity, dtype=np.float32)
        self.terminated = np.empty(capacity, dtype=bool)
        self.truncated = np.empty(capacity, dtype=bool)
        self._size = self._next = 0

    def __len__(self):
        return self._size

    def add(self, observation, action, reward, next_observation, terminated, truncated):
        current = np.asarray(observation, dtype=np.float32)
        following = np.asarray(next_observation, dtype=np.float32)
        for vector in (current, following):
            if vector.shape != (self.observation_size,) or not np.isfinite(vector).all():
                raise ValueError("replay observations must be finite vectors of the configured size")
        if isinstance(action, bool) or not isinstance(action, Integral) or action not in (0, 1, 2):
            raise ValueError("action must be 0, 1, or 2")
        reward = np.asarray(reward, dtype=np.float32)
        if reward.shape != () or not np.isfinite(reward):
            raise ValueError("reward must be a finite scalar")
        if not all(isinstance(flag, (bool, np.bool_)) for flag in (terminated, truncated)):
            raise ValueError("episode flags must be booleans")
        i = self._next
        self.observations[i], self.next_observations[i] = current, following
        self.actions[i], self.rewards[i] = action, reward
        self.terminated[i], self.truncated[i] = terminated, truncated
        self._next = (i + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size, rng):
        if (isinstance(batch_size, bool) or not isinstance(batch_size, Integral)
                or not 1 <= batch_size <= len(self)):
            raise ValueError("batch_size must be between 1 and replay size")
        indices = rng.choice(len(self), size=batch_size, replace=False)
        # Advanced indexing returns independent arrays, never views into replay.
        return ReplayBatch(*(getattr(self, name)[indices] for name in ReplayBatch.__dataclass_fields__))


@dataclass(frozen=True)
class ActionDecision:
    action: int
    greedy_action: int
    exploratory: bool
    epsilon: float
    q_values: np.ndarray


class DQNAgent:
    def __init__(self, config=None):
        self.config = config or DQNConfig()
        streams = np.random.SeedSequence(self.config.seed).spawn(4)
        self.online = MLP(self.config.network, rng=np.random.default_rng(streams[0]))
        self.target = MLP(self.config.network, rng=np.random.default_rng(streams[1]))
        self.action_rng = np.random.default_rng(streams[2])
        self.replay_rng = np.random.default_rng(streams[3])
        self.sync_target()  # independent allocation, identical starting values
        self.replay = ReplayBuffer(self.config.replay_capacity, self.config.network.input_size)
        self.environment_steps = self.updates = 0

    @property
    def epsilon(self):
        c = self.config
        fraction = min(self.environment_steps / c.epsilon_decay_steps, 1.0)
        return c.epsilon_start + fraction * (c.epsilon_end - c.epsilon_start)

    def act(self, observation, *, explore=True):
        q = self.online.forward(np.asarray(observation)[None, :])[0]
        greedy = int(np.argmax(q))
        epsilon = self.epsilon if explore else 0.0
        exploratory = bool(explore and self.action_rng.random() < epsilon)
        action = int(self.action_rng.integers(3)) if exploratory else greedy
        return ActionDecision(action, greedy, exploratory, epsilon, q)

    def sync_target(self):
        for name, parameter in self.online.params.items():
            self.target.params[name][...] = parameter

    def td_targets(self, batch):
        """Reward only at true termination; bootstrap all other transitions.

        Truncation is an external interruption, not a terminal task state. Its
        next_observation must be the final observation BEFORE any reset.
        The Snake no-food rule is termination, not truncation.
        """
        targets = batch.rewards.copy()
        continuing = ~batch.terminated
        if continuing.any():
            next_q = self.target.forward(batch.next_observations[continuing])
            targets[continuing] += self.config.gamma * next_q.max(axis=1)
        return targets

    def observe(self, observation, action, reward, next_observation, terminated, truncated):
        """Store one environment transition and attempt one minibatch update."""
        self.replay.add(observation, action, reward, next_observation, terminated, truncated)
        self.environment_steps += 1
        return self.learn()

    def learn(self):
        """No updates until warm-up and enough replay samples; sample uniformly."""
        c = self.config
        if self.environment_steps < c.warmup_steps or len(self.replay) < c.batch_size:
            return None
        batch = self.replay.sample(c.batch_size, self.replay_rng)
        targets = self.td_targets(batch)
        predicted = self.online.forward(batch.observations)[np.arange(c.batch_size), batch.actions]
        loss, gradients = self.online.loss_and_gradients(batch.observations, batch.actions, targets)
        self.online.sgd_step(gradients, c.learning_rate)
        self.updates += 1
        synchronized = self.updates % c.target_sync_updates == 0
        if synchronized:
            self.sync_target()
        return {
            "loss": loss,
            "predictions": predicted,
            "targets": targets,
            "b2_gradient": gradients["b2"],
            "gradient_l2": float(np.sqrt(sum(np.sum(g.astype(np.float64) ** 2) for g in gradients.values()))),
            "update": self.updates,
            "target_synchronized": synchronized,
        }
