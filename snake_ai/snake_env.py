"""Deterministic Snake rules. No rendering or learning dependencies."""

from dataclasses import dataclass
from numbers import Integral
import random


# Clockwise order makes relative turns simple: right, down, left, up.
DIRECTIONS = ((1, 0), (0, 1), (-1, 0), (0, -1))
ACTION_TURNS = (0, 1, -1)  # straight, right, left


@dataclass(frozen=True)
class SnakeConfig:
    width: int = 20
    height: int = 20
    max_steps_without_food: int | None = None  # None means width * height.

    def __post_init__(self):
        for name in ("width", "height"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral) or value < 3:
                raise ValueError(f"{name} must be an integer >= 3")
        limit = self.max_steps_without_food
        if limit is not None and (
            isinstance(limit, bool) or not isinstance(limit, Integral) or limit < 1
        ):
            raise ValueError("max_steps_without_food must be a positive integer")

    @property
    def food_limit(self):
        return self.max_steps_without_food or self.width * self.height


@dataclass(frozen=True)
class BoardObservation:
    """Raw state snapshot; observations.encode_observation supplies 11 floats.

    Coordinates use x rightwards and y downwards; snake is head first.
    direction indexes DIRECTIONS. Include the hunger clock in this raw state
    because it affects whether the next move ends the task.
    """

    snake: tuple[tuple[int, int], ...]
    food: tuple[int, int] | None
    direction: int
    steps_without_food: int


class SnakeEnv:
    def __init__(self, config: SnakeConfig | None = None):
        self.config = config or SnakeConfig()
        self.rng = random.Random()
        self.reset()

    def reset(self, seed=None):
        """Return (raw observation, info). An explicit seed restarts the RNG."""
        if seed is not None:
            self.rng.seed(seed)
        x, y = max(2, self.config.width // 2), self.config.height // 2
        self.snake = [(x, y), (x - 1, y), (x - 2, y)]
        self.direction = 0
        self.score = self.steps = self.steps_without_food = 0
        self.ending_reason = None
        self.food = self._spawn_food()
        return self._observation(), self._info()

    def _spawn_food(self):
        occupied = set(self.snake)
        free = [
            (x, y)
            for y in range(self.config.height)
            for x in range(self.config.width)
            if (x, y) not in occupied
        ]
        return self.rng.choice(free) if free else None

    def _project_move(self, action):
        if isinstance(action, bool) or not isinstance(action, Integral) or action not in (0, 1, 2):
            raise ValueError("action must be 0 (straight), 1 (right), or 2 (left)")
        direction = (self.direction + ACTION_TURNS[action]) % 4
        dx, dy = DIRECTIONS[direction]
        x, y = self.snake[0]
        return direction, (x + dx, y + dy)

    def _collision_reason(self, head, growing):
        x, y = head
        if not (0 <= x < self.config.width and 0 <= y < self.config.height):
            return "wall"
        # On a non-growing step the old tail leaves before the new head enters.
        occupied = self.snake if growing else self.snake[:-1]
        return "body" if head in occupied else None

    def would_collide(self, action):
        """Use exactly the same prediction as step (for later danger features)."""
        _, head = self._project_move(action)
        return self._collision_reason(head, head == self.food) is not None

    def step(self, action):
        """Return (observation, reward, terminated, truncated, info).

        +10 food (including a full-board win), -10 collision or no-food limit,
        0 otherwise. The no-food limit is task termination, never truncation.
        Collision takes priority; food resets the clock even on the limit step.
        """
        if self.ending_reason is not None:
            raise RuntimeError("Episode ended; call reset before step")
        direction, head = self._project_move(action)
        growing = head == self.food
        self.direction = direction
        self.steps += 1
        self.steps_without_food += 1
        self.ending_reason = self._collision_reason(head, growing)
        reward = 0.0
        if self.ending_reason is not None:
            # Keep the last valid body snapshot after a collision.
            reward = -10.0
        else:
            self.snake.insert(0, head)
            if growing:
                self.score += 1
                self.steps_without_food = 0
                reward = 10.0
                self.food = self._spawn_food()
                if self.food is None:
                    self.ending_reason = "win"
            else:
                self.snake.pop()
                if self.steps_without_food >= self.config.food_limit:
                    self.ending_reason = "no_food_limit"
                    reward = -10.0
        return self._observation(), reward, self.ending_reason is not None, False, self._info()

    def _observation(self):
        return BoardObservation(tuple(self.snake), self.food, self.direction, self.steps_without_food)

    def _info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "steps_without_food": self.steps_without_food,
            "ending_reason": self.ending_reason,
        }
