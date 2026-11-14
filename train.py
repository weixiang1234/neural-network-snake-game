"""Fast headless training with evaluation and episode-boundary resumption."""

import argparse
from dataclasses import asdict, dataclass, field, replace
import json
from numbers import Integral
from pathlib import Path
import signal

import numpy as np

from baseline import summarize
from checkpoints import decode_config, load_training, save_inference, save_training
from dqn import DQNAgent, DQNConfig
from evaluate import evaluate_policy, write_csv
from network import NetworkConfig
from observations import encode_observation
from reward_shaping import food_potential, shaped_reward
from snake_env import SnakeEnv


@dataclass(frozen=True)
class TrainingConfig:
    agent: DQNConfig = field(default_factory=DQNConfig)
    episodes: int = 500
    summary_every: int = 10
    evaluation_every: int = 50
    evaluation_episodes: int = 20
    evaluation_seed_start: int = 100_000
    checkpoint_every: int = 50
    rolling_window: int = 50
    reward_shaping_scale: float = 0.0

    def __post_init__(self):
        for name in ("episodes", "summary_every", "evaluation_every", "evaluation_episodes",
                     "evaluation_seed_start", "checkpoint_every", "rolling_window"):
            value = getattr(self, name)
            minimum = 0 if name in ("evaluation_every", "evaluation_seed_start") else 1
            if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
                raise ValueError(f"invalid {name}")
        if (isinstance(self.reward_shaping_scale, bool) or
                not np.isfinite(self.reward_shaping_scale) or self.reward_shaping_scale < 0):
            raise ValueError("reward_shaping_scale must be nonnegative and finite")


class Trainer:
    def __init__(self, config=None):
        self.config = config or TrainingConfig()
        self.agent = DQNAgent(self.config.agent)
        # This stream is never used by agent exploration, replay, or evaluation.
        self.environment_rng = np.random.default_rng(np.random.SeedSequence([self.config.agent.seed, 0x534E414B45]))
        self.env = SnakeEnv(self.config.agent.environment)
        self.records = []
        self.evaluations = []

    @property
    def rolling_column(self):
        return f"score_mean_last_{self.config.rolling_window}_episodes"

    def run_episode(self):
        seed = int(self.environment_rng.integers(0, 2**63 - 1))
        self.env.reset(seed=seed)
        episode_return = learning_return = loss_sum = 0.0
        loss_count = 0
        while True:
            size = self.config.agent.network.input_size
            observation = encode_observation(self.env, size)
            action = self.agent.act(observation).action
            scale = self.config.reward_shaping_scale
            previous_potential = food_potential(self.env) if scale else 0.0
            _, reward, terminated, truncated, info = self.env.step(action)
            learning_reward = (
                shaped_reward(reward, previous_potential, food_potential(self.env),
                              self.config.agent.gamma, scale)
                if scale else reward
            )
            result = self.agent.observe(observation, action, learning_reward,
                                        encode_observation(self.env, size), terminated, truncated)
            episode_return += reward
            learning_return += learning_reward
            if result is not None:
                loss_sum += result["loss"]
                loss_count += 1
            if terminated or truncated:
                break
        episode = len(self.records) + 1
        scores = [row["score"] for row in self.records[-(self.config.rolling_window - 1):]] if self.config.rolling_window > 1 else []
        scores.append(info["score"])
        row = {
            "episode": episode, "environment_steps": self.agent.environment_steps,
            "optimizer_updates": self.agent.updates, "env_seed": seed,
            "score": info["score"], "return": episode_return,
            "episode_length": info["steps"], "epsilon": self.agent.epsilon,
            "mean_loss": loss_sum / loss_count if loss_count else "",
            "ending_reason": info["ending_reason"], self.rolling_column: float(np.mean(scores)),
        }
        if self.config.reward_shaping_scale:
            row["learning_return"] = learning_return
        self.records.append(row)
        if self.config.evaluation_every and episode % self.config.evaluation_every == 0:
            c = self.config
            rows = evaluate_policy(self.agent.online, c.agent.environment,
                                   range(c.evaluation_seed_start, c.evaluation_seed_start + c.evaluation_episodes))
            for result_row in rows:
                self.evaluations.append({"training_episode": episode,
                                         "training_environment_steps": self.agent.environment_steps, **result_row})
        return row

    def save(self, directory):
        """Call only between episodes. Save canonical state before regenerable logs."""
        directory = Path(directory)
        state = {"config": asdict(self.config), "records": self.records, "evaluations": self.evaluations,
                 "environment_rng": self.environment_rng.bit_generator.state}
        save_training(directory / "training.npz", self.agent, state)
        save_inference(directory / "policy.npz", self.agent)
        training_fields = ["episode", "environment_steps", "optimizer_updates", "env_seed", "score", "return",
                           "episode_length", "epsilon", "mean_loss", "ending_reason", self.rolling_column]
        evaluation_fields = ["training_episode", "training_environment_steps", "env_seed", "score", "return",
                             "episode_length", "ending_reason"]
        if self.config.reward_shaping_scale:
            training_fields.append("learning_return")
        write_csv(directory / "training.csv", self.records, training_fields)
        write_csv(directory / "evaluation.csv", self.evaluations, evaluation_fields)
        (directory / "config.json").write_text(json.dumps(asdict(self.config), indent=2) + "\n")

    @classmethod
    def resume(cls, path, episodes=None):
        agent, state = load_training(path)
        settings = dict(state["config"])
        settings["agent"] = decode_config(settings["agent"])
        if settings["agent"] != agent.config:
            raise ValueError("runner and agent configurations differ")
        if episodes is not None:
            settings["episodes"] = episodes
        trainer = cls(TrainingConfig(**settings))
        trainer.agent = agent
        trainer.environment_rng.bit_generator.state = state["environment_rng"]
        trainer.records = state["records"]
        trainer.evaluations = state["evaluations"]
        if trainer.records and (trainer.records[-1]["episode"] != len(trainer.records)
                                or trainer.records[-1]["environment_steps"] != agent.environment_steps):
            raise ValueError("inconsistent episode history")
        return trainer

    def run(self, directory, stop_requested=lambda: False):
        self.save(directory)  # a recoverable boundary exists even before episode 1
        while len(self.records) < self.config.episodes and not stop_requested():
            row = self.run_episode()
            episode = row["episode"]
            if episode % self.config.summary_every == 0 or episode == self.config.episodes:
                loss = row["mean_loss"]
                loss_label = f"{loss:.4f}" if loss != "" else "not updated"
                print(f"episode={episode} steps={row['environment_steps']} score={row['score']} "
                      f"{self.rolling_column}={row[self.rolling_column]:.3f} "
                      f"epsilon={row['epsilon']:.4f} mean_loss={loss_label} end={row['ending_reason']}", flush=True)
            if self.config.evaluation_every and episode % self.config.evaluation_every == 0:
                summary = summarize(self.evaluations[-self.config.evaluation_episodes:])
                print(f"greedy evaluation at episode {episode}: {json.dumps(summary)}", flush=True)
            if episode % self.config.checkpoint_every == 0:
                self.save(directory)
        self.save(directory)
        return self


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, help="total episode target, including completed episodes on resume")
    parser.add_argument("--seed", type=int, help="new-run seed; cannot override on resume")
    parser.add_argument("--gamma", type=float, help="new-run discount factor (default: 0.95)")
    parser.add_argument("--reward-shaping-scale", type=float,
                        help="potential-based food-proximity shaping; default: 0 (disabled)")
    parser.add_argument("--observation-size", type=int, choices=(11, 13), default=None,
                        help="new-run state features: 11 (original) or 13 (food displacement)")
    parser.add_argument("--output", type=Path, default=Path("results/training"))
    parser.add_argument("--resume", type=Path, help="full training.npz, not policy.npz")
    parser.add_argument("--eval-every", type=int, help="new-run evaluation interval; 0 disables")
    parser.add_argument("--eval-episodes", type=int, help="new-run episodes per evaluation")
    parser.add_argument("--checkpoint-every", type=int, help="new-run checkpoint interval")
    args = parser.parse_args()
    if args.resume and any(value is not None for value in (args.seed, args.gamma, args.observation_size, args.reward_shaping_scale, args.eval_every, args.eval_episodes, args.checkpoint_every)):
        parser.error("resume restores saved settings; only --episodes and --output may change")
    if args.output.exists() and any(args.output.iterdir()):
        if not args.resume or args.resume.resolve() != (args.output / "training.npz").resolve():
            parser.error("output is nonempty; choose a new directory or resume its training.npz")
    try:
        if args.resume:
            trainer = Trainer.resume(args.resume, episodes=args.episodes)
            if trainer.config.episodes < len(trainer.records):
                parser.error("episode target cannot be below already completed episodes")
        else:
            agent_kwargs = {"seed": args.seed if args.seed is not None else 0}
            if args.gamma is not None:
                agent_kwargs["gamma"] = args.gamma
            if args.observation_size is not None:
                agent_kwargs["network"] = NetworkConfig(input_size=args.observation_size)
            agent = DQNConfig(**agent_kwargs)
            overrides = {name: value for name, value in {
                "episodes": args.episodes, "evaluation_every": args.eval_every,
                "evaluation_episodes": args.eval_episodes, "checkpoint_every": args.checkpoint_every,
                "reward_shaping_scale": args.reward_shaping_scale,
            }.items() if value is not None}
            trainer = Trainer(TrainingConfig(agent=agent, **overrides))
    except (ValueError, OSError) as error:
        parser.error(str(error))
    stop = {"requested": False}

    def request_stop(signum, frame):
        if stop["requested"]:
            raise KeyboardInterrupt
        stop["requested"] = True
        print("Stop requested: finishing this episode, then saving. A second interrupt exits immediately.", flush=True)

    previous = signal.signal(signal.SIGINT, request_stop)
    try:
        trainer.run(args.output, lambda: stop["requested"])
        print(f"Saved {len(trainer.records)} completed episodes to {args.output}")
    except KeyboardInterrupt:
        print("Exited immediately; resume the last saved episode boundary.")
    finally:
        signal.signal(signal.SIGINT, previous)


if __name__ == "__main__":
    main()
