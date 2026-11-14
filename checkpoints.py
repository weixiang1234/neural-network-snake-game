"""Versioned NPZ checkpoints, JSON metadata, no pickled objects."""

from dataclasses import asdict
import json
import os
from pathlib import Path
import platform
import tempfile

import numpy as np

from dqn import DQNAgent, DQNConfig, ReplayBatch
from network import MLP, NetworkConfig
from observations import observation_order
from snake_env import SnakeConfig


FORMAT_VERSION = 1
ACTION_MAPPING = ["straight", "right", "left"]
TASK_RULES = "snake-v1:food/win=10;collision/no-food=-10;other=0;no-food=terminal"


def decode_config(data):
    data = dict(data)
    data["network"] = NetworkConfig(**data["network"])
    data["environment"] = SnakeConfig(**data["environment"])
    return DQNConfig(**data)


def _write(path, agent, kind, state=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "format_version": FORMAT_VERSION, "kind": kind,
        "observation_order": list(observation_order(agent.config.network.input_size)), "action_mapping": ACTION_MAPPING,
        "task_rules": TASK_RULES, "dtype": "float32", "config": asdict(agent.config),
        "python_version": platform.python_version(), "numpy_version": np.__version__,
    }
    arrays = {f"online_{name}": p for name, p in agent.online.params.items()}
    if kind == "training":
        metadata.update({
            "environment_steps": agent.environment_steps, "updates": agent.updates,
            "action_rng": agent.action_rng.bit_generator.state,
            "replay_rng": agent.replay_rng.bit_generator.state,
            "replay_size": len(agent.replay), "replay_next": agent.replay._next,
            "optimizer": "plain_sgd_no_additional_state",
            "boundary": "between_episodes", "run_state": state,
        })
        arrays.update({f"target_{name}": p for name, p in agent.target.params.items()})
        # Only valid slots: never serialize uninitialized np.empty storage.
        arrays.update({f"replay_{name}": getattr(agent.replay, name)[:len(agent.replay)]
                       for name in ReplayBatch.__dataclass_fields__})
    arrays["metadata"] = np.array(json.dumps(metadata, allow_nan=False))
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False) as stream:
            temporary = Path(stream.name)
            np.savez_compressed(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def save_inference(path, agent):
    """Online weights and interpretation metadata; not resumable training."""
    _write(path, agent, "inference")


def save_training(path, agent, run_state):
    """Full agent plus runner state, saved only between completed episodes."""
    _write(path, agent, "training", run_state)


def _read(path, kind, expected_config):
    try:
        with np.load(path, allow_pickle=False) as data:
            metadata = json.loads(str(data["metadata"].item()))
            arrays = {name: data[name].copy() for name in data.files if name != "metadata"}
        config = decode_config(metadata["config"])
        for name, expected in (("format_version", FORMAT_VERSION), ("kind", kind),
                               ("observation_order", list(observation_order(config.network.input_size))),
                               ("action_mapping", ACTION_MAPPING), ("task_rules", TASK_RULES),
                               ("dtype", "float32")):
            if metadata.get(name) != expected:
                raise ValueError(f"incompatible checkpoint {name}")
        if expected_config is not None and config != expected_config:
            raise ValueError("checkpoint configuration does not match expected configuration")
        if kind == "training" and (metadata["python_version"] != platform.python_version()
                                   or metadata["numpy_version"] != np.__version__):
            raise ValueError("exact training resumption requires matching Python/NumPy versions")
        return metadata, config, arrays
    except (KeyError, TypeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid checkpoint: {error}") from error


def _restore_weights(model, arrays, prefix):
    for name, parameter in model.params.items():
        value = arrays.get(f"{prefix}_{name}")
        if value is None or value.shape != parameter.shape or value.dtype != parameter.dtype or not np.isfinite(value).all():
            raise ValueError(f"invalid checkpoint parameter {prefix}_{name}")
        parameter[...] = value


def load_inference(path, expected_config=None):
    _, config, arrays = _read(path, "inference", expected_config)
    model = MLP(config.network, rng=np.random.default_rng(0))
    _restore_weights(model, arrays, "online")
    return model, config


def load_training(path, expected_config=None):
    metadata, config, arrays = _read(path, "training", expected_config)
    if metadata.get("boundary") != "between_episodes" or metadata.get("optimizer") != "plain_sgd_no_additional_state":
        raise ValueError("unsupported training checkpoint state")
    agent = DQNAgent(config)
    _restore_weights(agent.online, arrays, "online")
    _restore_weights(agent.target, arrays, "target")
    size, position = metadata["replay_size"], metadata["replay_next"]
    if (type(size) is not int or type(position) is not int or not 0 <= size <= config.replay_capacity
            or not 0 <= position < config.replay_capacity or (size < config.replay_capacity and position != size)):
        raise ValueError("invalid replay counters")
    for name in ReplayBatch.__dataclass_fields__:
        destination = getattr(agent.replay, name)
        value = arrays.get(f"replay_{name}")
        if value is None or value.shape != destination[:size].shape or value.dtype != destination.dtype or not np.isfinite(value).all():
            raise ValueError(f"invalid replay array {name}")
        if name == "actions" and not np.isin(value, [0, 1, 2]).all():
            raise ValueError("invalid replay actions")
        destination[:size] = value
    agent.replay._size, agent.replay._next = size, position
    for name in ("environment_steps", "updates"):
        value = metadata[name]
        if type(value) is not int or value < 0:
            raise ValueError(f"invalid counter {name}")
        setattr(agent, name, value)
    agent.action_rng.bit_generator.state = metadata["action_rng"]
    agent.replay_rng.bit_generator.state = metadata["replay_rng"]
    return agent, metadata["run_state"]
