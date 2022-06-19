# `InputReader` generator (returns None if no input reader is needed on
# the respective worker).
import logging
import os
from typing import Optional

import gym
import numpy as np
from ray.rllib.agents import DefaultCallbacks
from ray.rllib.env import PolicyServerInput
from ray.rllib.offline import IOContext
from ray.tune import tune

SERVER_ADDRESS = "localhost"
SERVER_BASE_PORT = 9900
N_WORKERS = 0


def _input(ioctx: IOContext) -> Optional[PolicyServerInput]:
    if ioctx.worker_index > 0 or ioctx.worker.num_workers == 0:
        return PolicyServerInput(
            ioctx,
            SERVER_ADDRESS,
            SERVER_BASE_PORT + ioctx.worker_index - (1 if ioctx.worker_index > 0 else 0),
        )
    else:
        return None


def start_policy_server() -> None:
    config = {
        "env": None,
        # gridImbalance, ownBalancingCosts, customerNetDemand, wholesalePrice, customerCount, marketPosition
        "observation_space": gym.spaces.Box(
            low=np.array([np.finfo(np.float32).min, np.finfo(np.float32).min, np.finfo(np.float32).min, 0, 0, 0]),
            high=np.array(
                [
                    np.finfo(np.float32).max,
                    np.finfo(np.float32).max,
                    np.finfo(np.float32).max,
                    np.finfo(np.float32).max,
                    1e6,
                    4,
                ]
            ),
            dtype=np.float32,
        ),
        "action_space": gym.spaces.Discrete(5),
        # Use the `PolicyServerInput` to generate experiences.
        "input": _input,
        "callbacks": DefaultCallbacks,
        # Use n worker processes to listen on different ports.
        "num_workers": N_WORKERS,
        # Disable off-policy-evaluation, since the rollouts are coming from online clients.
        "input_evaluation": [],
        # DL framework to use.
        "framework": "tf2",
        "log_level": "DEBUG",
    }

    # For DQN
    config.update(
        {
            # Start learning immediately
            "learning_starts": 0,
            # In combination with checkpoint_freq=1 this will create a checkpoint every 2 timesteps
            "timesteps_per_iteration": 2,
            "train_batch_size": 2,
            # 1-step Q-Learning
            "n_step": 1,
        }
    )
    config["model"] = {
        "fcnet_hiddens": [64],
        "fcnet_activation": "linear",
    }

    log = logging.getLogger(__name__)
    log.debug("Starting training loop ...")
    tune.run(
        "DQN",
        config=config,
        stop=None,
        checkpoint_at_end=True,
        checkpoint_freq=1,
        verbose=2,
        local_dir=os.environ.get("LOG_DIR", "logs/"),
        log_to_file=True,
        name="DQN_Consumption_Trial1",
    )
