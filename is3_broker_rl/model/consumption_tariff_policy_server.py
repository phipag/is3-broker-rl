# `InputReader` generator (returns None if no input reader is needed on
# the respective worker).
import logging
import os
from typing import Any

import gym
import numpy as np
from ray.rllib.agents import DefaultCallbacks
from ray.rllib.env import PolicyServerInput
from ray.rllib.offline import IOContext
from ray.tune import tune

SERVER_ADDRESS = "localhost"
SERVER_BASE_PORT = 9900
N_WORKERS = 0


def _input(ioctx: IOContext) -> Any:
    if ioctx.worker_index > 0 or ioctx.worker and ioctx.worker.num_workers == 0:
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
        # gridImbalance, ownBalancingCosts, customerNetDemand, wholesalePrice, ownWholesalePrice, customerCount,
        # marketPosition
        "observation_space": gym.spaces.Box(
            low=np.array([np.finfo(np.float32).min, np.finfo(np.float32).min, np.finfo(np.float32).min, 0, 0, 0, 0]),
            high=np.array(
                [
                    np.finfo(np.float32).max,
                    np.finfo(np.float32).max,
                    np.finfo(np.float32).max,
                    1000,  # Typical wholesale buying prices are maximum 40-50 euro/MWh, and so we set a generous limit
                    1000,  # Typical wholesale buying prices are maximum 40-50 euro/MWh, and so we set a generous limit
                    1e6,
                    4,
                ]
            ),
            dtype=np.float32,
        ),
        "observation_filter": "MeanStdFilter",
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
            # Set this to a value larger or equal to the get_action calls per episode
            # (this makes sure that the episode_reward etc. is reported in tensorboard).
            # See org.powertac.is3broker.tariff.consumption.offerHorizon
            #     and org.powertac.is3broker.tariff.consumption.episodeLength
            "timesteps_per_iteration": 36,
            "train_batch_size": 16,
            # 1-step Q-Learning
            "n_step": 1,
        }
    )
    config["model"] = {
        "fcnet_hiddens": [64],
        "fcnet_activation": "relu",
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
        local_dir=os.environ.get("DATA_DIR", "logs/"),
        log_to_file=True,
        name="DQN_Consumption_Trial2",
        resume="AUTO",  # Will load the latest checkpoint from the local experiment directory or start a new one
    )
