from typing import Any

import gym
import numpy as np
from ray.rllib.agents import with_common_config
from ray.rllib.env import PolicyServerInput
from ray.rllib.offline import IOContext

from is3_broker_rl.model.normalize_reward_callback import NormalizeRewardCallback

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


dqn_config = with_common_config({
    "env": None,
    # Use the `PolicyServerInput` to generate experiences.
    "input": _input,
    # timeslot, gridImbalance, ownImbalanceKwh, customerNetDemand, wholesalePrice, ownWholesalePrice, customerCount,
    # marketPosition
    "observation_space": gym.spaces.Box(
        low=np.array([360, np.finfo(np.float32).min, np.finfo(np.float32).min, np.finfo(np.float32).min, 0, 0, 0, 0]),
        high=np.array(
            [
                4000,
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
    "callbacks": NormalizeRewardCallback,
    # Use n worker processes to listen on different ports.
    "num_workers": N_WORKERS,
    # Disable off-policy-evaluation, since the rollouts are coming from online clients.
    "input_evaluation": [],
    # DL framework to use.
    "framework": "tf2",
    "eager_tracing": True,
    "log_level": "DEBUG",
    "timesteps_per_iteration": 20,
    "rollout_fragment_length": 8,
    "train_batch_size": 8,
    "lr": 1e-2,

    # DQN
    "replay_buffer_config": {"learning_starts": 0},
    "n_step": 1,
    "model": {
        "fcnet_hiddens": [64],
        "fcnet_activation": "relu",
    },

})
