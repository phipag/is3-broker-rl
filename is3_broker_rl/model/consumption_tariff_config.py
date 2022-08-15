from typing import Any

import gym
import numpy as np
from ray.rllib.agents import with_common_config
from ray.rllib.env import PolicyServerInput
from ray.rllib.offline import IOContext

from is3_broker_rl.model.normalize_reward_callback import (
    ConsumptionNormalizeRewardCallback,
)

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


dqn_config = with_common_config(
    {
        "env": None,
        # Use the `PolicyServerInput` to generate experiences.
        "input": _input,
        "observation_space": gym.spaces.Box(
            low=np.array(
                [
                    360,  # timeslot
                    np.finfo(np.float32).min,  # gridImbalance
                    np.finfo(np.float32).min,  # ownImbalanceKwh
                    np.finfo(np.float32).min,  # customerNetDemand
                    0,  # wholesalePrice
                    0,  # ownWholesalePrice
                    np.finfo(np.float32).min,  # cashPosition
                    0,  # consumptionShare
                    0,  # productionShare
                    0,  # marketPosition
                ]
            ),
            high=np.array(
                [
                    4000,  # timeslot
                    np.finfo(np.float32).max,  # gridImbalance
                    np.finfo(np.float32).max,  # ownImbalanceKwh
                    np.finfo(np.float32).max,  # customerNetDemand
                    # wholesalePrice: Typical wholesale buying prices are maximum 40-50 euro/MWh, and so we set a
                    # generous limit
                    1000,
                    # ownWholesalePrice: Typical wholesale buying prices are maximum 40-50 euro/MWh, and so we set a
                    # generous limit
                    1000,
                    np.finfo(np.float32).max,  # cashPosition
                    1,  # consumptionShare
                    1,  # productionShare
                    4,  # marketPosition
                ]
            ),
            dtype=np.float32,
        ),
        # Normalize the observations
        "observation_filter": "MeanStdFilter",
        "action_space": gym.spaces.Discrete(5),
        # Normalize the rewards
        "callbacks": ConsumptionNormalizeRewardCallback,
        # Use n worker processes to listen on different ports.
        "num_workers": N_WORKERS,
        # Disable off-policy-evaluation, since the rollouts are coming from online clients.
        "input_evaluation": [],
        "framework": "tf2",
        "eager_tracing": True,
        "log_level": "DEBUG",
        "timesteps_per_iteration": 64,
        "rollout_fragment_length": 16,
        "train_batch_size": 16,
        "lr": 1e-2,
        # Discount factor for future reward (default value is 0.99)
        "gamma": 0.99,
        "explore": True,
        "exploration_config": {
            "type": "EpsilonGreedy",
            "initial_epsilon": 1.0,
            "final_epsilon": 0.02,
            "epsilon_timesteps": 1500,
        },
        # DQN
        "replay_buffer_config": {
            "type": "MultiAgentPrioritizedReplayBuffer",
            # Wait 500 steps before starting learning
            "learning_starts": 500,
        },
        "store_buffer_in_checkpoints": True,
        # The Java broker uses an episode length of 168 and gets a new action every 14 timeslots.
        # 168 / 14 = 12 timesteps will make sure that the capacity costs (every 168 timeslots) are associated
        # to the last 12 taken actions taken.
        "n_step": 12,
        "model": {
            "fcnet_hiddens": [64],
            "fcnet_activation": "relu",
        },
    }
)
