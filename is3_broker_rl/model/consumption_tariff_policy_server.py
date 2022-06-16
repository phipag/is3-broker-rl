# `InputReader` generator (returns None if no input reader is needed on
# the respective worker).
import logging

import gym
import numpy as np
from ray.rllib.env import PolicyServerInput
from ray.rllib.examples.custom_metrics_and_callbacks import MyCallbacks
from ray.tune import tune

from is3_broker_rl.utils import get_root_path

SERVER_ADDRESS = "localhost"
SERVER_BASE_PORT = 9900
CHECKPOINT_FILE = "last_checkpoint_{}.out"
N_WORKERS = 0


def _input(ioctx):
    # We are remote worker or we are local worker with num_workers=0:
    # Create a PolicyServerInput.
    if ioctx.worker_index > 0 or ioctx.worker.num_workers == 0:
        return PolicyServerInput(
            ioctx,
            SERVER_ADDRESS,
            SERVER_BASE_PORT + ioctx.worker_index - (1 if ioctx.worker_index > 0 else 0),
        )
    # No InputReader (PolicyServerInput) needed.
    else:
        return None


def start_policy_server():
    config = {
        "env": None,
        # For some reason, the tuple observation space crashes the PolicyServer when calling get_action the second time
        # with the PolicyClient.
        # "observation_space": gym.spaces.Tuple(
        #     [
        #         # gridImbalance, ownImbalance, customerNetDemand
        #         gym.spaces.Box(
        #             low=np.array([np.finfo(np.float32).min, np.finfo(np.float32).min, np.finfo(np.float32).min]),
        #             high=np.array([np.finfo(np.float32).max, np.finfo(np.float32).max, np.finfo(np.float32).max]),
        #             dtype=np.float32,
        #         ),
        #         # customerCount, marketPosition
        #         gym.spaces.Box(
        #             low=np.array([0, 0]),
        #             high=np.array([1e6, 4]),
        #             dtype=np.int32
        #         )
        #     ]
        # ),

        # gridImbalance, ownImbalance, customerNetDemand, customerCount, marketPosition
        "observation_space": gym.spaces.Box(
            low=np.array([np.finfo(np.float32).min, np.finfo(np.float32).min, np.finfo(np.float32).min, 0, 0]),
            high=np.array([np.finfo(np.float32).max, np.finfo(np.float32).max, np.finfo(np.float32).max, 1e6, 4]),
            dtype=np.float32,
        ),
        "action_space": gym.spaces.Discrete(5),
        # Use the `PolicyServerInput` to generate experiences.
        "input": _input,
        # Use n worker processes to listen on different ports.
        "num_workers": N_WORKERS,
        # Disable OPE, since the rollouts are coming from online clients.
        "input_evaluation": [],
        # Create a "chatty" client/server or not.
        "callbacks": MyCallbacks,
        # DL framework to use.
        "framework": "tf2",
        # Set to INFO so we'll see the server's actual address:port.
        "log_level": "DEBUG",
    }

    # For DQN
    config.update(
        {
            "learning_starts": 100,
            "timesteps_per_iteration": 200,
            "n_step": 3,
        }
    )
    config["model"] = {
        "fcnet_hiddens": [64],
        "fcnet_activation": "linear",
    }

    # Checkpoint path
    checkpoint_path = CHECKPOINT_FILE.format("DQN")

    log = logging.getLogger(__name__)
    log.debug("Starting training loop ...")
    tune.run(
        "DQN",
        config=config,
        stop=None,
        verbose=2,
        local_dir=str(get_root_path() / "logs"),
        log_to_file=True,
        name="DQN_Test",
    )
