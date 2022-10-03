# `InputReader` generator (returns None if no input reader is needed on
# the respective worker).
from email.mime import base
from functools import partial
from gc import callbacks
import logging
import os
import numpy as np
import ray
import tensorflow as tf
from ray.rllib.agents.callbacks import RE3UpdateCallbacks
from ray.rllib.agents.trainer import with_common_config
from ray.rllib.env import PolicyServerInput
from ray.tune import tune
from is3_broker_rl.model.normalize_reward_callback import WholesaleNormalizeRewardCallback
from ray.rllib.agents.callbacks import MultiCallbacks
from is3_broker_rl.model.wholesale_callbacks import MyCallbacks
from is3_broker_rl.model.wholesale_util import Env_config
from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.rllib.env import BaseEnv
from ray.rllib.evaluation import Episode, RolloutWorker
from ray.rllib.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch

from ray.rllib.utils.typing import AgentID, EnvType, PolicyID
from ray.rllib.utils.annotations import override
from typing import Dict, Tuple, Union
import argparse

SERVER_ADDRESS = "localhost"
SERVER_BASE_PORT = 9920
N_WORKERS = 0





def start_policy_server():
    log = logging.getLogger(__name__)
    log.info("start_policy_server")
    try:
        
        env_config = Env_config(SERVER_ADDRESS, SERVER_BASE_PORT, N_WORKERS, True)
        observation_space, action_space = env_config.get_gym_spaces()
        
        trainer_name = "SAC" # "SAC" or "TD3"
        enable_RE3_exploration = False
        config = env_config.get_rl_config(trainer_name)

        
        
        
        path_log = os.environ.get("DATA_DIR", "logs/")

        
        log.debug("Starting training loop ...")
        tune.run(
            trainer_name,
            config=config,
            #callbacks=[WholesaleNormalizeRewardCallback],
            stop=None,
            checkpoint_at_end=True,
            checkpoint_freq=500,
            verbose=3,
            local_dir=os.environ.get("DATA_DIR", "logs/"),
            log_to_file=True,
            name=f"{trainer_name}_28_09_discrete_1",
            resume="AUTO", # If the trial failed use restore="path_to_checkpoint" instead. 
            mode="max",
            max_failures = -1,
            #restore="logs/SAC_17_09_new_reward/SAC_None_9dd99_00000_0_2022-09-17_15-49-58/checkpoint_011000/checkpoint-11000"
            )
    except Exception as e:
        log.error(f"Cant create episode.  {e}", exc_info=True)
