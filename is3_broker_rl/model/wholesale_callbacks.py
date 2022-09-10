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
from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.rllib.env import BaseEnv
from ray.rllib.evaluation import Episode, RolloutWorker
from ray.rllib.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch

from ray.rllib.utils.typing import AgentID, EnvType, PolicyID
from ray.rllib.utils.annotations import override
from typing import Dict, Tuple, Union
import argparse

# See https://github.com/ray-project/ray/blob/master/rllib/examples/custom_metrics_and_callbacks.py
class MyCallbacks(DefaultCallbacks):


    def __init__(self, legacy_callbacks_dict: Dict[str, callable] = None):
        super().__init__(legacy_callbacks_dict)
        logging.basicConfig(level=logging.INFO, filename="stdout")
        #logging.basicConfig(level=logging.info)
        logging.info("Test3")
        logging.info("Test")
        self.last_reward = 0

    @override(DefaultCallbacks)
    def on_episode_start(
        self,
        *,
        worker: RolloutWorker,
        base_env: BaseEnv,
        policies: Dict[str, Policy],
        episode: Episode,
        env_index: int,
        **kwargs
    ):
        try:
            episode.user_data["env_reward"] = []
            episode.hist_data["env_reward"] = []
            logging.info("Start episode1")
            logging.info("Test_warn")
        except Exception as e:
            logging.info(f"Get Action error: {e}", exc_info=True)


    @override(DefaultCallbacks)
    def on_episode_step(
        self,
        *,
        worker: RolloutWorker,
        base_env: BaseEnv,
        policies: Dict[str, Policy],
        episode: Episode,
        env_index: int,
        **kwargs
    ):
        try:
            episode.user_data["env_reward"] = []
            episode.hist_data["env_reward"] = []
            logging.info("Start episode2")
            logging.info("Test_warn2")
        except Exception as e:
            logging.info(f"Get Action error: {e}", exc_info=True)

    @override(DefaultCallbacks)   
    def on_episode_end(
        self,
        *,
        worker: "RolloutWorker",
        base_env: BaseEnv,
        policies: Dict[PolicyID, Policy],
        episode: Episode,
        **kwargs,
    ) -> None:
        """Runs when an episode is done.
        Args:
            worker: Reference to the current rollout worker.
            base_env: BaseEnv running the episode. The underlying
                sub environment objects can be retrieved by calling
                `base_env.get_sub_environments()`.
            policies: Mapping of policy id to policy
                objects. In single agent mode there will only be a single
                "default_policy".
            episode: Episode object which contains episode
                state. You can use the `episode.user_data` dict to store
                temporary data, and `episode.custom_metrics` to store custom
                metrics for the episode.
            kwargs: Forward compatibility placeholder.
        """
        try:
            logging.info(f"Episode_end: hist reward: {episode.hist_data['env_reward']}")

            logging.info(f"Episode_end: user data reward: {episode.user_data['env_reward']}")
        
            logging.info(f"Episode_end: user data reward: {episode}")
            
        except Exception as e:
            logging.info(f"Get Action error: {e}", exc_info=True)

    @override(DefaultCallbacks)
    def on_sample_end(
        self, *, worker: "RolloutWorker", samples: SampleBatch, **kwargs
    ) -> None:
        """Called at the end of RolloutWorker.sample().
        Args:
            worker: Reference to the current rollout worker.
            samples: Batch to be returned. You can mutate this
                object to modify the samples generated.
            kwargs: Forward compatibility placeholder.
        """
        try:
            a= 1
            logging.info("on_sample_end: ")
        except Exception as e:
            logging.info(f"Get Action error: {e}", exc_info=True)

    @override(DefaultCallbacks)
    def on_postprocess_trajectory(
        self,
        *,
        worker: RolloutWorker,
        episode: Episode,
        agent_id: str,
        policy_id: str,
        policies: Dict[str, Policy],
        postprocessed_batch: SampleBatch,
        original_batches: Dict[str, Tuple[Policy, SampleBatch]],
        **kwargs
    ):
        
        try:
            #logging.info(f"episode {episode}")
            #logging.info("postprocessed {} ".format())
            # Do this on_episode_start to initialize the list.
            logging.info(f"on_postprocess_trajectory: last_reward reward: {self.last_reward}")
        #
            #logging.info(f"on_postprocess_trajectory: user data reward: {episode.user_data['env_reward']}")
        #
            logging.info(f"on_postprocess_trajectory: user data reward: {episode}")
            #
            logging.info(f"on_postprocess_trajectory: user data reward: {episode.last_action_for()}")

            # set the reward to the reward for the whole batch. 
            reward = postprocessed_batch["rewards"]
            new_reward = np.ones((len(reward))) *-1
            if len(reward) <= 100:
                if reward[len(reward)-1] != 0.0:
                    
                    i = 0
                    for value in reward:
                        new_reward[i] = reward[len(reward)-1]
                        i+=1
                    postprocessed_batch["rewards"] = new_reward
                    self.last_reward = reward[len(reward)-1]
                    episode.user_data["env_reward"].append(reward[len(reward)-1])
                    
                    episode.hist_data["env_reward"].append(reward[len(reward)-1])
                else:
                    logging.info(postprocessed_batch["rewards"])
                    postprocessed_batch["rewards"] = new_reward
            else:
                logging.info(f"len of reward not 24 instead: {len(reward)}")


            #batch = episode.new_batch_builder()
            #for each transition:
            #    batch.add_values(...)  # see sampler for usage
            #episode.extra_batches.add(batch.build_and_reset())
            env_reward = episode.user_data["env_reward"]
            return_reward = postprocessed_batch["rewards"]
            episode_info = episode.hist_data["env_reward"]
            #episode_info2 = episode.last_pi_info_for()
            episode_id = episode.episode_id
            logging.info(f"new_reward {return_reward}, , env_reward={env_reward}")
            logging.info(f"episode {episode_info}, , episode_id={episode_id}")
            
            # TODO: Here we could penalize actions more if they contribute to huge losses.
            #super().on_postprocess_trajectory(
            #        worker=worker,
            #        episode=episode,
            #        agent_id=agent_id,
            #        policy_id=policy_id,
            #        policies=policies,
            #        postprocessed_batch=postprocessed_batch,
            #        original_batches=original_batches,
            #        **kwargs
            #    )
        except Exception as e:
            logging.info(f"Callback error: {e}", exc_info=True)

    @override(DefaultCallbacks)
    def on_learn_on_batch(
        self, *, policy: Policy, train_batch: SampleBatch, result: dict, **kwargs
    ) -> None:
            try:
                a = 0
                #train_batch["rewards"]
                logging.info(
                    "policy.learn_on_batch() result: {} ->  rewards: {}".format(
                        policy, train_batch["rewards"]
                    )
                )
                #logging.info(
                #    "policy.learn_on_batch() result: {} ->  episode ids: {}, {}, {}".format(
                #        policy, train_batch["eps_id"], train_batch["infos"],  train_batch["actions"]
                #    )
                #    )
                #logging.info(
                #    "policy.learn_on_batch() result: {}".format(
                #        train_batch["infos"]
                #    )
                #    )
                #    #train_batch.
                    
                    
            except Exception as e:
                logging.info(f"Callback error: {e}", exc_info=True)