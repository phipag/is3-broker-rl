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
            
            episode_data = episode.last_info_for()
            
            user_data = episode.user_data
            #logging.info(f"user_data on step: {user_data}, id: {episode.episode_id}, episode_info {episode_data}")
            agent_rewards = episode.agent_rewards
            last_obs = episode.last_observation_for()
            
            #logging.info(f"Agent rewards on step: {agent_rewards}, env_index: {env_index}, episode_length: {episode.length},last_obs: {last_obs}")


            

        except Exception as e:
            logging.info(f"Get Action error: {e}", exc_info=True)

    @override(DefaultCallbacks)   
    def on_episode_end(
        self,
        *,
        worker: RolloutWorker,
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
            #logging.info(f"Episode_end: hist reward: {episode.hist_data['env_reward']}")

            #logging.info(f"Episode_end: user data reward: {episode.user_data['env_reward']}")
        
            #logging.info(f"Episode_end: user data reward: {episode}")
            a=0
            
        except Exception as e:
            logging.info(f"Get Action error: {e}", exc_info=True)

    @override(DefaultCallbacks)
    def on_sample_end(self, *, worker: RolloutWorker, samples: SampleBatch, **kwargs):
        try:
            logging.info("returned sample batch of size {}".format(samples.count))
        except Exception as e:
            logging.info(f"On_sample_end error: {e}", exc_info=True)

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
            logging.info(f"post_processed_next_obs {postprocessed_batch}")
            #logging.info(f" pp_b reward shape: {len(postprocessed_batch['rewards'])}")
            logging.info(f"post_processed_next_obs {np.shape(postprocessed_batch['new_obs'])}")
            #logging.info(f"on_postprocess_trajectory: last_reward reward: {self.last_reward}")
        #
            logging.info(f"on_postprocess_trajectory: user data reward: {episode.user_data['env_reward']}")
        #
            logging.info(f"on_postprocess_trajectory: user data reward: {episode.last_info_for()}")
            #
            
            logging.info(f"on_postprocess_trajectory: user data last action: {episode.last_action_for()}")

            info_dict = episode.last_info_for()
            reward = postprocessed_batch["rewards"]
            new_reward = np.zeros((len(reward))) *-1
            
            #new_reward = np.ones((len(postprocessed_batch))) *-1
            # At first only filter really bad rewards.
            if info_dict["reward"] < -0.1:
                # Check in what direction we are unbalanced.
                if info_dict["sum_mWh"] < info_dict["reward_market_balance"]:
                    imbalance_bought_too_much = 1

                else:
                    imbalance_bought_too_much = -1

                actions = postprocessed_batch['actions']
                #logging.info(f"obs: {postprocessed_batch['obs']}")
                
                # Go through every action
                for time_i in range(len(postprocessed_batch)):
                    # actions are sorted the other way around from our logic.
                    action_i = len(postprocessed_batch) - time_i - 1
                    logging.info(f"Actions in post_batch: {actions[23-time_i]}")
                    # Is actions reversed? Oldest actions on which side?
                    # check all actions where the market_balance is less than the needed value and the 
                    # total imbalance is negative. Thus the action we select is contributing to this imbalance.
                    logging.info(f"market_balance: {info_dict['market_balance'][time_i]},sum_mWh {info_dict['sum_mWh']},imbalance bool: {imbalance_bought_too_much}")
                    if ((info_dict["market_balance"][time_i] < info_dict["sum_mWh"]) & (imbalance_bought_too_much == -1) 
                            & (actions[action_i][0] < 0) & (actions[action_i][2]>0)
                            ):

                        new_reward[action_i] = info_dict["reward"]*2

                    # catch all no actions and give them halve the reward of the final actions. (The agent should have tried to buy energy.)
                    elif ((info_dict["market_balance"][time_i] < info_dict["sum_mWh"]) & (imbalance_bought_too_much == -1) 
                            & (actions[action_i][0] < 0) & (actions[action_i][2]<0)
                            ):

                        new_reward[action_i] = info_dict["reward"]

                    elif ((info_dict["market_balance"][time_i] < info_dict["sum_mWh"]) & (imbalance_bought_too_much == -1) 
                            & (actions[action_i][0] > 0) & (actions[action_i][2]>0)
                            ):

                        new_reward[action_i] = info_dict["reward"] / 10



                    # The case of buying too much. 
                    elif ((info_dict["market_balance"][time_i] > info_dict["sum_mWh"]) & (imbalance_bought_too_much == 1) 
                            & (actions[action_i][0] > 0) & (actions[action_i][2]>0)
                            ):

                        new_reward[action_i] = info_dict["reward"]*2

                    # catch all no actions and give them halve the reward of the final actions. (The agent should have tried to buy energy.)
                    elif ((info_dict["market_balance"][time_i] > info_dict["sum_mWh"]) & (imbalance_bought_too_much == 1) 
                            & (actions[action_i][0] > 0) & (actions[action_i][2]<0)
                            ):

                        new_reward[action_i] = info_dict["reward"]

                    elif ((info_dict["market_balance"][time_i] > info_dict["sum_mWh"]) & (imbalance_bought_too_much == 1) 
                            & (actions[action_i][0] < 0) & (actions[action_i][2]>0)
                            ):

                        new_reward[action_i] = info_dict["reward"] / 10

            # Catch all other rewards and give them the reward.
            for time_i in range(len(postprocessed_batch)):
                if new_reward[time_i] == 0:
                    new_reward[time_i] = info_dict["reward"]

                    
                        
                        

            postprocessed_batch["rewards"] = new_reward
            logging.info(f"post_pro: {new_reward}")


                






            # This means if sum_mWh is smaller than reward_market_balance and the diff is over 100.
            #if info_dict["balancing_reward"] < -0.01: 
            #    i = 0
            #    
            #    #for action in postprocessed_batch["action"]:
#
            #    #    if info_dict[""]
            #    #    #logging.info(f"obs {obs}")
            #    #    new_reward[i] = -2
            #    #    i+=1
#
            #else:

                # set the reward to the reward for the whole batch. 
                
                
            #if len(reward) <= 100:
            #    if reward[len(reward)-1] != 0.0:
            #        
            #        i = 0
            #        for value in reward:
            #            new_reward[i] = reward[len(reward)-1]
            #            i+=1
            #        postprocessed_batch["rewards"] = new_reward
            #        self.last_reward = reward[len(reward)-1]
            #        episode.user_data["env_reward"].append(reward[len(reward)-1])
            #        
            #        episode.hist_data["env_reward"].append(reward[len(reward)-1])
            #    else:
            #        
            #        postprocessed_batch["rewards"] = new_reward
            #else:
            #    logging.info(f"len of reward not 24 instead: {len(reward)}")


            #batch = episode.new_batch_builder()
            #for each transition:
            #    batch.add_values(...)  # see sampler for usage
            #episode.extra_batches.add(batch.build_and_reset())
            env_reward = episode.user_data["env_reward"]
            return_reward = postprocessed_batch["rewards"]
            episode_info = episode.hist_data["env_reward"]
            episode.custom_metrics["reward"] = env_reward
            #episode_info2 = episode.last_pi_info_for()
            episode_id = episode.episode_id
            #logging.info(f"post_pro: new_reward {return_reward}, , env_reward={env_reward}")
            #logging.info(f"post_pro: episode {episode_info}, , episode_id={episode_id}")
            
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
                    "policy.learn_on_batch() result: {} ->  result: {}".format(
                        policy, result
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

    @override(DefaultCallbacks)
    def on_train_result(self, *, result: dict, **kwargs):
        try:
            logging.info(
                "Algorithm.train() result: -> {} episodes".format(
                    result["episodes_this_iter"]
                )
            )
            # you can mutate the result dict to add new fields to return
            result["callback_ok"] = True
        except Exception as e:
                logging.info(f"Callback error: {e}", exc_info=True)
