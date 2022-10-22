
import logging
from typing_extensions import Self
import gym
import numpy as np
from ray.rllib.agents.trainer import with_common_config
from ray.rllib.env import PolicyServerInput
#from is3_broker_rl.model import wholesale_custom_model
from is3_broker_rl.model.consumption_tariff_config import N_WORKERS
from is3_broker_rl.model.wholesale_callbacks import MyCallbacks
from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.rllib.models import ModelCatalog
from ray.rllib.agents.trainer_config import TrainerConfig
# Class to initialize the action and observation space


class Env_config:
    def __init__(self, server_address, server_base_port, num_workers, discrete_action = True) -> None:
        self.server_address = server_address
        self.server_base_port = server_base_port
        self.num_workers = num_workers
        self.discrete_action = discrete_action
        l_bounds = []
        h_bounds = []
        
        # This looks like this, so we can change the state space with ease. 

        l_bounds.append(np.array([-np.inf] * 1))  # p_grid_imbalance = 0
        h_bounds.append(np.array([np.inf] * 1))
        l_bounds.append(np.array([-np.inf] * 1))  # p_customer_prosumption = 0
        h_bounds.append(np.array([np.inf] * 1))
        l_bounds.append(np.array([-np.inf] * 1))  # p_wholesale_price = 0
        h_bounds.append(np.array([np.inf] * 1))
        l_bounds.append(np.array([-np.inf] * 1))  # p_cloud_cover = 0
        h_bounds.append(np.array([np.inf] * 1))
        l_bounds.append(np.array([-np.inf] * 1))  # p_temperature = 0
        h_bounds.append(np.array([np.inf] * 1))
        l_bounds.append(np.array([-np.inf] * 1))  # p_wind_speed = 0
        h_bounds.append(np.array([np.inf] * 1))
        l_bounds.append(np.array([-np.inf] * 1))  # cleared_orders_price = 0
        h_bounds.append(np.array([np.inf] * 1))
        l_bounds.append(np.array([-np.inf] * 1))  # cleared_orders_energy = 0
        h_bounds.append(np.array([np.inf] * 1))
        l_bounds.append(np.array([-np.inf] * 1))  # cleared_trade_price = 0
        h_bounds.append(np.array([np.inf] * 1))
        l_bounds.append(np.array([-np.inf] * 1))  # cleared_trade_energy = 0
        h_bounds.append(np.array([np.inf] * 1))
        #l_bounds.append(np.array([-np.inf] * 1))  # customer_count
        #h_bounds.append(np.array([np.inf] * 1))
        #l_bounds.append(np.array([-np.inf] * 1))  # customer_change
        #h_bounds.append(np.array([np.inf] * 1))
        l_bounds.append(np.array([-np.inf] * 1))  # total_prosumption
        h_bounds.append(np.array([np.inf] * 1))
        l_bounds.append(np.array([-np.inf] * 1))  # market_position = 0
        h_bounds.append(np.array([np.inf] * 1))
        l_bounds.append(np.array([-np.inf] * 20))  # percentageSubs
        h_bounds.append(np.array([np.inf] * 20))
        l_bounds.append(np.array([-np.inf] * 20))  # ProsumptionPerGroup
        h_bounds.append(np.array([np.inf] * 20))
        l_bounds.append(np.array([-np.inf] * 1))  # NeededmWh
        h_bounds.append(np.array([np.inf] * 1))
        l_bounds.append(np.array([-np.inf] * 24))  # hour of the start with dummy.
        h_bounds.append(np.array([np.inf] * 24))
        l_bounds.append(np.array([-np.inf] * 7))  # day of the start with dummy
        h_bounds.append(np.array([np.inf] * 7))
        l_bounds.append(np.array([-np.inf] * 1))  # timeslot
        h_bounds.append(np.array([np.inf] * 1))
        l_bounds.append(np.array([-np.inf] * 24))  # time_diff
        h_bounds.append(np.array([np.inf] * 24))
        l_bounds.append(np.array([-np.inf] * 24))  # action_hist
        h_bounds.append(np.array([np.inf] * 24))
        l_bounds.append(np.array([-np.inf] * 1))  # unclearedOrdersMWhAsks
        h_bounds.append(np.array([np.inf] * 1))
        l_bounds.append(np.array([-np.inf] * 1))  # unclearedOrdersMWhBids
        h_bounds.append(np.array([np.inf] * 1))
        l_bounds.append(np.array([-np.inf] * 1))  # weigthedAvgPriceAsks
        h_bounds.append(np.array([np.inf] * 1))
        l_bounds.append(np.array([-np.inf] * 1))  # weigthedAvgPriceBids
        h_bounds.append(np.array([np.inf] * 1))

        l_bound_total = np.array([])
        for j in l_bounds:
            l_bound_total = np.append(l_bound_total, j)
        r_bound_total = np.array([])
        for j in h_bounds:
            r_bound_total = np.append(r_bound_total, j)

        self.observation_space = gym.spaces.Box(
            low=np.ravel(l_bound_total),
            high=np.ravel(r_bound_total),
            dtype=np.float32
            # shape=observation_space_bounds[:, 0].shape,
        )

        self.energy_alpha = [-1, -0.75, -0.5, -0.25, -0.1, 0.1, 0.25, 0.5, 0.75, 1]
        self.price_beta = [0.5, 1, 2]
        if discrete_action == True:
            # Alpha, beta is in the paper. 
            
            number_actions = len(self.energy_alpha) * len(self.price_beta) +1 # +1 for no action
            self.action_space = gym.spaces.Discrete(n=number_actions) # Energy, Price 

        else:
            self.action_space = gym.spaces.Box(low=-1, high=1, shape=(3,)) # Energy, Price 

        
    def get_gym_spaces(self):

        return self.observation_space, self.action_space

    def get_descrete_action_bool(self):
        return self.discrete_action

    def get_action_size(self):
        return (len(self.price_beta) * len(self.energy_alpha)) +1

    


    def get_rl_config(self, trainer_name: str):
        def _input(ioctx):
        
        # We are remote worker or we are local worker with num_workers=0:
        # Create a PolicyServerInput.
            if ioctx.worker_index > 0 or ioctx.worker.num_workers == 0:
                return PolicyServerInput(
                    ioctx,
                    self.server_address,
                    self.server_base_port + ioctx.worker_index - (1 if ioctx.worker_index > 0 else 0),
                )
            # No InputReader (PolicyServerInput) needed.
            else:
                return None
        
        config_for_all_algos = {
            "env": None,
            "observation_space": self.observation_space,
            "action_space": self.action_space,
            "input": _input,
            "num_workers": 1,
            #"learning_starts": 5000 * 24, # * 24 because we take 24 actions per timeslot
            # Prevents learning on batches that do not have the right reward set.
            "batch_mode": "complete_episodes",
            
            # Use a e.g. conv2D state preprocessing network before concatenating the
            # resulting (feature) vector with the action input for the input to
            # the Q-networks.
            # "use_state_preprocessor": DEPRECATED_VALUE,
            #"observation_filter": "MeanStdFilter",
            # Model options for the Q network(s). These will override MODEL_DEFAULTS.
            # The `Q_model` dict is treated just as the top-level `model` dict in
            # setting up the Q-network(s) (2 if twin_q=True).
            # That means, you can do for different observation spaces:
            # obs=Box(1D) -> Tuple(Box(1D) + Action) -> concat -> post_fcnet
            # obs=Box(3D) -> Tuple(Box(3D) + Action) -> vision-net -> concat w/ action
            #   -> post_fcnet
            # obs=Tuple(Box(1D), Box(3D)) -> Tuple(Box(1D), Box(3D), Action)
            #   -> vision-net -> concat w/ Box(1D) and action -> post_fcnet
            # You can also have SAC use your custom_model as Q-model(s), by simply
            # specifying the `custom_model` sub-key in below dict (just like you would
            # do in the top-level `model` dict.
            # N-step target updates. If >1, sars' tuples in trajectories will be
            # postprocessed to become sa[discounted sum of R][s t+n] tuples.
            "n_step": 1,
            # Number of env steps to optimize for before returning.
            "timesteps_per_iteration": 24,
            # The intensity with which to update the model (vs collecting samples from
            # the env). If None, uses the "natural" value of:
            # `train_batch_size` / (`rollout_fragment_length` x `num_workers` x
            # `num_envs_per_worker`).
            # If provided, will make sure that the ratio between ts inserted into and
            # sampled from the buffer matches the given value.
            # Example:
            #   training_intensity=1000.0
            #   train_batch_size=250 rollout_fragment_length=1
            #   num_workers=1 (or 0) num_envs_per_worker=1
            #   -> natural value = 250 / 1 = 250.0
            #   -> will make sure that replay+train op will be executed 4x as
            #      often as rollout+insert op (4 * 250 = 1000).
            # See: rllib/agents/dqn/dqn.py::calculate_rr_weights for further details.
            "training_intensity": 500,
            # Update the replay buffer with this many samples at once. Note that this
            # setting applies per-worker if num_workers > 1.
            
            "rollout_fragment_length": 24,
            # Size of a batched sampled from replay buffer for training.
            "train_batch_size": 24*100,
            # Update the target network every `target_network_update_freq` steps.
            "target_network_update_freq": 1,
            "store_buffer_in_checkpoints": True,
            "callbacks": MyCallbacks,
            
            "log_level" : "INFO",
            "input_evaluation": [],
            # Whether to compute priorities on workers.
            "worker_side_prioritization": False,
            # Prevent reporting frequency from going lower than this time span.
            "min_time_s_per_reporting": 5,

        }
        config = with_common_config(config_for_all_algos)

        if trainer_name == "SAC":
            config["framework"] = "tf2"
            config["twin_q"] = False
            config["optimization"] = {
                "actor_learning_rate": 3e-5,
                "critic_learning_rate": 3e-3,
                "entropy_learning_rate": 3e-3,
            }
            config["Q_model"] = {
                "fcnet_hiddens": [1024, 1024],
                "fcnet_activation": "relu",
                "post_fcnet_hiddens": [],
                "post_fcnet_activation": None,
                "custom_model": None,  # Use this to define custom Q-model(s).
                "custom_model_config": {},
            }
            config["policy_model"] = {
                "fcnet_hiddens": [1024, 1024],
                "fcnet_activation": "relu",
                "post_fcnet_hiddens": [],
                "post_fcnet_activation": None,
                "custom_model": None,  # Use this to define a custom policy model.
                "custom_model_config": {},
            }
            config["replay_buffer_config"] = {
                "_enable_replay_buffer_api": True,
                "type": "MultiAgentPrioritizedReplayBuffer",
                "capacity": int(1e5),
                # How many steps of the model to sample before learning starts.
                "learning_starts": 50000, # * 24 because we take 24 actions per timeslot
                "storage_unit": "timesteps",
            
            
                # If True prioritized replay buffer will be used.
                "prioritized_replay": True,
                "prioritized_replay_alpha": 0.5,
                "prioritized_replay_beta": 0.2,
                "prioritized_replay_eps": 1e-5,
                # Whether to LZ4 compress observations
                "compress_observations": True,
            }
            config["tau"] = 5e-4
            config["initial_alpha"] = 1
            config["target_entropy"] = "auto"


        if trainer_name == "TD3":
            config["framework"] = "tf2"
            config["twin_q"] = True
            config["policy_delay"]= 1
            config.update({
                "smooth_target_policy": False,
                "target_noise": 0.2,
            # target noise limit (bound)
            "target_noise_clip": 0.5,

            # === Evaluation ===
            # Evaluate with epsilon=0 every `evaluation_interval` training iterations.
            # The evaluation stats will be reported under the "evaluation" metric key.
            # Note that evaluation is currently not parallelized, and that for Ape-X
            # metrics are already only reported for the lowest epsilon workers.
            "evaluation_interval": None,
            # Number of episodes to run per evaluation period.
            "evaluation_duration": 10,

            # === Model ===
            # Apply a state preprocessor with spec given by the "model" config option
            # (like other RL algorithms). This is mostly useful if you have a weird
            # observation shape, like an image. Disabled by default.
            #"use_state_preprocessor": "MeanStdFilter",
            # Postprocess the policy network model output with these hidden layers. If
            # use_state_preprocessor is False, then these will be the *only* hidden
            # layers in the network.
            "actor_hiddens": [64, 64, 64,64],
            # Hidden layers activation of the postprocessing stage of the policy
            # network
            "actor_hidden_activation": "relu",
            # Postprocess the critic network model output with these hidden layers;
            # again, if use_state_preprocessor is True, then the state will be
            # preprocessed by the model specified with the "model" config option first.
            "critic_hiddens": [200, 100],
            # Hidden layers activation of the postprocessing state of the critic.
            "critic_hidden_activation": "relu",
            # Number of env steps to optimize for before returning
            "timesteps_per_iteration": 24,
            "replay_buffer_config": {
                "type": "MultiAgentReplayBuffer",
                "capacity": 100000,
                "_enable_replay_buffer_api": True,
                "prioritized_replay":True,
            },
            # Set this to True, if you want the contents of your buffer(s) to be
            # stored in any saved checkpoints as well.
            # Warnings will be created if:
            # - This is True AND restoring from a checkpoint that contains no buffer
            #   data.
            # - This is False AND restoring from a checkpoint that does contain
            #   buffer data.
            "store_buffer_in_checkpoints": True,
            # If True prioritized replay buffer will be used.
            "prioritized_replay": True,
            # Alpha parameter for prioritized replay buffer.
            "prioritized_replay_alpha": 0.8,
            # Beta parameter for sampling from prioritized replay buffer.
            "prioritized_replay_beta": 0.4,
            # Epsilon to add to the TD errors when updating priorities.
            "prioritized_replay_eps": 1e-6,
            # Whether to LZ4 compress observations
            "compress_observations": True,
            "training_intensity": None,
             # === Optimization ===
            # Learning rate for the critic (Q-function) optimizer.
            "critic_lr": 0.001, #tf.keras.optimizers.schedules.ExponentialDecay(0.0001, 100, 0.85),
            # Learning rate for the actor (policy) optimizer.
            "actor_lr": 0.005,#tf.keras.optimizers.schedules.ExponentialDecay(0.0001, 100, 0.85),
            # Update the target network every `target_network_update_freq` steps.
            "target_network_update_freq": 1,
            "tau": 0.004,
            # If True, use huber loss instead of squared loss for critic network
            # Conventionally, no need to clip gradients if using a huber loss
            "use_huber": True,
            # Threshold of a huber loss
            "huber_threshold": 1.0,
            # Weights for L2 regularization
            "l2_reg": 1e-6,
            # If not None, clip gradients during optimization at this value
            "grad_clip": None,
            "smooth_target_policy": False,
            "exploration_config":{
                # TD3 uses simple Gaussian noise on top of deterministic NN-output
                # actions (after a possible pure random phase of n timesteps).
                "type": "GaussianNoise",
                # For how many timesteps should we return completely random
                # actions, before we start adding (scaled) noise?
                "random_timesteps": 10000,
                # Gaussian stddev of action noise for exploration.
                "stddev": 0.1,
                # Scaling settings by which the Gaussian noise is scaled before
                # being added to the actions. NOTE: The scale timesteps start only
                # after(!) any random steps have been finished.
                # By default, do not anneal over time (fixed 1.0).
                "initial_scale": 1.0,
                "final_scale": 1.0,
                "scale_timesteps": 1,
                }
            })

        if trainer_name == "A3C":

            config = with_common_config({
                "env": None,
                "observation_space": self.observation_space,
                "action_space": self.action_space,
                "input": _input,
                "framework": "tf2",
                "num_workers": 1,
                "input_evaluation": [],
                #"learning_starts": 5000 * 24, # * 24 because we take 24 actions per timeslot
                # Prevents learning on batches that do not have the right reward set.
                "batch_mode": "complete_episodes",
                "rollout_fragment_length": 24,
                # Use a e.g. conv2D state preprocessing network before concatenating the
                # resulting (feature) vector with the action input for the input to
                # the Q-networks.
                # "use_state_preprocessor": DEPRECATED_VALUE,
                #"observation_filter": "MeanStdFilter",
                "log_level": "INFO",
                "callbacks": MyCallbacks,
                "model": {
                    "use_lstm": True,
                    "lstm_cell_size": 256,
                    "lstm_use_prev_action_reward": True,
                    #"lstm_use_prev_reward": False,
                },
                
                
                
                "train_batch_size": 24,
                #"lstm_use_prev_action_reward": -1,
                #"batch_size": 24,
            })
            
            #t_config = TrainerConfig().training().environment(observation_space=self.observation_space, action_space=self.action_space).input_config().callbacks(MyCallbacks)
            #config = t_config.to_dict()
            
            #ModelCatalog.register_custom_model(
            #    "custom_model",
            #    wholesale_custom_model.MyModelClass
            #)
#
            config.update({
                #"sample_async" : False,
                #"model": "custom_model",
                #"microbatch_size" : None,
                #"learning_starts": None,
                
                "train_batch_size": 24,


            })

        if trainer_name == "PPO":

            config = with_common_config({
                "env": None,
                "observation_space": self.observation_space,
                "action_space": self.action_space,
                "input": _input,
                "framework": "tf2",
                "num_workers": 1,
                "input_evaluation": [],
                #"learning_starts": 5000 * 24, # * 24 because we take 24 actions per timeslot
                # Prevents learning on batches that do not have the right reward set.
                "batch_mode": "complete_episodes",
                "rollout_fragment_length": 24,
                "sgd_minibatch_size": 24,
                "num_sgd_iter" : 30,
                # Use a e.g. conv2D state preprocessing network before concatenating the
                # resulting (feature) vector with the action input for the input to
                # the Q-networks.
                # "use_state_preprocessor": DEPRECATED_VALUE,
                #"observation_filter": "MeanStdFilter",
                "log_level": "INFO",
                "callbacks": MyCallbacks,
                "shuffle_sequences": True,
                "lr": 1e-5,
                "model": {
                    "use_lstm": True,
                    "lstm_cell_size": 256,
                    "lstm_use_prev_action_reward": True,
                    #"lstm_use_prev_reward": False,
                },
                
                
                
                "train_batch_size": 24,
                #"lstm_use_prev_action_reward": -1,
                #"batch_size": 24,
            })
            # TODO: Register custom model.
            custom_model = wholesale_custom_model.MyModelClass(
                obs_space=self.observation_space,
                action_space=self.action_space,
                seq_len=24,
                model_config={
                },
            )
            #t_config = TrainerConfig().training().environment(observation_space=self.observation_space, action_space=self.action_space).input_config().callbacks(MyCallbacks)
            #config = t_config.to_dict()
            
            #ModelCatalog.register_custom_model(
            #    "custom_model",
            #    wholesale_custom_model.MyModelClass
            #)
#
            config.update({
                #"sample_async" : False,
                #"model": "custom_model",
                #"microbatch_size" : None,
                #"learning_starts": None,
                
                "train_batch_size": 24,


            })

        # See https://github.com/ray-project/ray/blob/c9c3f0745a9291a4de0872bdfa69e4ffdfac3657/rllib/utils/exploration/tests/test_random_encoder.py#L35=
        
        #config = with_common_config(config)
        
        """config= {
        "num_gpus": 0,
        "framework": "torch",
        "num_workers": 4,
        "num_envs_per_worker": 1,
        "num_cpus_per_worker": 1,
        "log_level": "INFO",
        "env": None,
        "observation_space": observation_space,
        "action_space": action_space,
        "input": _input,
        "twin_q": True,
        "gamma": 0.95,
        "batch_mode": "complete_episodes",
        "replay_buffer_config": {
          "_enable_replay_buffer_api": True,
            "type": "MultiAgentReplayBuffer",
            "storage_unit": "timesteps",
            "capacity": 100000,
            "learning_starts": 100,
            "replay_burn_in": 0,
            "replay_sequence_length": 24
        },
        "train_batch_size": 1,
        "target_network_update_freq": 4,
        "tau": 0.3,
        #"zero_init_states": False,
        "optimization": {
            "actor_learning_rate": 0.0005,
            "critic_learning_rate": 0.0005,
            "entropy_learning_rate": 0.0001,
        },
        "model": {
            "max_seq_len": 24,
        },
        "policy_model": {
            "use_lstm": True,
            "lstm_cell_size": 28,
            "fcnet_hiddens": [64, 64],
            "post_fcnet_hiddens": [24],
            "post_fcnet_activation": "relu",
            "lstm_use_prev_action": True,
            "lstm_use_prev_reward": True,
        },
        "Q_model": {
            "use_lstm": True,
            "lstm_cell_size": 28,
            "fcnet_hiddens": [64, 64],
            "post_fcnet_hiddens": [24],
            "post_fcnet_activation": "relu",
            "lstm_use_prev_action": True,
            "lstm_use_prev_reward": True,
        },
        "simple_optimizer": True,
         "store_buffer_in_checkpoints": True,
        # If True prioritized replay buffer will be used.
        "prioritized_replay": True,
        # Alpha parameter for prioritized replay buffer.
        "prioritized_replay_alpha": 0.6,
        # Beta parameter for sampling from prioritized replay buffer.
        "prioritized_replay_beta": 0.4,
        # Epsilon to add to the TD errors when updating priorities.
        "prioritized_replay_eps": 1e-6,
        # Whether to LZ4 compress observations
        "compress_observations": True,
        }
        """







        logging.info(config)


        return config


