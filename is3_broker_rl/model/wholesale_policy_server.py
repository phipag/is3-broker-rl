# `InputReader` generator (returns None if no input reader is needed on
# the respective worker).
from functools import partial
from gc import callbacks
import logging
import os
import ray
import tensorflow as tf
from tensorflow.nn import leaky_relu
from ray.rllib.agents.callbacks import RE3UpdateCallbacks
from ray.rllib.agents.trainer import with_common_config
from ray.rllib.env import PolicyServerInput
from ray.tune import tune
from is3_broker_rl.model.normalize_reward_callback import WholesaleNormalizeRewardCallback
from ray.rllib.agents.callbacks import MultiCallbacks
from is3_broker_rl.model.wholesale_util import Env_config

SERVER_ADDRESS = "localhost"
SERVER_BASE_PORT = 9920
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
    
    env_config = Env_config()
    observation_space, action_space = env_config.get_gym_spaces()

    trainer_name = "TD3"
    enable_RE3_exploration = False


    #if trainer_name == "DQN":
    #    config = {
    #        "env": None,
    #        "observation_space": observation_space,
    #        "action_space": action_space,
    #        # Use the `PolicyServerInput` to generate experiences.
    #        "input": _input,
    #        # Use n worker processes to listen on different ports.
    #        "num_workers": N_WORKERS,
    #        # Disable OPE, since the rollouts are coming from online clients.
    #        "input_evaluation": [],
    #        # Create a "chatty" client/server or not.
    #        # "callbacks": MyCallbacks,
    #        # DL framework to use.
    #        "framework": "tf2",
    #        # Set to INFO so we'll see the server's actual address:port.
    #        "log_level": "DEBUG",
    #    }
#
    #    # For DQN
    #    config.update(
    #        {
    #            "learning_starts": 100,
    #            "timesteps_per_iteration": 200,
    #            "n_step": 3,
    #        }
    #    )
    #    config["model"] = {
    #        "fcnet_hiddens": [64],
    #        "fcnet_activation": "linear",
    #    }
#
    #    log = logging.getLogger(__name__)
    #    log.debug("Starting training loop ...")
    #    tune.run(
    #        "DQN",
    #        config=config,
    #        stop=None,
    #        verbose=2,
    #        local_dir=os.environ.get("DATA_DIR", "logs/"),
    #        log_to_file=True,
    #        name="wholesale_DQN_Test",
    #    )
    #elif trainer_name == "PPO":
    #    config = {
    #        "env": None,
    #        # gridImbalance, ownImbalance, customerNetDemand, customerCount, marketPosition
    #        "observation_space": observation_space,
    #        "action_space": action_space,
    #        # Use the `PolicyServerInput` to generate experiences.
    #        "input": _input,
    #        # Use n worker processes to listen on different ports.
    #        "num_workers": N_WORKERS,
    #        # Disable OPE, since the rollouts are coming from online clients.
    #        "input_evaluation": [],
    #        # Create a "chatty" client/server or not.
    #        # "callbacks": MyCallbacks,
    #        # DL framework to use.
    #        "framework": "tf2",
    #        # Set to INFO so we'll see the server's actual address:port.
    #        "log_level": "DEBUG",
    #        "train_batch_size": 2,
    #        "timesteps_per_iteration": 512,
    #        "observation_filter": "MeanStdFilter",
    #    }
#
    #    config.update(
    #        {
    #            "lr_schedule": None,
    #            "use_critic": True,
    #            "use_gae": True,
    #            "lambda": 1.0,
    #            "kl_coeff": 0.2,
    #            "sgd_minibatch_size": 1,
    #            "num_sgd_iter": 30,
    #            "shuffle_sequences": True,
    #            "vf_loss_coeff": 0.5,
    #            "entropy_coeff": 0.2,
    #            "entropy_coeff_schedule": None,
    #            "clip_param": 0.3,
    #            "grad_clip": None,
    #            "kl_target": 0.01,
    #        }
    #    )
#
    #    config["model"] = {
    #        "fcnet_hiddens": [64],
    #        "fcnet_activation": "linear",
    #    }
#
    #elif trainer_name == "SAC":
    #    config = {
    #        # gridImbalance, ownImbalance, customerNetDemand, customerCount, marketPosition
    #        "env": None,
    #        "observation_space": observation_space,
    #        "action_space": action_space,
    #        "input": _input,
    #        # === Model ===
    #        # Use two Q-networks (instead of one) for action-value estimation.
    #        # Note: Each Q-network will have its own target network.
    #        "twin_q": True,
    #        # Use a e.g. conv2D state preprocessing network before concatenating the
    #        # resulting (feature) vector with the action input for the input to
    #        # the Q-networks.
    #        # "use_state_preprocessor": DEPRECATED_VALUE,
    #        "observation_filter": "MeanStdFilter",
    #        # Model options for the Q network(s). These will override MODEL_DEFAULTS.
    #        # The `Q_model` dict is treated just as the top-level `model` dict in
    #        # setting up the Q-network(s) (2 if twin_q=True).
    #        # That means, you can do for different observation spaces:
    #        # obs=Box(1D) -> Tuple(Box(1D) + Action) -> concat -> post_fcnet
    #        # obs=Box(3D) -> Tuple(Box(3D) + Action) -> vision-net -> concat w/ action
    #        #   -> post_fcnet
    #        # obs=Tuple(Box(1D), Box(3D)) -> Tuple(Box(1D), Box(3D), Action)
    #        #   -> vision-net -> concat w/ Box(1D) and action -> post_fcnet
    #        # You can also have SAC use your custom_model as Q-model(s), by simply
    #        # specifying the `custom_model` sub-key in below dict (just like you would
    #        # do in the top-level `model` dict.
    #        # N-step target updates. If >1, sars' tuples in trajectories will be
    #        # postprocessed to become sa[discounted sum of R][s t+n] tuples.
    #        "n_step": 25,
    #        # Number of env steps to optimize for before returning.
    #        "timesteps_per_iteration": 1,
    #        # The intensity with which to update the model (vs collecting samples from
    #        # the env). If None, uses the "natural" value of:
    #        # `train_batch_size` / (`rollout_fragment_length` x `num_workers` x
    #        # `num_envs_per_worker`).
    #        # If provided, will make sure that the ratio between ts inserted into and
    #        # sampled from the buffer matches the given value.
    #        # Example:
    #        #   training_intensity=1000.0
    #        #   train_batch_size=250 rollout_fragment_length=1
    #        #   num_workers=1 (or 0) num_envs_per_worker=1
    #        #   -> natural value = 250 / 1 = 250.0
    #        #   -> will make sure that replay+train op will be executed 4x as
    #        #      often as rollout+insert op (4 * 250 = 1000).
    #        # See: rllib/agents/dqn/dqn.py::calculate_rr_weights for further details.
    #        "training_intensity": 100,
    #        # Update the replay buffer with this many samples at once. Note that this
    #        # setting applies per-worker if num_workers > 1.
    #        
    #        "rollout_fragment_length": 8,
    #        # Size of a batched sampled from replay buffer for training.
    #        "train_batch_size": 8,
    #        # Update the target network every `target_network_update_freq` steps.
    #        "target_network_update_freq": 1,
    #        # === Optimization ===
    #        "optimization": {
    #            "actor_learning_rate": 3e-4,
    #            "critic_learning_rate": 3e-4,
    #            "entropy_learning_rate": 3e-5,
    #        },
    #        "input_evaluation": [],
    #        #"simple_optimizer": True,
    #        "framework": "tf2",
    #        "Q_model": {
    #            "fcnet_hiddens": [256, 256, 256, 256],
    #            "fcnet_activation": "relu",
    #            "post_fcnet_hiddens": [],
    #            "post_fcnet_activation": None,
    #            "custom_model": None,  # Use this to define custom Q-model(s).
    #            "custom_model_config": {},
    #        },
    #        # Model options for the policy function (see `Q_model` above for details).
    #        # The difference to `Q_model` above is that no action concat'ing is
    #        # performed before the post_fcnet stack.
    #        "policy_model": {
    #            "fcnet_hiddens": [256, 256, 256, 256],
    #            "fcnet_activation": "relu",
    #            "post_fcnet_hiddens": [],
    #            "post_fcnet_activation": None,
    #            "custom_model": None,  # Use this to define a custom policy model.
    #            "custom_model_config": {},
    #        },
    #        # === Replay buffer ===
    #        # Size of the replay buffer (in time steps).
    #        "replay_buffer_config": {
    #            "_enable_replay_buffer_api": True,
    #            "type": "MultiAgentPrioritizedReplayBuffer",
    #            "capacity": int(1e5),
    #            # How many steps of the model to sample before learning starts.
    #            "learning_starts": 1,
    #            "storage_unit": "timesteps",
    #        
    #        
    #            # If True prioritized replay buffer will be used.
    #            #"prioritized_replay": True,
    #            "prioritized_replay_alpha": 0.5,
    #            "prioritized_replay_beta": 0.2,
    #            "prioritized_replay_eps": 1e-5,
    #            # Whether to LZ4 compress observations
    #            "compress_observations": True,
    #        },
    #        "store_buffer_in_checkpoints": True,
    #        "tau" : 5e-4,
    #        "initial_alpha": 0.8,
    #        "target_entropy" :"auto"
    #        #"callbacks" : MultiCallbacks([RE3Callbacks(), WholesaleNormalizeRewardCallback()]),
    #    }
#
    #
    ## See https://github.com/ray-project/ray/blob/c9c3f0745a9291a4de0872bdfa69e4ffdfac3657/rllib/utils/exploration/tests/test_random_encoder.py#L35=
    #
    #config = with_common_config(config)
    """
    config= {
        "num_gpus": 1
        "fake_gpus": true
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
        "zero_init_states": False,
        "optimization": {
            "actor_learning_rate": 0.005,
            "critic_learning_rate": 0.005,
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
        "simple_optimizer": True
    }
    #config["callbacks"] =,
    if enable_RE3_exploration == True:
        config["callbacks"] = MultiCallbacks(
        [
            config["callbacks"],
            partial(
                RE3UpdateCallbacks,
                embeds_dim=128,
                beta_schedule="linear_decay",
                k_nn=50,
            ),
            WholesaleNormalizeRewardCallback,
        ]
        )
        config["exploration_config"] = {
            "type": "RE3",
            # the dimensionality of the observation embedding vectors in latent space.
            "embeds_dim": 128,
            "rho": 0.01,  # Beta decay factor, used for on-policy algorithm.
            "k_nn": 50,  # Number of neighbours to set for K-NN entropy estimation.
            # Configuration for the encoder network, producing embedding vectors from observations.
            # This can be used to configure fcnet- or conv_net setups to properly process any
            # observation space. By default uses the Policy model configuration.
            "encoder_net_config": {
                "fcnet_hiddens": [],
                "fcnet_activation": "relu",
            },
            # Hyperparameter to choose between exploration and exploitation. A higher value of beta adds
            # more importance to the intrinsic reward, as per the following equation
            # `reward = r + beta * intrinsic_reward`
            "beta": 0.4,
            # Schedule to use for beta decay, one of constant" or "linear_decay".
            "beta_schedule": "linear_decay",
            # Specify, which exploration sub-type to use (usually, the algo's "default"
            # exploration, e.g. EpsilonGreedy for DQN, StochasticSampling for PG/SAC).
            "sub_exploration": {
                "type": "StochasticSampling",
            },
        }
    else:
        pass
         #config["callbacks"] = WholesaleNormalizeRewardCallback
    """

    DEFAULT_CONFIG = with_common_config({
    # === Twin Delayed DDPG (TD3) and Soft Actor-Critic (SAC) tricks ===
    # TD3: https://spinningup.openai.com/en/latest/algorithms/td3.html
    # In addition to settings below, you can use "exploration_noise_type" and
    # "exploration_gauss_act_noise" to get IID Gaussian exploration noise
    # instead of OU exploration noise.
    # twin Q-net
    "framework": "tf2",
    "twin_q": True,
    "env": None,
        "observation_space": observation_space,
        "action_space": action_space,
        "input": _input,
    # delayed policy update
    "policy_delay": 1,
    # target policy smoothing
    # (this also replaces OU exploration noise with IID Gaussian exploration
    # noise, for now)
    "smooth_target_policy": False,
    # gaussian stddev of target action noise for smoothing
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
    "actor_hiddens": [400, 300, 200],
    # Hidden layers activation of the postprocessing stage of the policy
    # network
    "actor_hidden_activation": "relu",
    # Postprocess the critic network model output with these hidden layers;
    # again, if use_state_preprocessor is True, then the state will be
    # preprocessed by the model specified with the "model" config option first.
    "critic_hiddens": [400, 300, 200],
    # Hidden layers activation of the postprocessing state of the critic.
    "critic_hidden_activation": "relu",
    # N-step Q learning
    "n_step": 24,

    # === Exploration ===
    "exploration_config": {
        # DDPG uses OrnsteinUhlenbeck (stateful) noise to be added to NN-output
        # actions (after a possible pure random phase of n timesteps).
        "type": "OrnsteinUhlenbeckNoise",
        # For how many timesteps should we return completely random actions,
        # before we start adding (scaled) noise?
        "random_timesteps": 1500,
        # The OU-base scaling factor to always apply to action-added noise.
        "ou_base_scale": 0.1,
        # The OU theta param.
        "ou_theta": 0.15,
        # The OU sigma param.
        "ou_sigma": 0.2,
        # The initial noise scaling factor.
        "initial_scale": 0.1,
        # The final noise scaling factor.
        "final_scale": 0.0002,
        # Timesteps over which to anneal scale (from initial to final values).
        "scale_timesteps": 10000,
    },
    # Number of env steps to optimize for before returning
    "timesteps_per_iteration": 1,
    # Extra configuration that disables exploration.
    "input_evaluation": [],
    # === Replay buffer ===
    # Size of the replay buffer. Note that if async_updates is set, then
    # each worker will have a replay buffer of this size.
    "replay_buffer_config": {
        "type": "MultiAgentReplayBuffer",
        "capacity": 100000,
        "_enable_replay_buffer_api": True,
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
    "prioritized_replay_alpha": 0.6,
    # Beta parameter for sampling from prioritized replay buffer.
    "prioritized_replay_beta": 0.4,
    # Epsilon to add to the TD errors when updating priorities.
    "prioritized_replay_eps": 1e-6,
    # Whether to LZ4 compress observations
    "compress_observations": True,
    
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
    "training_intensity": None,

    # === Optimization ===
    # Learning rate for the critic (Q-function) optimizer.
    "critic_lr": tf.keras.optimizers.schedules.ExponentialDecay(0.001, 100, 0.85),
    # Learning rate for the actor (policy) optimizer.
    "actor_lr": tf.keras.optimizers.schedules.ExponentialDecay(0.001, 100, 0.85),
    # Update the target network every `target_network_update_freq` steps.
    "target_network_update_freq": 1,
    # Update the target by \tau * policy + (1-\tau) * target_policy
    "tau": 0.002,
    # If True, use huber loss instead of squared loss for critic network
    # Conventionally, no need to clip gradients if using a huber loss
    "use_huber": False,
    # Threshold of a huber loss
    "huber_threshold": 1.0,
    # Weights for L2 regularization
    "l2_reg": 1e-6,
    # If not None, clip gradients during optimization at this value
    "grad_clip": None,
    # How many steps of the model to sample before learning starts.
    "learning_starts": 1500,
    # Update the replay buffer with this many samples at once. Note that this
    # setting applies per-worker if num_workers > 1.
    "rollout_fragment_length": 1,
    # Size of a batched sampled from replay buffer for training. Note that
    # if async_updates is set, then each worker returns gradients for a
    # batch of this size.
    "train_batch_size": 8,
    "smooth_target_policy": True,
    # === Parallelism ===
    # Number of workers for collecting samples with. This only makes sense
    # to increase if your environment is particularly slow to sample, or if
    # you're using the Async or Ape-X optimizers.
    "num_workers": 0,
    # Whether to compute priorities on workers.
    "worker_side_prioritization": False,
    # Prevent reporting frequency from going lower than this time span.
    "min_time_s_per_reporting": 1,
    # Experimental flag.
    # If True, the execution plan API will not be used. Instead,
    # a Trainer's `training_iteration` method will be called as-is each
    # training iteration.
    "_disable_execution_plan_api": False,
    "train_batch_size": 4,
    
    #"num_gpus": 1,
    #"fake_gpus": True,
    #"simple_optimizer":True,
})
    DEFAULT_CONFIG["callbacks"] = WholesaleNormalizeRewardCallback
        



    log = logging.getLogger(__name__)
    log.debug("Starting training loop ...")
    tune.run(
        trainer_name,
        config=DEFAULT_CONFIG,
        #callbacks=[WholesaleNormalizeRewardCallback],
        stop=None,
        checkpoint_at_end=True,
        checkpoint_freq=500,
        verbose=3,
        local_dir=os.environ.get("DATA_DIR", "logs/"),
        log_to_file=True,
        name=f"{trainer_name}_fRlargerNN_nore3_newconffromRewaNor_Test6",
        resume="AUTO", # If the trial failed use restore="path_to_checkpoint" instead. 
        mode="max",
        fail_fast=True,
        #restore="is3_broker_rl/logs/TD3_fRlargerNN_nore3_newconffromRewaNor_Test6/TD3_None_32702_00000_0_2022-07-27_14-20-00/checkpoint_016500/checkpoint-16500"
        )