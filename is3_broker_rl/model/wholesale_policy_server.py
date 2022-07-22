# `InputReader` generator (returns None if no input reader is needed on
# the respective worker).
import logging
import os

from ray.rllib.agents.callbacks import RE3UpdateCallbacks
from ray.rllib.agents.trainer import with_common_config
from ray.rllib.env import PolicyServerInput
from ray.tune import tune

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

    trainer_name = "SAC"
    if trainer_name == "DQN":
        config = {
            "env": None,
            "observation_space": observation_space,
            "action_space": action_space,
            # Use the `PolicyServerInput` to generate experiences.
            "input": _input,
            # Use n worker processes to listen on different ports.
            "num_workers": N_WORKERS,
            # Disable OPE, since the rollouts are coming from online clients.
            "input_evaluation": [],
            # Create a "chatty" client/server or not.
            # "callbacks": MyCallbacks,
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

        log = logging.getLogger(__name__)
        log.debug("Starting training loop ...")
        tune.run(
            "DQN",
            config=config,
            stop=None,
            verbose=2,
            local_dir=os.environ.get("DATA_DIR", "logs/"),
            log_to_file=True,
            name="wholesale_DQN_Test",
        )
    elif trainer_name == "PPO":
        config = {
            "env": None,
            # gridImbalance, ownImbalance, customerNetDemand, customerCount, marketPosition
            "observation_space": observation_space,
            "action_space": action_space,
            # Use the `PolicyServerInput` to generate experiences.
            "input": _input,
            # Use n worker processes to listen on different ports.
            "num_workers": N_WORKERS,
            # Disable OPE, since the rollouts are coming from online clients.
            "input_evaluation": [],
            # Create a "chatty" client/server or not.
            # "callbacks": MyCallbacks,
            # DL framework to use.
            "framework": "tf2",
            # Set to INFO so we'll see the server's actual address:port.
            "log_level": "DEBUG",
            "train_batch_size": 2,
            "timesteps_per_iteration": 512,
            "observation_filter": "MeanStdFilter",
        }

        config.update(
            {
                "lr_schedule": None,
                "use_critic": True,
                "use_gae": True,
                "lambda": 1.0,
                "kl_coeff": 0.2,
                "sgd_minibatch_size": 1,
                "num_sgd_iter": 30,
                "shuffle_sequences": True,
                "vf_loss_coeff": 0.5,
                "entropy_coeff": 0.2,
                "entropy_coeff_schedule": None,
                "clip_param": 0.3,
                "grad_clip": None,
                "kl_target": 0.01,
            }
        )

        config["model"] = {
            "fcnet_hiddens": [64],
            "fcnet_activation": "linear",
        }

    elif trainer_name == "SAC":
        config = {
            # gridImbalance, ownImbalance, customerNetDemand, customerCount, marketPosition
            "env": None,
            "observation_space": observation_space,
            "action_space": action_space,
            "input": _input,
            # === Model ===
            # Use two Q-networks (instead of one) for action-value estimation.
            # Note: Each Q-network will have its own target network.
            "twin_q": True,
            # Use a e.g. conv2D state preprocessing network before concatenating the
            # resulting (feature) vector with the action input for the input to
            # the Q-networks.
            # "use_state_preprocessor": DEPRECATED_VALUE,
            "observation_filter": "MeanStdFilter",
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
            "n_step": 25,
            # Number of env steps to optimize for before returning.
            "timesteps_per_iteration": 100,
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
            "training_intensity": 100,
            # Update the replay buffer with this many samples at once. Note that this
            # setting applies per-worker if num_workers > 1.
            "rollout_fragment_length": 1,
            # Size of a batched sampled from replay buffer for training.
            "train_batch_size": 32,
            # Update the target network every `target_network_update_freq` steps.
            "target_network_update_freq": 1,
            # === Optimization ===
            "optimization": {
                "actor_learning_rate": 3e-3,
                "critic_learning_rate": 3e-3,
                "entropy_learning_rate": 3e-3,
            },
            "input_evaluation": [],
            #"simple_optimizer": True,
            "framework": "tf2",
            "Q_model": {
                "fcnet_hiddens": [256, 256],
                "fcnet_activation": "relu",
                "post_fcnet_hiddens": [],
                "post_fcnet_activation": None,
                "custom_model": None,  # Use this to define custom Q-model(s).
                "custom_model_config": {},
            },
            # Model options for the policy function (see `Q_model` above for details).
            # The difference to `Q_model` above is that no action concat'ing is
            # performed before the post_fcnet stack.
            "policy_model": {
                "fcnet_hiddens": [256, 256],
                "fcnet_activation": "relu",
                "post_fcnet_hiddens": [],
                "post_fcnet_activation": None,
                "custom_model": None,  # Use this to define a custom policy model.
                "custom_model_config": {},
            },
            # === Replay buffer ===
            # Size of the replay buffer (in time steps).
            "replay_buffer_config": {
                "_enable_replay_buffer_api": True,
                "type": "MultiAgentPrioritizedReplayBuffer",
                "capacity": int(1e5),
                # How many steps of the model to sample before learning starts.
                "learning_starts": 1,
                "storage_unit": "timesteps",
            
            
                # If True prioritized replay buffer will be used.
                #"prioritized_replay": True,
                "prioritized_replay_alpha": 0.6,
                "prioritized_replay_beta": 0.4,
                "prioritized_replay_eps": 1e-6,
                # Whether to LZ4 compress observations
                "compress_observations": True,
            },
            "store_buffer_in_checkpoints": True,
            
        }

    config = with_common_config(config)
    # See https://github.com/ray-project/ray/blob/c9c3f0745a9291a4de0872bdfa69e4ffdfac3657/rllib/utils/exploration/tests/test_random_encoder.py#L35=
    class RE3Callbacks(RE3UpdateCallbacks, config["callbacks"]):
        pass

    config["callbacks"] = RE3Callbacks
    config["exploration_config"] = {
        "type": "RE3",
        # the dimensionality of the observation embedding vectors in latent space.
        "embeds_dim": 128,
        "rho": 0.1,  # Beta decay factor, used for on-policy algorithm.
        "k_nn": 20,  # Number of neighbours to set for K-NN entropy estimation.
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
        "beta": 0.2,
        # Schedule to use for beta decay, one of constant" or "linear_decay".
        "beta_schedule": "constant",
        # Specify, which exploration sub-type to use (usually, the algo's "default"
        # exploration, e.g. EpsilonGreedy for DQN, StochasticSampling for PG/SAC).
        "sub_exploration": {
            "type": "StochasticSampling",
        },
    }
    
    log = logging.getLogger(__name__)
    log.debug("Starting training loop ...")
    tune.run(
        trainer_name,
        config=config,
        stop=None,
        checkpoint_at_end=True,
        checkpoint_freq=1,
        verbose=3,
        local_dir=os.environ.get("DATA_DIR", "logs/"),
        log_to_file=True,
        name=f"{trainer_name}_fixedReward_Test4",
        #resume="AUTO", # If the trial failed use restore="path_to_checkpoint" instead. 
        mode="max",
        fail_fast=True,
        restore="logs/SAC_fixedReward_Test4/SAC_None_f615c_00000_0_2022-07-22_14-58-20/checkpoint_000013/checkpoint-13"
        )
