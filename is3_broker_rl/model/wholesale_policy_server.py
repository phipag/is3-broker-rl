# `InputReader` generator (returns None if no input reader is needed on
# the respective worker).
import logging
import os

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

    trainer_name = "PPO"
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

        config = with_common_config(config)

        log = logging.getLogger(__name__)
        log.debug("Starting training loop ...")
        tune.run(
            "PPO",
            config=config,
            stop=None,
            checkpoint_at_end=True,
            checkpoint_freq=1,
            verbose=2,
            local_dir=os.environ.get("DATA_DIR", "logs/"),
            log_to_file=True,
            name="wholesale_PPO_Test",
            resume="AUTO",
        )
