import os

from ray.rllib.agents import with_common_config
from ray.tune import tune

from is3_broker_rl.model.normalize_reward_callback import NormalizeRewardCallback

N_WORKERS = 0

cartpole_config = with_common_config(
    {
        "env": "CartPole-v0",
        # Normalize the observations
        "observation_filter": "MeanStdFilter",
        # Normalize the rewards
        "callbacks": NormalizeRewardCallback,
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
            "epsilon_timesteps": 2000,
        },
        # DQN
        "replay_buffer_config": {
            "type": "MultiAgentPrioritizedReplayBuffer",
            # Wait at least one episode before starting learning.
            "learning_starts": 12
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

tune.run(
    "DQN",
    config=cartpole_config,
    stop=None,
    verbose=2,
    local_dir=os.environ.get("DATA_DIR", "logs/"),
    log_to_file=True,
    name="Cartpole_Test1",
    # resume="AUTO",  # Will load the latest checkpoint from the local experiment directory or start a new one
)
