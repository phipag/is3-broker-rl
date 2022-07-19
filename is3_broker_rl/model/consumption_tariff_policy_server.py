import logging
import os

from ray.tune import tune

from is3_broker_rl.model.consumption_tariff_config import dqn_config


def start_policy_server() -> None:
    log = logging.getLogger(__name__)
    log.debug("Starting training loop ...")
    tune.run(
        "DQN",
        config=dqn_config,
        stop=None,
        checkpoint_at_end=True,
        checkpoint_freq=1,
        verbose=2,
        local_dir=os.environ.get("DATA_DIR", "logs/"),
        log_to_file=True,
        name="DQN_Consumption_Test8",
        resume="AUTO",  # Will load the latest checkpoint from the local experiment directory or start a new one
    )
