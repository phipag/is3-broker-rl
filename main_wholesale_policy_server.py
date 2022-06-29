import logging

import dotenv
import ray

from is3_broker_rl.conf import setup_logging
from is3_broker_rl.model.wholesale_policy_server import start_policy_server


def main() -> None:
    dotenv.load_dotenv(override=False)
    setup_logging(log_filename="is3_wholesale_rl.log")
    log = logging.getLogger(__name__)

    ray.init(address="auto")
    log.info("Starting policy server ...")
    start_policy_server()


if __name__ == "__main__":
    main()
