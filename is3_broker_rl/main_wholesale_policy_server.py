import logging

import dotenv
import ray
from ray import serve
from is3_broker_rl.conf import setup_logging
from is3_broker_rl.model.wholesale_policy_server import start_policy_server


def main() -> None:
    dotenv.load_dotenv(override=False)
    setup_logging()
    log = logging.getLogger(__name__)

    ray.init(address="auto")
    #serve.start()
    log.info("Starting policy server ...")
    start_policy_server()


if __name__ == "__main__":
    main()
