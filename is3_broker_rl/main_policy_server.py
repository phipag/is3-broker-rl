import logging
import os

import dotenv
import ray

from is3_broker_rl.conf import setup_logging
from is3_broker_rl.model.consumption_tariff_policy_server import start_policy_server


def main() -> None:
    dotenv.load_dotenv(override=False)
    setup_logging(log_filename="is3_consumption_rl.log")
    log = logging.getLogger(__name__)
    log.debug(f"Policy server driver process using environment variables: {os.environ}")

    ray.init(address="auto")
    log.info("Starting policy server ...")
    try:
        start_policy_server()
    except Exception:
        log.exception("Policy server main loop crashed.")
        raise


if __name__ == "__main__":
    main()
