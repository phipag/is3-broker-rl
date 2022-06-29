import logging
import os

import dotenv
import ray

from is3_broker_rl.api.consumption_tariff_controller import ConsumptionTariffController
from is3_broker_rl.api.wholesale_controller import WholesaleController
from is3_broker_rl.conf import setup_logging


def main() -> None:
    dotenv.load_dotenv(override=False)
    setup_logging(log_filename="is3_policy_client_rl.log")
    log = logging.getLogger(__name__)
    log.debug(f"Policy client driver process using environment variables: {os.environ}")

    ray.init(address="auto")

    log.info("Starting policy client API ...")
    ConsumptionTariffController.deploy()  # type: ignore
    WholesaleController.deploy()  # type: ignore

    # We keep this alive until terminated
    while True:
        continue


if __name__ == "__main__":
    main()
