import logging

import dotenv
import ray
from ray import serve

from is3_broker_rl.api.consumption_tariff_controller import ConsumptionTariffController
from is3_broker_rl.conf import setup_logging


def main() -> None:
    dotenv.load_dotenv(override=False)
    setup_logging()
    log = logging.getLogger(__name__)

    ray.init(address="auto")
    serve.start()

    log.info("Starting policy client API ...")
    ConsumptionTariffController.deploy()  # type: ignore

    # We keep this alive until terminated
    while True:
        continue


if __name__ == "__main__":
    main()
