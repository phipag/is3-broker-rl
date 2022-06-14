import dotenv
import ray
from ray import serve

from is3_broker_rl.api.consumption_tariff_controller import ConsumptionTariffController
from is3_broker_rl.conf import setup_logging


def main() -> None:
    dotenv.load_dotenv(override=False)
    setup_logging()

    ray.init(address="auto", namespace="serve")
    serve.start()

    ConsumptionTariffController.deploy()  # type: ignore

    # We keep this script alive
    while True:
        continue


if __name__ == "__main__":
    main()
