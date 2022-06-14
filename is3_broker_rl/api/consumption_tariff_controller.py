import logging
from typing import Dict

import dotenv
from ray import serve

from is3_broker_rl.api.fastapi_app import fastapi_app
from is3_broker_rl.conf import setup_logging


@serve.deployment(route_prefix="/consumption-tariff")
@serve.ingress(fastapi_app)
class ConsumptionTariffController:
    def __init__(self) -> None:
        dotenv.load_dotenv(override=False)
        setup_logging()
        self._log = logging.getLogger(__name__)

    @fastapi_app.get("/hello")
    def hello_world(self) -> Dict[str, str]:
        self._log.debug("Called hello_world.")

        return {"hello": "World"}
