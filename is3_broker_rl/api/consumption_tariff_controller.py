import logging

import dotenv
from ray import serve

from is3_broker_rl.api.fastapi_app import fastapi_app
from is3_broker_rl.api.models import (
    EndEpisodeRequest,
    Episode,
    GetActionRequest,
    LogReturnsRequest,
    StartEpisodeRequest,
)
from is3_broker_rl.conf import setup_logging


@serve.deployment(route_prefix="/consumption-tariff")
@serve.ingress(fastapi_app)
class ConsumptionTariffController:
    def __init__(self) -> None:
        dotenv.load_dotenv(override=False)
        setup_logging()
        self._log = logging.getLogger(__name__)

    @fastapi_app.post("/start-episode", response_model=Episode)
    def start_episode(self, request: StartEpisodeRequest) -> Episode:
        self._log.debug(f"Called start_episode with {request}.")

        return Episode(episode_id="ID")

    @fastapi_app.post("/end-episode")
    def end_episode(self, request: EndEpisodeRequest) -> None:
        self._log.debug(f"Called end_episode with {request}.")

    @fastapi_app.post("/log-returns")
    def log_returns(self, request: LogReturnsRequest) -> None:
        self._log.debug(f"Called log_returns with {request}.")

    @fastapi_app.post("/get-action")
    def get_action(self, request: GetActionRequest) -> None:
        self._log.debug(f"Called get_action with {request}.")
