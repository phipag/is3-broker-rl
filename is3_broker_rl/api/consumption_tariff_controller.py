import logging
from typing import Optional

import dotenv
from fastapi import HTTPException
from ray import serve
from ray.rllib.env import PolicyClient

from is3_broker_rl.api.dto import (
    Action,
    ActionResponse,
    EndEpisodeRequest,
    Episode,
    GetActionRequest,
    LogReturnsRequest,
    StartEpisodeRequest,
)
from is3_broker_rl.api.fastapi_app import fastapi_app
from is3_broker_rl.conf import setup_logging
from is3_broker_rl.model.consumption_tariff_policy_server import (
    SERVER_ADDRESS,
    SERVER_BASE_PORT,
)


@serve.deployment(route_prefix="/consumption-tariff")
@serve.ingress(fastapi_app)
class ConsumptionTariffController:
    def __init__(self) -> None:
        # This runs in a Ray actor worker process, so we have to initialize the logging again
        dotenv.load_dotenv(override=False)
        setup_logging()
        self._log = logging.getLogger(__name__)

        self._policy_client = PolicyClient(f"http://{SERVER_ADDRESS}:{SERVER_BASE_PORT}", inference_mode="remote")
        self._episode: Optional[Episode] = None

    def _check_episode_started(self):
        if not self._episode:
            raise HTTPException(
                status_code=412, detail="Cannot call this method before starting an episode. Call /start-episode first."
            )

    @fastapi_app.post("/start-episode", response_model=Episode)
    def start_episode(self, request: StartEpisodeRequest) -> Episode:
        self._log.debug(f"Called start_episode with {request}.")
        episode_id = self._policy_client.start_episode(training_enabled=request.training_enabled)
        self._episode = Episode(episode_id=episode_id)
        self._log.info(f"Started new episode with episode_id={episode_id}.")

        return self._episode

    @fastapi_app.post("/get-action", response_model=ActionResponse)
    def get_action(self, request: GetActionRequest) -> ActionResponse:
        self._log.debug(f"Called get_action with {request}. Feature vector: {request.observation.to_feature_vector()}.")
        self._check_episode_started()

        action = self._policy_client.get_action(self._episode.episode_id, request.observation.to_feature_vector())
        self._log.info(f"Algorithm predicted action={action}. Persisting to .csv file ...")
        # TODO: Implement persistence to .csv file

        return ActionResponse(action=Action(action))

    @fastapi_app.post("/log-returns")
    def log_returns(self, request: LogReturnsRequest) -> None:
        self._log.debug(f"Called log_returns with {request}.")
        self._check_episode_started()

        self._policy_client.log_returns(self._episode.episode_id, request.reward)

    @fastapi_app.post("/end-episode")
    def end_episode(self, request: EndEpisodeRequest) -> None:
        self._log.debug(f"Called end_episode with {request}.")
        self._policy_client.end_episode(self._episode.episode_id, request.observation.to_feature_vector())
