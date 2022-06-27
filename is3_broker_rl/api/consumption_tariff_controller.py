import logging
import os
from pathlib import Path
from typing import Optional

import dotenv
import pandas as pd
from fastapi import HTTPException
from ray import serve
from ray.rllib.env import PolicyClient
from typing_extensions import TypeGuard

from is3_broker_rl.api.dto import (
    Action,
    ActionResponse,
    EndEpisodeRequest,
    Episode,
    GetActionRequest,
    LogReturnsRequest,
    Observation,
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
        setup_logging(log_filename="is3_consumption_rl.log")
        self._log = logging.getLogger(__name__)
        self._log.debug(f"Policy client actor using environment variables: {os.environ}")
        self._DATA_DIR: Path = Path(os.environ.get("DATA_DIR", "data/"))

        # We set inference_mode="remote" to guarantee on-policy behavior
        # (https://discuss.ray.io/t/externalenv-vs-external-application-clients/2371/2?u=phipag)
        self._policy_client = PolicyClient(f"http://{SERVER_ADDRESS}:{SERVER_BASE_PORT}", inference_mode="remote")
        self._episode: Optional[Episode] = None

    def _check_episode_started(self) -> TypeGuard[Episode]:
        if self._episode is None:
            raise HTTPException(
                status_code=412, detail="Cannot call this method before starting an episode. Call /start-episode first."
            )
        return True

    def _persist_action(self, observation: Observation, action: Action) -> None:
        self._check_episode_started()
        assert isinstance(self._episode, Episode)  # Make mypy happy
        os.makedirs(self._DATA_DIR, exist_ok=True)

        df = pd.DataFrame({"episode_id": self._episode.episode_id, **observation.dict(), "action": action}, index=[0])
        self._log.debug(df.iloc[0].to_json())

        file = self._DATA_DIR / "consumption_action.csv"
        header = False if os.path.exists(file) else True
        df.to_csv(file, mode="a", index=False, header=header)

    def _persist_reward(self, reward: float, observation: Observation, last_action: Optional[Action]) -> None:
        self._check_episode_started()
        assert isinstance(self._episode, Episode)  # Make mypy happy
        os.makedirs(self._DATA_DIR, exist_ok=True)

        df = pd.DataFrame(
            {
                "episode_id": self._episode.episode_id,
                "reward": reward,
                **observation.dict(),
                "last_action": last_action,
            },
            index=[0],
        )
        self._log.debug(df.iloc[0].to_json())

        file = self._DATA_DIR / "consumption_reward.csv"
        header = False if os.path.exists(file) else True
        df.to_csv(file, mode="a", index=False, header=header)

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
        assert isinstance(self._episode, Episode)  # Make mypy happy

        action_id = self._policy_client.get_action(self._episode.episode_id, request.observation.to_feature_vector())
        action = Action(action_id)
        self._log.info(f"Algorithm predicted action={action}. Persisting to .csv file ...")
        self._persist_action(request.observation, action)

        return ActionResponse(action=action)

    @fastapi_app.post("/log-returns")
    def log_returns(self, request: LogReturnsRequest) -> None:
        self._log.debug(f"Called log_returns with {request}.")
        self._check_episode_started()
        assert isinstance(self._episode, Episode)  # Make mypy happy

        self._log.debug("Persisting reward to .csv file ...")
        self._persist_reward(request.reward, request.observation, request.last_action)

        self._policy_client.log_returns(self._episode.episode_id, request.reward)

    @fastapi_app.post("/end-episode")
    def end_episode(self, request: EndEpisodeRequest) -> None:
        self._log.debug(f"Called end_episode with {request}.")
        self._check_episode_started()
        assert isinstance(self._episode, Episode)  # Make mypy happy

        self._policy_client.end_episode(self._episode.episode_id, request.observation.to_feature_vector())
        self._episode = None
