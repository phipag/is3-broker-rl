import logging
import os
from pathlib import Path
from typing import Dict, Optional, Tuple

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
    PPFAction,
    Reward,
    StartEpisodeRequest,
    TariffRateAction,
)
from is3_broker_rl.api.fastapi_app import fastapi_app
from is3_broker_rl.conf import setup_logging
from is3_broker_rl.model.consumption_tariff_config import (
    SERVER_ADDRESS,
    SERVER_BASE_PORT,
    dqn_config,
)


@serve.deployment(route_prefix="/consumption-tariff")
@serve.ingress(fastapi_app)
class ConsumptionTariffController:
    _ACTION_ID_MAPPING: Dict[int, Tuple[TariffRateAction, PPFAction]] = {
        0: (TariffRateAction.NO_OP, 0),  # Since NO_OP does nothing we can drop different combinations with PPFAction
        1: (TariffRateAction.TRAILER, 0),
        2: (TariffRateAction.TRAILER, 2),
        3: (TariffRateAction.TRAILER, 4),
        4: (TariffRateAction.TRAILER, 6),
        5: (TariffRateAction.TRAILER, 8),
        6: (TariffRateAction.TRAILER, 10),
        7: (TariffRateAction.AVERAGE, 0),
        8: (TariffRateAction.AVERAGE, 2),
        9: (TariffRateAction.AVERAGE, 4),
        10: (TariffRateAction.AVERAGE, 6),
        11: (TariffRateAction.AVERAGE, 8),
        12: (TariffRateAction.AVERAGE, 10),
        13: (TariffRateAction.LEADER, 0),
        14: (TariffRateAction.LEADER, 2),
        15: (TariffRateAction.LEADER, 4),
        16: (TariffRateAction.LEADER, 6),
        17: (TariffRateAction.LEADER, 8),
        18: (TariffRateAction.LEADER, 10),
        19: (TariffRateAction.NEW_ITERATION, 0),
        20: (TariffRateAction.NEW_ITERATION, 2),
        21: (TariffRateAction.NEW_ITERATION, 4),
        22: (TariffRateAction.NEW_ITERATION, 6),
        23: (TariffRateAction.NEW_ITERATION, 8),
        24: (TariffRateAction.NEW_ITERATION, 10),
    }

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

        if len(self._ACTION_ID_MAPPING) != dqn_config["action_space"].n:
            msg = (
                f"The implemented action ID mapping does not match the expected length of the gym action space of the "
                f"RL model. Expected: {dqn_config['action_space'].n}, actual: {len(self._ACTION_ID_MAPPING)}"
            )
            self._log.error(msg)
            raise SystemError(msg)

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

        df = pd.DataFrame({"episode_id": self._episode.episode_id, **observation.dict(), **action.dict()}, index=[0])
        self._log.debug(df.iloc[0].to_json())

        file = self._DATA_DIR / "consumption_action.csv"
        header = False if os.path.exists(file) else True
        df.to_csv(file, mode="a", index=False, header=header)

    def _persist_reward(
        self, reward: float, reward_info: Reward, observation: Observation, last_action: Optional[Action]
    ) -> None:
        self._check_episode_started()
        assert isinstance(self._episode, Episode)  # Make mypy happy
        os.makedirs(self._DATA_DIR, exist_ok=True)

        df = pd.DataFrame(
            {
                "episode_id": self._episode.episode_id,
                "reward": reward,
                **reward_info.dict(),
                **observation.dict(),
                "last_tariff_rate_action": last_action.tariff_rate_action if last_action is not None else None,
                "last_ppf_action": last_action.ppf_action if last_action is not None else None,
            },
            index=[0],
        )
        self._log.debug(df.iloc[0].to_json())

        file = self._DATA_DIR / "consumption_reward.csv"
        header = False if os.path.exists(file) else True
        df.to_csv(file, mode="a", index=False, header=header)

    @classmethod
    def _get_tariff_rate_ppf_tuple(cls, action_id: int) -> Tuple[TariffRateAction, PPFAction]:
        """
        Translates the returned action_id which is a number between 0 and 24 to a unique tariff rate and ppf action
        tuple.
        :param action_id: Action id returned by the RL algorithm
        :return: Action tuple where first entry is the tariff rate action and the second entry the ppf action
        """
        # The validity of this access is checked in the constructor
        return cls._ACTION_ID_MAPPING[action_id]

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
        tariff_rate_action, ppf_action = self._get_tariff_rate_ppf_tuple(action_id)
        action = Action(tariff_rate_action=tariff_rate_action, ppf_action=ppf_action)
        self._log.info(
            f"Algorithm predicted action_id={action_id} which is action={action}. Persisting to .csv file ..."
        )
        self._persist_action(request.observation, action)

        return ActionResponse(action=action)

    @fastapi_app.post("/log-returns")
    def log_returns(self, request: LogReturnsRequest) -> None:
        self._log.debug(f"Called log_returns with {request}.")
        self._check_episode_started()
        assert isinstance(self._episode, Episode)  # Make mypy happy

        self._log.debug("Persisting reward to .csv file ...")
        self._persist_reward(request.reward, request.reward_info, request.observation, request.last_action)

        self._policy_client.log_returns(self._episode.episode_id, request.reward, info=request.reward_info.dict())

    @fastapi_app.post("/end-episode")
    def end_episode(self, request: EndEpisodeRequest) -> None:
        self._log.debug(f"Called end_episode with {request}.")
        self._check_episode_started()
        assert isinstance(self._episode, Episode)  # Make mypy happy

        self._policy_client.end_episode(self._episode.episode_id, request.observation.to_feature_vector())
        self._episode = None
