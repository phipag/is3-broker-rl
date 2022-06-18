import logging
import sys
from typing import Optional
from urllib.request import Request
from ray import serve
import dotenv
from fastapi import HTTPException
from ray import serve
from ray.rllib.env import PolicyClient
import json
import numpy as np
from starlette.requests import Request
from is3_broker_rl.api.wholesale_dto import (
    Action,
    ActionResponse,
    EndEpisodeRequest,
    Episode,
    GetActionRequest,
    LogReturnsRequest,
    StartEpisodeRequest,
    Observation,
)
from is3_broker_rl.api.fastapi_app import fastapi_app
from is3_broker_rl.conf import setup_logging
from is3_broker_rl.model.wholesale_policy_server import (
    SERVER_ADDRESS,
    SERVER_BASE_PORT,
)


@serve.deployment(route_prefix="/wholesale")
@serve.ingress(fastapi_app)
class WholesaleController:
    last_obs: np.ndarray

    def __init__(self) -> None:
        # This runs in a Ray actor worker process, so we have to initialize the logging again
        dotenv.load_dotenv(override=False)
        setup_logging()
        self._log = logging.getLogger(__name__)
        self.obs_dict = {}

        self._policy_client = PolicyClient(f"http://{SERVER_ADDRESS}:{SERVER_BASE_PORT}", inference_mode="remote")
        self._episode: Optional[Episode] = None
        self._log.info("Wholesale init done.")


    def _check_episode_started(self):
        if not self._episode:
            raise HTTPException(
                status_code=412, detail="Cannot call this method before starting an episode. Call /start-episode first."
            )


    @fastapi_app.post("/start-episode")
    def start_episode(self, request: StartEpisodeRequest) -> Episode:
        #self._log.info(f"1Started new episode with episode_id={episode_id}.")
        self._log.debug(f"Called start_episode with {request}.")
        try:
            episode_id = self._policy_client.start_episode(training_enabled=request.training_enabled)
        except:
            episode_id = self._policy_client.start_episode(training_enabled=True)
        self._episode = Episode(episode_id=episode_id)
        self._log.info(f"Started new episode with episode_id={episode_id}.")
        self.finished_observation = False
        return self._episode


    @fastapi_app.post("/get-action")
    def get_action(self, request: GetActionRequest):
        self._log.debug(f"Test3, {self.last_obs}")
        try:
            feature_vector = self.last_obs.to_feature_vector()
        except Exception as e:
            self._log.error(e)
        self._log.debug(f"Called get_action with {request}. Feature vector: {feature_vector}.")
        self._check_episode_started()
        self.finished_observation = False
        action = self._policy_client.get_action(self._episode.episode_id, self.last_obs.to_feature_vector())
        self._log.info(f"Algorithm predicted action={action}. Persisting to .csv file ...")
        # TODO: Implement persistence to .csv file
        #self._log.info(str(action[0])+":"+ str(action[1]))
        return_string = ""
        for act in action:
            return_string = return_string + ":" +str(act)
        return return_string
        




    @fastapi_app.post("/log-returns")
    def log_returns(self, request: LogReturnsRequest) -> None:
        self._log.debug(f"Called log_returns with {request}.")
        self._check_episode_started()
        

        self._policy_client.log_returns(self._episode.episode_id, request.reward)


    @fastapi_app.post("/end-episode")
    def end_episode(self, request: EndEpisodeRequest) -> None:
        self._log.debug(f"Called end_episode with {request}.")
        self.finished_observation = False
        self._policy_client.end_episode(self._episode.episode_id, request.observation.to_feature_vector())


    @fastapi_app.post("/build_observation")
    def build_observation(self, request: Request) -> None:

        obs = request.query_params["obs"]
        obs = np.array(json.loads(obs))
        shapeof = np.shape(obs)
        typeof = type(obs)
        self._log.debug(f"Obs: {shapeof}, type {typeof}")
        timeslot = request.query_params["timeslot"]
        game_id = request.query_params["game_id"]
        self._log.info("Observation received")
        
        self.last_obs = Observation(
            gameId=game_id,
            timeslot=timeslot,
            p_grid_imbalance=obs[0:24].tolist(),
            p_customer_prosumption=obs[24:48].tolist(),
            p_wholesale_price=obs[48:72].tolist(),
            p_cloud_cover=obs[72:96].tolist(),
            p_temperature=obs[96:120].tolist(),
            p_wind_speed=obs[120:144].tolist(),
            p_wind_direction=obs[144:168].tolist(),
            hour_of_day=obs[168:192].tolist(),
            day_of_week=obs[192:199].tolist(),
        )
        self._log.debug(self.last_obs.p_wholesale_price)
        feature_vector_test = self.last_obs.to_feature_vector()
        self._log.info(f"Building observation with: {obs}")
        self._log.debug(f"Testing feature vector: {feature_vector_test}")
        self.finished_observation = True

    @fastapi_app.post("/test")
    def test(self,request: Request):
        self._log.info("Test")