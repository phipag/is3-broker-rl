import json
import logging
import os
import pwd
from cmath import e
from pathlib import Path
from posixpath import split
from typing import Optional
import math

import dotenv
import numpy as np
import pandas as pd
from fastapi import HTTPException
from ray import serve
from ray.rllib.env import PolicyClient
from starlette.requests import Request

from is3_broker_rl.api.fastapi_app import fastapi_app
from is3_broker_rl.api.wholesale_dto import (
    EndEpisodeRequest,
    Episode,
    GetActionRequest,
    LogReturnsRequest,
    Observation,
    StartEpisodeRequest,
)
from is3_broker_rl.conf import setup_logging
from is3_broker_rl.model.wholesale_policy_server import SERVER_ADDRESS, SERVER_BASE_PORT


@serve.deployment(route_prefix="/wholesale")
@serve.ingress(fastapi_app)
class WholesaleController:
    last_obs: np.ndarray

    def __init__(self) -> None:
        # This runs in a Ray actor worker process, so we have to initialize the logging again
        self.last_action_str = ""
        dotenv.load_dotenv(override=False)
        setup_logging("is3_wholesale_rl.log")
        self._log = logging.getLogger(__name__)
        self.obs_dict = {}
        self._log.debug(f"Policy client actor using environment variables: {os.environ}")
        self._DATA_DIR: Path = Path(os.environ.get("DATA_DIR", "data/"))
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
        self._log.debug(f"Called start_episode with {request}.")
        try:
            episode_id = self._policy_client.start_episode(training_enabled=True)
        except Exception:
            episode_id = self._policy_client.start_episode(training_enabled=True)
        self._episode = Episode(episode_id=episode_id)
        self._log.info(f"Started new episode with episode_id={episode_id}.")
        self.finished_observation = False
        return self._episode

    @fastapi_app.post("/get-action")
    def get_action(self, request: GetActionRequest):
        try:
            self._check_episode_started()
            self.finished_observation = False
            # Own cleared trades
            self.last_obs.cleared_orders_price = self.string_to_list(request.cleared_orders_price)
            self.last_obs.cleared_orders_energy = self.string_to_list(request.cleared_orders_energy)
            # Market prices
            self.last_obs.cleared_trade_price = self.string_to_list(request.cleared_trade_price)
            self.last_obs.cleared_trade_energy = self.string_to_list(request.cleared_trade_energy)
            self.last_obs.customer_count = request.customer_count
            self.last_obs.total_prosumption = float(request.total_prosumption)
            self.last_obs.market_position = self.string_to_list(request.market_position)

            # TODO: Preprocess obs:
            obs = self._standardize_observation(self.last_obs)
            action = self._policy_client.get_action(self._episode.episode_id, obs.to_feature_vector())
            #self._log.debug(f"Action: {action}")
            self.last_action = action
            # Transforms the action space from [-150:150] for the energy.
            # Transform the action space from [-1:1] to [0:100] for the price.
            # The sign is applied.
            #  See powertac game specification.
            action_scaled = np.zeros((48))
            for i in range(48):
                if i % 2 == 0:
                    action_scaled[i] = action[i] * 50
                    temp_action = action_scaled[i]
                else:
                    if temp_action < 0:
                        sign = 1
                    else:
                        sign = -1
                    action_scaled[i] = ((action[i] * 50) + 50) * sign

            #self._log.info(f"Algorithm predicted action={action_scaled}. Persisting to .csv file ...")
            return_string = ""

            for act in action_scaled:
                act1 = str(act)
                return_string = return_string + ";" + act1

            self._persist_action(return_string)
            return return_string
        except Exception as e:
            self._log.error(f"Get Action error: {e}", exc_info=True)
            return ""

    @fastapi_app.post("/log-returns")
    def log_returns(self, request: LogReturnsRequest) -> None:
        try:
            reward = request.reward / 100000
            balancing_reward = request.balancing_reward / 100000
            wholesale_reward =request.wholesale_reward / 100000
            tariff_reward = request.tariff_reward / 100000
            # Adding reward shaping with the difference between the market prices and our prices.
            # Percentage difference between actual and our prices?
            sum_mWh = request.sum_mWh
            final_market_balance = request.final_market_balance
            i=0
            
            shaped_return = abs( final_market_balance - sum_mWh) / -100
            shaped_return2 = abs( final_market_balance - (self.last_obs.p_customer_prosumption[0]/1000))
            
            final_reward = shaped_return2  #shaped_return#balancing_reward + wholesale_reward #+ tariff_reward
            self._log.info(f"Only shaped_reward2: {shaped_return}, mWh {sum_mWh}, mb {final_market_balance}")
            #reward = reward + shaped_return
            self._log.info(f"Called log_returns with {final_reward}.")
            self._check_episode_started()

            self._persist_reward(final_reward, balancing_reward, wholesale_reward, tariff_reward, shaped_return, sum_mWh)
            self._policy_client.log_returns(self._episode.episode_id, final_reward)
        except Exception as e:
            self._log.error(f"Log reward error: {e}", exc_info=True)
            return ""

    @fastapi_app.post("/end-episode")
    def end_episode(self, request: EndEpisodeRequest) -> None:
        self._log.debug(f"Called end_episode with {request}.")
        self._check_episode_started()
        self.finished_observation = False

        self.last_obs.cleared_orders_price = self.string_to_list(request.cleared_orders_price)
        self.last_obs.cleared_orders_energy = self.string_to_list(request.cleared_orders_energy)
        self.last_obs.cleared_trade_price = self.string_to_list(request.cleared_trade_price)
        self.last_obs.cleared_trade_energy = self.string_to_list(request.cleared_trade_energy)
        self.last_obs.customer_count = request.customer_count
        self.last_obs.total_prosumption = float(request.total_prosumption)
        self.last_obs.market_position = self.string_to_list(request.market_position)
        obs = self._standardize_observation(self.last_obs)
        self._policy_client.end_episode(self._episode.episode_id, obs.to_feature_vector())
        # self.last_action_str = ""
        # self.last_obs = None

    @fastapi_app.post("/build_observation")
    def build_observation(self, request: Request) -> None:
        try:
            obs = request.query_params["obs"]
            obs = np.array(json.loads(obs))
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
                cleared_orders_price=[0] * 24,  # Inputs empty values. These will be filled later.
                cleared_orders_energy=[0] * 24,  # Inputs empty values. These will be filled later.
                cleared_trade_price=[0] * 24,  # Inputs empty values. These will be filled later.
                cleared_trade_energy=[0] * 24,  # Inputs empty values. These will be filled later.
                customer_count=0,
                total_prosumption=float(0),
                hour_of_day=obs[144:168].tolist(),
                day_of_week=obs[168:175].tolist(),
                market_position=[0] * 24,
            )

            self.finished_observation = True
        except Exception as e:
            self._log.error(f"Observation building error: {e}", exc_info=True)

    def _persist_action(self, action) -> None:
        try:
            self._check_episode_started()
            assert isinstance(self._episode, Episode)  # Make mypy happy
            os.makedirs(self._DATA_DIR, exist_ok=True)
            self.last_action_str = action
            observation = self.last_obs
            df = pd.DataFrame(
                {"episode_id": self._episode.episode_id, "observation": observation.json(), "action": action}, index=[0]
            )
            self._log.debug(df.iloc[0].to_json())

            file = self._DATA_DIR / "wholesale_action.csv"
            header = False if os.path.exists(file) else True
            df.to_csv(file, mode="a", index=False, header=header)
        except Exception as e:
            self._log.debug(f"Persist action error {e}", exc_info=True)

    def _persist_reward(self, reward: float, balancing_reward: float, wholesale_reward: float, tariff_reward: float, shaped_return: float, sum_mWh: float) -> None:
        self._check_episode_started()
        assert isinstance(self._episode, Episode)  # Make mypy happy
        observation = self.last_obs.json()
        action = self.last_action_str
        os.makedirs(self._DATA_DIR, exist_ok=True)

        df = pd.DataFrame(
            {
                "episode_id": self._episode.episode_id,
                "reward": reward,
                "balancing_reward":balancing_reward,
                "wholesale_reward":wholesale_reward,
                "tariff_reward":tariff_reward,
                "shaped_return": shaped_return,
                "observation": observation,
                "last_action": action,
                "sum_mWh": sum_mWh,
            },
            index=[0],
        )
        self._log.debug(df.iloc[0].to_json())

        file = self._DATA_DIR / "wholesale_reward.csv"
        header = False if os.path.exists(file) else True
        df.to_csv(file, mode="a", index=False, header=header)

    def _standardize_observation(self, obs: Observation):
        mean = {
            0: 13.63367803668246,  # temperature
            1: 2.8749491311183646,  # windspeed
            2: 0.3507285267995091,  # cloudCover
            3: -5562.860502868995,  # gridImbalance
            4: 67.94300403314354,  # wholesale_price t0- t23
            5: 52.525031707832994,
            6: 52.88933002824563,
            7: 53.6404562916743,
            8: 54.242977901342435,
            9: 54.64121266757686,
            10: 55.68414875394494,
            11: 55.95338863257504,
            12: 56.500170033661576,
            13: 56.86790772811706,
            14: 57.26890408075484,
            15: 57.546169037693495,
            16: 58.32161144972453,
            17: 58.325472360126476,
            18: 58.860712114852674,
            19: 58.95012248169707,
            20: 58.56377617684579,
            21: 57.7975016266282,
            22: 56.76283333858017,
            23: 55.84977154330559,
            24: 54.361149855941854,
            25: 53.6202408477646,
            26: 53.19566381723961,
            27: 88.24426167683156,
            28: -6204.5187837914200000,  # enc_customer_prosumption
        }

        std = {
            0: 12.991492001148853,
            1: 1.9648730336774647,
            2: 0.3178986946278265,
            3: 21020.252493759675,
            4: 97.7488901733044,
            5: 81.90114882212681,
            6: 80.93475336382566,
            7: 81.15237969890913,
            8: 81.0773639240314,
            9: 81.15059495302327,
            10: 81.56561194539107,
            11: 82.21001272330156,
            12: 82.68724464729176,
            13: 83.08028951245949,
            14: 83.35342675797074,
            15: 83.33694606058906,
            16: 83.41922174065809,
            17: 83.64445398567099,
            18: 83.52051975557954,
            19: 83.4591317857155,
            20: 82.94899235919999,
            21: 82.21190741907145,
            22: 81.76050603123286,
            23: 80.62699046407442,
            24: 80.07524192939118,
            25: 79.12346949638426,
            26: 78.50764659516479,
            27: 102.93501402012687,
            28: 10273.1983911710890000,  # enc_customer_prosumption
        }
        try:
            wholesale_price = []
            i = 4
            for x in obs.p_wholesale_price:

                wholesale_price.append((x - mean[i]) / std[i])
                i += 1

            self._log.debug(f"Scaled wholesale: {wholesale_price}")

            scaled_obs = Observation(
                gameId=obs.gameId,
                timeslot=obs.timeslot,
                p_temperature=[((x - mean[0]) / std[0]) for x in obs.p_temperature],
                p_wind_speed=[((x - mean[1]) / std[1]) for x in obs.p_wind_speed],
                p_cloud_cover=[((x - mean[2]) / std[2]) for x in obs.p_cloud_cover],
                p_grid_imbalance=[((x - mean[3]) / std[3]) for x in obs.p_grid_imbalance],
                p_customer_prosumption=[((x - mean[28]) / std[28]) for x in obs.p_customer_prosumption],
                p_wholesale_price=wholesale_price,
                hour_of_day=obs.hour_of_day,
                day_of_week=obs.day_of_week,
                cleared_orders_energy=obs.cleared_orders_energy,
                cleared_orders_price=obs.cleared_orders_price,
                cleared_trade_energy=obs.cleared_trade_energy,
                cleared_trade_price=obs.cleared_trade_price,
                customer_count=obs.customer_count,
                total_prosumption=obs.total_prosumption,
                market_position=obs.market_position,
            )
            x = scaled_obs.total_prosumption
            
            self._log.debug(f"Scaled Obs: {scaled_obs}")
        except Exception as e:
            self._log.debug(f"Scaling obs error {e}", exc_info=True)

        return scaled_obs

    def string_to_list(self, input_string: str, delimeter=";"):

        splits = input_string.split(delimeter)
        # self._log.info(f"Splits {splits}")
        return_list = []
        for value in splits:
            if value != "":
                return_list.append(float(value))

        return return_list
