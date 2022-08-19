import json
import logging
import os
import pickle
import pwd
from cmath import e
from pathlib import Path
from posixpath import split
import time
from typing import Optional
import math
from black import diff
from sklearn.ensemble import RandomForestRegressor

import dotenv
import numpy as np
import pandas as pd
from fastapi import HTTPException
from ray import serve
from ray.rllib.env import PolicyClient
import requests
from starlette.requests import Request
from tqdm import tqdm

from is3_broker_rl.api.fastapi_app import fastapi_app
from is3_broker_rl.api.wholesale_dto import (
    EndEpisodeRequest,
    Episode,
    GetActionRequest,
    LogReturnsRequest,
    Observation,
    StartEpisodeRequest,
    ProsumptionRequest
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
        self._DATA_DIR: Path = Path(os.environ.get("DATA_DIR", "data/"))
        setup_logging("is3_wholesale_rl.log")
        self._log = logging.getLogger(__name__)
        self._log.info(f"Starting the wholesale controller.")
        self.obs_dict = {}
        self.cc_change = np.zeros((24))
        self.temp_obs = []

        # Also loads the model if false
        self.save_model = False
        self.load_bootstrap_dataset("wholesale_reward_16_08.csv")
        self.cc_i = 0
        self.last_pred = []
        self.hist_sum_mWh = np.zeros((48))
        self.time_i = 0
        self.percentageSubs = np.zeros((20))
        self.prosumptionPerGroup = np.zeros((20))
        self._log.debug(f"Policy client actor using environment variables: {os.environ}")
        
        self._policy_client = PolicyClient(f"http://{SERVER_ADDRESS}:{SERVER_BASE_PORT}", inference_mode="remote")
        self._episode: Optional[Episode] = None
        self._log.info("Wholesale init done.")
        #self.bootstrap_action("wholesale_reward.csv")

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
            self.last_obs.needed_mWh = self.string_to_list(request.needed_mWh)

            # Set the change of customers over 24h
            
            self.last_obs.customer_change = request.customer_count - self.cc_change[self.cc_i]
            #self._log.info(f"customer_count {request.customer_count}, {self.cc_change[self.cc_i]}")
            self.cc_change[self.cc_i] = request.customer_count
            self.cc_i += 1
            # Loop back to one so it visits the right value after 24h.
            if self.cc_i >= 23:
                self.cc_i = 0
            # Preprocess obs:
            pred = self.predict_sum_mWh_diff(self.last_obs)
            pred_list = []
            for x in pred:
                #self._log.info(f"pred x {x}")
                pred_list.append(float(x))
            self._log.info(f"pred {pred_list}")
            #self._log.info(f"obs before {self.last_obs}")
            self.last_obs.p_customer_prosumption = pred_list
            #self._log.info(f"obs after {self.last_obs}")
            obs = self._standardize_observation(self.last_obs)
            action = self._policy_client.get_action(self._episode.episode_id, obs.to_feature_vector())
            #self._log.debug(f"Action: {action}")
            # just to save the raw action for easier access later.
            self.raw_action= ""
            for act in action:
                act1 = str(act)
                self.raw_action = self.raw_action + ";" + act1
            # Transforms the action space from [-50:50] for the energy.
            # Transform the action space from [-1:1] to [0:100] for the price.
            # The sign is applied.
            #  See powertac game specification.
            #action_scaled = np.zeros((48))
            #for i in range(48):
            #    if i % 2 == 0:
            #        action_scaled[i] = action[i] * 50
            #        temp_action = action_scaled[i]
            #    else:
            #        if temp_action < 0:
            #            sign = 1
            #        else:
            #            sign = -1
            #        action_scaled[i] = ((action[i] * 50) + 50) * sign
            action_scaled = np.zeros((48))

            for i in range(48):
                if i % 2 == 0:
                    # Take the difference between the predicted prosumption and the last market position.
                    if i ==46: # For the first bid there is no last market position.
                        market_position = 0
                    else:
                        
                        # market position of the timeslot before our bidding.
                        market_position  = self.last_obs.market_position[int((i/2))+1]
                    # Abs here because the action should decide whether to buy or sell.
                    
                    action_scaled[i] = abs(self.last_obs.needed_mWh[int(i/2)]  - market_position) * action[int(i/2)]
                    
                    if self.last_obs.p_customer_prosumption[int(i/2)]  < market_position:
                        temp_action = action_scaled[i] * -1
                    else:
                        temp_action = action_scaled[i]
                else:
                    # Reverse the sign of the action. Else we gift energy to the market or get no trades.
                    if temp_action < 0:
                        sign = 1
                    else:
                        sign = -1
                    price = self.last_obs.p_wholesale_price[int((i-1)/2)]
                    # Just check if price is negative to help learning. 
                    # TODO: Check later how often it actually is negative.
                    if price < 0:
                        #sign = -1
                        price = 1
                    # Action ranges from -1 to 1.
                    # Adding +1 results in a price distribution from 0% to 200%.
                    action_scaled[i] = ((1+ action[int((i-1)/2)]) * price) * sign

            self._log.info(f"Algorithm predicted action={action_scaled}. Persisting to .csv file ...")
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
            # Save last 48 hours of prosumption for the prediction. 
            # Needs to be 48h to give 24h of data for the prediction of t24.
            self.hist_sum_mWh[self.time_i] = sum_mWh
            self.time_i += 1
            if self.time_i >= 47:
                self.time_i = 0

            shaped_return = abs( final_market_balance - sum_mWh) / -100
            #shaped_return2 = abs( final_market_balance - (self.last_obs.p_customer_prosumption[0]/1000)) * -1
            
            #final_reward = balancing_reward + wholesale_reward #+ tariff_reward
            #self._log.info(f"Only shaped_reward: {shaped_return}, mWh {sum_mWh}, mb {final_market_balance}")
            final_reward = shaped_return
            self._log.info(f"Called log_returns with {final_reward}.")
            self._check_episode_started()
            self.train_sum_mWh_diff(self.last_obs, sum_mWh)
            self._persist_reward(final_reward, balancing_reward, wholesale_reward, tariff_reward, final_market_balance, sum_mWh)#, final_market_balance)
            self._policy_client.log_returns(self._episode.episode_id, final_reward)
        except Exception as e:
            self._log.error(f"Log reward error: {e}", exc_info=True)
            return ""

    @fastapi_app.post("/end-episode")
    def end_episode(self, request: EndEpisodeRequest) -> None:
        try:
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
            self.last_obs.needed_mWh = self.string_to_list(request.needed_mWh)
            # Set the change of customers over 24h
            self.last_obs.customer_change = self.cc_change[self.cc_i]
            #
            self.cc_change[self.cc_i] = request.customer_count
            
            
            obs = self._standardize_observation(self.last_obs)
            self._policy_client.end_episode(self._episode.episode_id, obs.to_feature_vector())
            self._episode = None
            # self.last_action_str = ""
            # self.last_obs = None
            # TODO: save online prediction model:
            #if self.last_obs.timeslot % 500 == 0:
            #    os.makedirs(self._DATA_DIR, exist_ok=True)
            #    for diff_i in range(24):
            #        pkl_filename = self._DATA_DIR / f"prediction_{diff_i}.pkl"
            #        with open(pkl_filename, "wb") as file:
            #            pickle.dump(self.rf[diff_i], file)

        except Exception as e:
            self._log.error(f"Observation building error: {e}", exc_info=True)

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
                customer_change=0,
                total_prosumption=float(0),
                needed_mWh=[0] * 24,
                hour_of_day=obs[144:168].tolist(),
                day_of_week=obs[168:175].tolist(),
                market_position=[0] * 24,
                percentageSubs=self.percentageSubs.tolist(),
                prosumptionPerGroup=self.prosumptionPerGroup.tolist(),
            )
            #self.last_obs.p_customer_prosumption = self.last_obs.p_customer_prosumption /1000

            

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
            #df = pd.DataFrame(
            #    {"episode_id": self._episode.episode_id, "observation": observation.json(), "action": action}, index=[0]
            #)
            #self._log.debug(df.iloc[0].to_json())

            #file = self._DATA_DIR / "wholesale_action.csv"
            #header = False if os.path.exists(file) else True
            #df.to_csv(file, mode="a", index=False, header=header)
        except Exception as e:
            self._log.error(f"Persist action error {e}", exc_info=True)

    def _persist_reward(self, reward: float, balancing_reward: float, wholesale_reward: float, tariff_reward: float, 
                        shaped_return: float, sum_mWh: float):#, final_market_balance: float) -> None:
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
                "raw_action": self.raw_action,
                #"final_market_balance": final_market_balance,

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
                customer_change=obs.customer_change,
                total_prosumption=obs.total_prosumption,
                market_position=obs.market_position,
                percentageSubs=obs.percentageSubs,
                prosumptionPerGroup=obs.prosumptionPerGroup,
                needed_mWh=obs.needed_mWh,
            )
            x = scaled_obs.total_prosumption
            
            #self._log.debug(f"Scaled Obs: {scaled_obs}")
        except Exception as e:
            self._log.error(f"Scaling obs error {e}", exc_info=True)

        return scaled_obs

    def string_to_list(self, input_string: str, delimeter=";"):

        splits = input_string.split(delimeter)
        # self._log.info(f"Splits {splits}")
        return_list = []
        for value in splits:
            if value != "":
                return_list.append(float(value))

        return return_list


    # This function reads old logs and uses off-policy
    # learning to train the model and fill the replay buffer.
    # Can be turned on and off.
    def bootstrap_action(self, reward_csv_name):
        try:

            episode_id = self._policy_client.start_episode(training_enabled=True)
            self._episode = Episode(episode_id=episode_id)
            os.makedirs(self._DATA_DIR, exist_ok=True)
            file = self._DATA_DIR / reward_csv_name 
            
            df = pd.read_csv(file)
            self._log.info(f"{df.iloc[0]}")
            cc_i = 0
            cc_change = np.zeros((24))
            start = time.time()
            hist_sum_mWh = np.zeros((48))
            bootstrap_i = 0
            for index, row in df.iterrows():
                obs = json.loads(row["observation"])
                if obs.get("customer_change") == None:
                    self.last_obs = cc_change[cc_i]
                    # 
                    cc_change_value = cc_change[cc_i]
                    cc_i += 1
                    # Loop back to one so it visits the right value after 24h.
                    if cc_i > 23:
                        cc_i = 0

                else:
                    cc_change_value = obs.get("customer_change")
                
                
                obs = Observation(
                    gameId=obs.get("gameId"),
                    timeslot=obs.get("timeslot"),
                    p_grid_imbalance=obs.get("p_grid_imbalance"),
                    p_customer_prosumption=obs.get("p_customer_prosumption"),
                    p_wholesale_price=obs.get("p_wholesale_price"),
                    p_cloud_cover=obs.get("p_cloud_cover"),
                    p_temperature=obs.get("p_temperature"),
                    p_wind_speed=obs.get("p_wind_speed"),
                    cleared_orders_price=obs.get("cleared_orders_price"),
                    cleared_orders_energy=obs.get("cleared_orders_energy"),
                    cleared_trade_price=obs.get("cleared_trade_price"),
                    cleared_trade_energy=obs.get("cleared_trade_energy"),
                    customer_count=obs.get("customer_count"),
                    customer_change=cc_change_value,
                    total_prosumption=obs.get("total_prosumption"),
                    percentageSubs = obs.get("percentageSubs"),
                    prosumptionPerGroup = obs.get("prosumptionPerGroup"),
                    hour_of_day=obs.get("hour_of_day"),
                    day_of_week=obs.get("day_of_week"),
                    market_position=obs.get("market_position"),
                    needed_mWh=obs.get("needed_mWh"),
                )
                temp_prosumption = obs.p_customer_prosumption
                hist_sum_mWh[bootstrap_i] = row["sum_mWh"]
                for i in range(len(temp_prosumption)):

                    predict_vector = obs.to_prediction_vector()

                    # Find the correct value for the historic prosumption.

                    time_i = bootstrap_i - 24 - i
                    # Loop back to one so it visits the right value.
                    if time_i < 0:
                        time_i = 48 + time_i
                    temp_value = hist_sum_mWh[time_i]
                    predict_vector2 = np.append(predict_vector,temp_value)
                    temp_prosumption[i] = self.rf[i].predict(predict_vector2.reshape(1, -1))# + sum
                
                bootstrap_i +=1
                if bootstrap_i > 47:
                    bootstrap_i = 0
                obs.p_customer_prosumption = temp_prosumption
                obs = self._standardize_observation(obs)
                #self._log.info(f"Obs feature: {obs.to_feature_vector()}")
                #reward = row["reward"]
                reward = row["wholesale_reward"] + row["balancing_reward"]
                action_str = row["last_action"]
                raw_action_str = row["raw_action"]
               
                
                
                action = self.string_to_list(raw_action_str)
                

                #self._log.info(f"action: {action}")

                self._policy_client.log_action(self._episode.episode_id, obs.to_feature_vector(), action)

                    
                self._policy_client.log_returns(self._episode.episode_id, reward)
                
            elapsed_time_fl = (time.time() - start)
            self._log.info(f"Bootstrap finished. Time {elapsed_time_fl}")

            self._policy_client.end_episode(self._episode.episode_id, obs.to_feature_vector())
            self._episode = None

        except Exception as e:
            self._log.error(f"Bootstrap error {e}", exc_info=True)


        #self._policy_client.end_episode(self._episode.episode_id,obs)
        return


    @fastapi_app.post("/history")
    def log_customer_history(self, request: ProsumptionRequest) -> None:
        try:
            
            #prosumption = request.prosumption
            i = 0
            variable = request.__dict__
            timeslot = variable.get("timeslot")
            #self._log.info(f"timeslot {timeslot}")
            prosumption = list(variable.get("prosumption").values())[0]
            #self._log.info(f"Variable: {prosumption}")
            groupName = variable["groupName"]
            if groupName >= 0:
                self.percentageSubs[groupName] = float(variable.get("percentageSubs"))
                self.prosumptionPerGroup[groupName] = variable.get("prosumption").get(str(timeslot-1))
                if i>18:
                    i=0
                    #self.last_obs.percentageSubs = self.percentageSubs
                    #self.last_obs.prosumption = self.prosumption

                i+=1



                
 

        except Exception as e:
            self._log.error(f"prosumption_history error {e}", exc_info=True)




    def predict_sum_mWh_diff(self, obs: Observation):
        #Predicts the sum of the mWh difference between the prosumption and the predicted prosumption from the LSTM.
        #This adds online learning to the LSTM model.
        
        try:
            input_vector = obs.to_prediction_vector()
            #self._log.info(f"Input: {input}")
            pred = []
            for diff_i in range(0,24):
                # Find the correct value for the historic prosumption.

                time_i = self.time_i - 24 - diff_i -1
                # Loop back to one so it visits the right value.
                if time_i < 0:
                    time_i = 48 + time_i
                hist_sum_mWh = self.hist_sum_mWh[time_i]
                #self._log.info(input_vector)
                input_vector2 = np.append(input_vector,hist_sum_mWh)
                #self._log.info(input_vector2)
                pred.append(self.rf[diff_i].predict(input_vector2.reshape(1, -1)))
            self.last_pred = pred
            return pred
        except Exception as e:
            self._log.error(f"Predict random forrest error {e}", exc_info=True)
            return 0.0

    def train_sum_mWh_diff(self, obs: Observation, sum_mWh: float):
        try:
            # Add the observation to the training data.
            #self._log.info(f"obs {obs.to_prediction_vector()}")
            if self.last_obs == None:
                self._log.info("obs is none")
                return
            #    self.temp_obs = [obs.to_prediction_vector]
            #else:
            self.temp_obs.append(obs.to_prediction_vector())
            #self._log.info(len(self.temp_obs))
            if (len(self.temp_obs) >= 24):
                self.temp_obs.pop(0)
                
                # Train the model.
                y_true = np.array([sum_mWh]*24)# - np.array(obs.p_customer_prosumption)/1000
                # Only train the model if the obs is available.
                for diff_i in range(0,len(self.temp_obs)):
                    input_train = self.temp_obs[diff_i]
                    
                    # Find the correct value for the historic prosumption.

                    time_i = self.time_i - 24 - diff_i -1
                    # Loop back to one so it visits the right value.
                    if time_i < 0:
                        time_i = 48 + time_i
                    hist_sum_mWh = self.hist_sum_mWh[time_i]
                    input_train2 = np.append(input_train,hist_sum_mWh)
                    #self._log.info(input_train2)
                    #self._log.info(f"input_train {input_train}")
                    #self._log.info(f"input_train {type(input_train)}")
                    #self._log.info(f"y_true {y_true[diff_i]}")
                    
                    self.rf[diff_i].fit(input_train2.reshape(1, -1), y_true[diff_i].reshape(-1, 1), sample_weight=[10])


                mae = np.abs(np.array(self.last_pred) - np.array(y_true).reshape(-1, 1))
                self._log.info(f"Pred error: {mae}")
                os.makedirs(self._DATA_DIR, exist_ok=True)
                
                df = pd.DataFrame(
                    {
                        "timeslot": self.last_obs.timeslot,
                        "prediciton": self.last_pred[diff_i],
                        "true_value": y_true[diff_i],
                    },
                    index=[0],
                )

                file = self._DATA_DIR / "wholesale_prediction.csv"
                header = False if os.path.exists(file) else True
                df.to_csv(file, mode="a", index=False, header=header)

            
            
        except Exception as e:
            self._log.error(f"Train random forrest error {e}", exc_info=True)
            return 0.0



    def load_bootstrap_dataset(self, reward_csv_name="wholesale_reward.csv"):
        try:
            # Load the model if needed.	
            if self.save_model == False:
                self.rf = []
                for diff_i in range(0,24):
                    pkl_filename = self._DATA_DIR / f"prediction_{diff_i}.pkl"
                    
                    with open(pkl_filename, "rb") as file:
                        self.rf.append(pickle.load(file))
                    self._log.info(f"Loaded model {pkl_filename}")
                return

            self._log.info("Training started")
            start = time.time()
            os.makedirs(self._DATA_DIR, exist_ok=True)
            file = self._DATA_DIR / reward_csv_name 
            #self._log.info("Training started1")
            df = pd.read_csv(file)
            
            df.dropna(inplace=True) # Drop first few timesteps that have no action. 
            #self._log.info("Training started2")
            df_reward2 = df["observation"].apply(json.loads)
            col_names = list(df_reward2.iloc[0].keys())
            df_reward2 = df_reward2.apply(lambda x: list(x.values()))
            temp_list = []
            for row in df_reward2:

                temp_list.append(row)

            temp_df = pd.DataFrame(temp_list, columns=col_names)
            #self._log.info("Training started3")
            temp_merge_df = pd.DataFrame()
            temp_name = []
            for column in temp_df.columns:
                if type(temp_df[column].iloc[0]) == list:
                    temp_df2 = temp_df[column].apply(pd.Series)
                    i=0
                    for column2 in temp_df2:
                        temp_merge_df[f"{column}_{i}"] = temp_df2[column2]
                        i+=1
                        
                    
                else:
                    temp_merge_df[column] = temp_df[column]
            #self._log.info("Training started4")
            temp_merge_df["sum_mWh"] = df["sum_mWh"].astype(float)
            #self._log.info(temp_merge_df[f"p_customer_prosumption_{0}"])
            self.rf = []
            for diff_i in tqdm(range(0,24)):
                #temp_merge_df["sum_diff"] = temp_merge_df[f"p_customer_prosumption_{diff_i}"]/1000 - temp_merge_df["sum_mWh"]
                temp_merge_df["sum_diff"] = temp_merge_df["sum_mWh"]
                
                test = temp_merge_df["sum_mWh"].shift(-24 - diff_i -1)
                temp_merge_df["last_sum"] = test
                temp_merge_df.fillna(0, inplace=True)
                
                

                X = temp_merge_df[temp_merge_df.columns.difference(["gameId", "timeslot", "sum_diff", "sum_mWh"])]
                
                #y = df["sum_mWh"]
                y = temp_merge_df["sum_diff"].shift(diff_i+1).fillna(0)
                self.rf.append(RandomForestRegressor(warm_start=True, n_estimators=20, max_features="sqrt", n_jobs=-1))
                self.rf[diff_i].fit(X, y)
                # Save the model.
                pkl_filename = self._DATA_DIR / f"prediction_{diff_i}.pkl"
                if self.save_model == True:
                    os.makedirs(self._DATA_DIR, exist_ok=True)
                    with open(pkl_filename, "wb") as file:
                        pickle.dump(self.rf[diff_i], file)

                self._log.info(f"finished training run {diff_i}")
            end = time.time()
            self._log.info(f"Training finished. Time {end-start}")
        except Exception as e:
            self._log.error(f"Train random forrest error {e}", exc_info=True)


    
        

