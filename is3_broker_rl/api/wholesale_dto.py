"""
This file contains the Data Transfer Objects (DTOs) used for API communication with the Java broker.
"""
from typing import List
import dotenv

import numpy as np
from pydantic import BaseModel
from is3_broker_rl.conf import setup_logging
import logging
from pathlib import Path
import os


class Action(BaseModel):
    price: float
    energy: float

    def Action(self, price, energy):
        self.price = price
        self.energy = energy


class ActionResponse(Action):
    action: Action


class Observation(BaseModel):
    gameId: str
    timeslot: int
    p_grid_imbalance: List[float] = []
    p_customer_prosumption: List[float] = []
    p_wholesale_price: List[float] = []
    p_cloud_cover: List[float] = []
    p_temperature: List[float] = []
    p_wind_speed: List[float] = []
    cleared_orders_price: List[float] = []
    cleared_orders_energy: List[float] = []
    cleared_trade_price: List[float] = []
    cleared_trade_energy: List[float] = []
    customer_count: float
    customer_change: float
    total_prosumption: float
    market_position: List[float] = []
    percentageSubs: List[float] = []
    prosumptionPerGroup: List[float] = []
    needed_mWh: List[float] = []
    hour_of_day: List[float] = []
    day_of_week: List[float] = []
    action_history: List[List[float]] = [[]]
    unclearedOrdersMWhAsks: List[float] = []
    unclearedOrdersMWhBids: List[float] = []
    weigthedAvgPriceAsks: List[float] = []
    weigthedAvgPriceBids: List[float] = []

    # remember to change the observation space in the wholsesale_util.py ;)

    def to_feature_vector(self, time_diff: int):
        
        full_state_space = True
        time_diff_list = [0]*24
        time_diff_list[time_diff] = 1
        if len(self.p_cloud_cover) != 24:
            self.p_cloud_cover = [0]*24
            print(f"not long enough: {self.p_cloud_cover}")
        if len(self.p_temperature) != 24:
            self.p_temperature = [0]*24
            print(f"not long enough: {self.p_temperature}")
        if len(self.p_wind_speed) != 24:
            self.p_wind_speed = [0]*24
            print(f"not long enough: {self.p_wind_speed}")

        if full_state_space != True:
            if time_diff == 23:
                market_position = 0
                unclearedOrdersMWhAsks = 0
                unclearedOrdersMWhBids = 0
                weigthedAvgPriceAsks = 0
                weigthedAvgPriceBids = 0
            else:
                market_position = self.market_position[time_diff+1]
                unclearedOrdersMWhAsks = self.unclearedOrdersMWhAsks[time_diff+1]
                unclearedOrdersMWhBids = self.unclearedOrdersMWhBids[time_diff+1]
                weigthedAvgPriceAsks = self.weigthedAvgPriceAsks[time_diff+1]
                weigthedAvgPriceBids = self.weigthedAvgPriceBids[time_diff+1]

        

            
        


            return np.concatenate(
                (
                    np.array([self.p_grid_imbalance[time_diff]]),
                    np.array([self.p_customer_prosumption[time_diff]]),
                    np.array([self.p_wholesale_price[time_diff]]),
                    np.array([self.p_cloud_cover[time_diff]]),
                    np.array([self.p_temperature[time_diff]]),
                    np.array([self.p_wind_speed[time_diff]]),
                    np.array([self.cleared_orders_price[time_diff]]),
                    np.array([self.cleared_orders_energy[time_diff]]),
                    np.array([self.cleared_trade_price[time_diff]]),
                    np.array([self.cleared_trade_energy[time_diff]]),
                    #np.array([self.customer_count]),#
                    #np.array([self.customer_change]),
                    np.array([self.total_prosumption]),
                    np.array([market_position]),
                    np.array(self.percentageSubs),
                    np.array(self.prosumptionPerGroup),
                    np.array([self.needed_mWh[time_diff]]),
                    np.array(self.hour_of_day),
                    np.array(self.day_of_week),
                    np.array([self.timeslot]),
                    np.array(time_diff_list),
                    np.array(np.ravel(self.action_history[time_diff])),
                    np.array([unclearedOrdersMWhAsks]),
                    np.array([unclearedOrdersMWhBids]),
                    np.array([weigthedAvgPriceAsks]),
                    np.array([weigthedAvgPriceBids]),

                ))
        else:
            return np.concatenate(
                (
                    np.array(self.p_grid_imbalance),
                    np.array(self.p_customer_prosumption),
                    np.array(self.p_wholesale_price),
                    np.array(self.p_cloud_cover),
                    np.array(self.p_temperature),
                    np.array(self.p_wind_speed),
                    np.array(self.cleared_orders_price),
                    np.array(self.cleared_orders_energy),
                    np.array(self.cleared_trade_price),
                    np.array(self.cleared_trade_energy),
                    #np.array([self.customer_count]),#
                    #np.array([self.customer_change]),
                    np.array([self.total_prosumption]),
                    np.array(self.market_position),
                    np.array(self.percentageSubs),
                    np.array(self.prosumptionPerGroup),
                    np.array(self.needed_mWh),
                    np.array(self.hour_of_day),
                    np.array(self.day_of_week),
                    np.array([self.timeslot]),
                    np.array(time_diff_list),
                    np.array(np.ravel(self.action_history[time_diff])),
                    np.array(self.unclearedOrdersMWhAsks),
                    np.array(self.unclearedOrdersMWhBids),
                    np.array(self.weigthedAvgPriceAsks),
                    np.array(self.weigthedAvgPriceBids),

                )

            )
    # For prediciton only:
    def to_prediction_vector(self):
        return np.concatenate(
            (
                np.array(self.p_grid_imbalance),
                np.array(self.p_customer_prosumption),
                np.array(self.p_wholesale_price),
                np.array(self.p_cloud_cover),
                np.array(self.p_temperature),
                np.array(self.p_wind_speed),
                np.array(self.cleared_orders_price),
                np.array(self.cleared_orders_energy),
                np.array(self.cleared_trade_price),
                np.array(self.cleared_trade_energy),
                np.array([self.customer_count]),
                np.array([self.customer_change]),
                np.array([self.total_prosumption]),
                np.array(self.market_position),
                np.array(self.percentageSubs),
                np.array(self.prosumptionPerGroup),
                #np.array(self.needed_mWh),
                np.array(self.hour_of_day),
                np.array(self.day_of_week),
            )
        )


class Episode(BaseModel):
    episode_id: str


class StartEpisodeRequest(BaseModel):
    training_enabled: bool = True


class EndEpisodeRequest(BaseModel):
    episode_id: str
    game_id: str
    timeslot: int
    cleared_orders_price: str
    cleared_orders_energy: str
    cleared_trade_price: str
    cleared_trade_energy: str
    customer_count: int
    total_prosumption: str
    market_position: str
    needed_mWh: str


class LogReturnsRequest(BaseModel):
    episode_id: str
    reward: float
    balancing_reward: float
    wholesale_reward: float
    tariff_reward: float
    sum_mWh: float
    final_market_balance: float


class GetActionRequest(BaseModel):
    episode_id: str
    game_id: str
    timeslot: int
    cleared_orders_price: str
    cleared_orders_energy: str
    cleared_trade_price: str
    cleared_trade_energy: str
    customer_count: int
    total_prosumption: str
    market_position: str
    needed_mWh: str
    unclearedOrdersMWhAsks: str
    unclearedOrdersMWhBids: str
    weigthedAvgPriceAsks: str
    weigthedAvgPriceBids: str


class ProsumptionRequest(BaseModel):
    gameId: str
    timeslot: int
    groupName: int
    prosumption: dict
    percentageSubs: str
