"""
This file contains the Data Transfer Objects (DTOs) used for API communication with the Java broker.
"""
from typing import List

import numpy as np
from pydantic import BaseModel


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
    hour_of_day: List[float] = []
    day_of_week: List[float] = []

    def to_feature_vector(self):
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


class LogReturnsRequest(BaseModel):
    episode_id: str
    reward: float


class GetActionRequest(BaseModel):
    episode_id: str
    game_id: str
    timeslot: int
    cleared_orders_price: str
    cleared_orders_energy: str
