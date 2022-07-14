"""
This file contains the Data Transfer Objects (DTOs) used for API communication with the Java broker.
"""
import enum
from typing import List, Optional, Union

from pydantic import BaseModel


class Action(enum.IntEnum):
    NEW_ITERATION = 4
    LEADER = 3
    AVERAGE = 2
    TRAILER = 1
    NO_OP = 0


class ActionResponse(BaseModel):
    action: Action


class MarketPosition(enum.IntEnum):
    LEADER = 3
    AVERAGE = 2
    TRAILER = 1
    NONE = 0


class Observation(BaseModel):
    gameId: str
    timeslot: int
    gridImbalance: float
    ownImbalanceKwh: float
    customerCount: int
    customerNetDemand: float
    marketPosition: MarketPosition
    wholesalePrice: float
    ownWholesalePrice: float
    cashPosition: float

    def to_feature_vector(self) -> List[Union[float, int]]:
        return [
            self.timeslot,
            self.gridImbalance,
            self.ownImbalanceKwh,
            self.customerNetDemand,
            self.wholesalePrice,
            self.ownWholesalePrice,
            self.cashPosition,
            self.customerCount,
            self.marketPosition.value,
        ]


class Episode(BaseModel):
    episode_id: str


class StartEpisodeRequest(BaseModel):
    training_enabled: bool = True


class EndEpisodeRequest(BaseModel):
    episode_id: str
    observation: Observation


class LogReturnsRequest(BaseModel):
    episode_id: str
    reward: float
    observation: Observation
    last_action: Optional[Action]


class GetActionRequest(BaseModel):
    episode_id: str
    observation: Observation
