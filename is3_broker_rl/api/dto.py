"""
This file contains the Data Transfer Objects (DTOs) used for API communication with the Java broker.
"""
import enum
from typing import Dict, List, Optional, Union

from pydantic import BaseModel


class TariffRateAction(enum.IntEnum):
    NEW_ITERATION = 4
    LEADER = 3
    AVERAGE = 2
    TRAILER = 1
    NO_OP = 0


class Action(BaseModel):
    tariff_rate_action: TariffRateAction
    ppf_action: float


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
    customerNetDemand: float
    marketPosition: MarketPosition
    wholesalePrice: float
    ownWholesalePrice: float
    cashPosition: float
    consumptionShare: float
    productionShare: float

    def to_feature_vector(self) -> List[Union[float, int]]:
        return [
            self.timeslot,
            self.gridImbalance,
            self.ownImbalanceKwh,
            self.customerNetDemand,
            self.wholesalePrice,
            self.ownWholesalePrice,
            self.cashPosition,
            self.consumptionShare,
            self.productionShare,
            self.marketPosition.value,
        ]

    def to_feature_dict(self) -> Dict[str, Union[float, int]]:
        return {
            "timeslot": self.timeslot,
            "gridImbalance": self.gridImbalance,
            "ownImbalanceKwh": self.ownImbalanceKwh,
            "customerNetDemand": self.customerNetDemand,
            "wholesalePrice": self.wholesalePrice,
            "ownWholesalePrice": self.ownWholesalePrice,
            "cashPosition": self.cashPosition,
            "consumptionShare": self.consumptionShare,
            "productionShare": self.productionShare,
            "marketPosition": self.marketPosition.value,
        }


class Episode(BaseModel):
    episode_id: str


class StartEpisodeRequest(BaseModel):
    training_enabled: bool = True


class EndEpisodeRequest(BaseModel):
    episode_id: str
    observation: Observation


class Reward(BaseModel):
    consumption_profit: float
    action_penalty: float
    consumption_fees: float
    balancing_costs: float
    capacity_costs: float
    wholesale_costs: float


class LogReturnsRequest(BaseModel):
    episode_id: str
    reward: float
    reward_info: Reward
    observation: Observation
    last_action: Optional[Action]


class GetActionRequest(BaseModel):
    episode_id: str
    observation: Observation
