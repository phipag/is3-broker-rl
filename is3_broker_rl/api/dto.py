"""
This file contains the Data Transfer Objects (DTOs) used for API communication with the Java broker.
"""
import enum
from typing import List, Optional, Union

from pydantic import BaseModel


class TariffRateAction(enum.IntEnum):
    NEW_ITERATION = 4
    LEADER = 3
    AVERAGE = 2
    TRAILER = 1
    NO_OP = 0


# Use this in type hints to indicate that a PPFAction is returned
PPFAction = int


class Action(BaseModel):
    tariff_rate_action: TariffRateAction
    ppf_action: PPFAction


class ActionResponse(BaseModel):
    action: Action


class MarketPosition(enum.IntEnum):
    LEADER = 3
    AVERAGE = 2
    TRAILER = 1
    NONE = 0


class CustomerGroup(BaseModel):
    name: str
    total_population: int
    subscribed_population: int

    @property
    def subscription_share(self) -> float:
        if self.total_population == 0:
            return 0

        return self.subscribed_population / self.total_population


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
    customerGroups: List[CustomerGroup]
    productionShare: float

    def to_feature_vector(self) -> List[Union[float, int]]:
        subscriptionSharePerCustomerGroup = [
            group.subscription_share
            for group in
            # We sort the customerGroups to maintain the order in the feature_vector.
            # This is important for the gym space.
            sorted(self.customerGroups, key=lambda group: group.name)
        ]
        return [
            self.timeslot,
            self.gridImbalance,
            self.ownImbalanceKwh,
            self.customerNetDemand,
            self.wholesalePrice,
            self.ownWholesalePrice,
            self.cashPosition,
            *subscriptionSharePerCustomerGroup,
            self.productionShare,
            self.marketPosition.value,
        ]


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
    consumption_share_penalty: float
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
