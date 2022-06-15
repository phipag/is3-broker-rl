from pydantic import BaseModel


class Observation(BaseModel):
    gameId: str
    timeslot: int
    gridImbalance: float
    ownImbalance: float
    customerCount: int
    customerNetDemand: float
    marketPosition: str


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


class GetActionRequest(BaseModel):
    episode_id: str
    observation: Observation
