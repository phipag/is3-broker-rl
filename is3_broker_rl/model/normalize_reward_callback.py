# `InputReader` generator (returns None if no input reader is needed on
# the respective worker).
from typing import Dict, Tuple

import numpy as np
from ray.rllib import Policy, RolloutWorker, SampleBatch
from ray.rllib.agents import DefaultCallbacks
from ray.rllib.evaluation import Episode
from ray.rllib.utils.filter import MeanStdFilter
from ray.rllib.utils.typing import AgentID, PolicyID


class NormalizeRewardCallback(DefaultCallbacks):
    def __init__(self, legacy_callbacks_dict: Dict[str, callable] = None):
        super().__init__(legacy_callbacks_dict)
        self._reward_normalizer = MeanStdFilter(shape=(1,))

    def on_postprocess_trajectory(
        self,
        *,
        worker: RolloutWorker,
        episode: Episode,
        agent_id: AgentID,
        policy_id: PolicyID,
        policies: Dict[PolicyID, Policy],
        postprocessed_batch: SampleBatch,
        original_batches: Dict[AgentID, Tuple[Policy, SampleBatch]],
        **kwargs,
    ) -> None:
        """Called immediately after a policy's postprocess_fn is called.

        You can use this callback to do additional postprocessing for a policy,
        including looking at the trajectory data of other agents in multi-agent
        settings.

        Args:
            worker: Reference to the current rollout worker.
            episode: Episode object.
            agent_id: Id of the current agent.
            policy_id: Id of the current policy for the agent.
            policies: Mapping of policy id to policy objects. In single
                agent mode there will only be a single "default_policy".
            postprocessed_batch: The postprocessed sample batch
                for this agent. You can mutate this object to apply your own
                trajectory postprocessing.
            original_batches: Mapping of agents to their unpostprocessed
                trajectory data. You should not mutate this object.
            kwargs: Forward compatibility placeholder.
        """
        print("Original rewards:", postprocessed_batch["rewards"])
        # We normalize each reward separately because we do not want a column-wise normalization
        postprocessed_batch["rewards"] = np.array(
            [self._reward_normalizer([reward])[0] for reward in postprocessed_batch["rewards"]]
        )
        print("Normalized rewards:", postprocessed_batch["rewards"])
        if self.legacy_callbacks.get("on_postprocess_traj"):
            self.legacy_callbacks["on_postprocess_traj"](
                {
                    "episode": episode,
                    "agent_id": agent_id,
                    "pre_batch": original_batches[agent_id],
                    "post_batch": postprocessed_batch,
                    "all_pre_batches": original_batches,
                }
            )
