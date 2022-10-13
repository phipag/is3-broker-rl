from gym import spaces
from gym.envs.registration import EnvSpec
import gym
import numpy as np
import pickle
import unittest
import tensorflow as tf
import ray
from ray.rllib.agents.a3c import a2c
from ray.rllib.env import MultiAgentEnv
from ray.rllib.env.base_env import convert_to_base_env
#from ray.rllib.env.tests.test_external_env import SimpleServing
from ray.rllib.env.vector_env import VectorEnv
from ray.rllib.models import ModelCatalog
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.agents.sac.sac_tf_model import SACTFModel
from ray.rllib.evaluate import rollout
from ray.tune.registry import register_env
from ray.rllib.utils.framework import try_import_tf, try_import_torch
from ray.rllib.utils.numpy import one_hot
from ray.rllib.utils.spaces.repeated import Repeated
from ray.rllib.utils.test_utils import check
from ray.rllib.models.tf.misc import normc_initializer
from typing import Dict, List, Optional
from ray.rllib.models.catalog import ModelCatalog
import logging
import dotenv
from pathlib import Path
import os
from ray.rllib.utils.spaces.simplex import Simplex
from ray.rllib.utils.typing import ModelConfigDict, TensorType, TensorStructType
from ray.rllib.utils.annotations import override
from is3_broker_rl import model
import gym
from gym.spaces import Box, Discrete
import numpy as np
import tree  # pip install dm_tree
from typing import Dict, List, Optional

from ray.rllib.models.catalog import ModelCatalog
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_tf
from ray.rllib.utils.spaces.simplex import Simplex
from ray.rllib.utils.typing import ModelConfigDict, TensorType, TensorStructType

#tf1, tf, tfv = try_import_tf()
#_, nn = try_import_torch()

# https://docs.ray.io/en/latest/rllib/rllib-models.html

class MyModelClass(SACTFModel):
    """Custom model for policy gradient algorithms."""

    #def __init__(self,
    #    obs_space: gym.spaces.Space,
    #    action_space: gym.spaces.Space,
    #    num_outputs: Optional[int],
    #    model_config: ModelConfigDict,
    #    name: str,
    #    policy_model_config: ModelConfigDict = None,
    #    q_model_config: ModelConfigDict = None,
    #    twin_q: bool = False,
    #    initial_alpha: float = 1.0,
    #    target_entropy: Optional[float] = None,
    #    **kwargs):
#
    #    logging.basicConfig(level=logging.INFO, filename="stdout")
    #    try:
    #        dotenv.load_dotenv(override=False)
    #        logging.basicConfig(level=logging.INFO, filename="stdout")
    #        self._DATA_DIR: Path = Path(os.environ.get("DATA_DIR", "data/"))
    #        logging.info("Test_Model created")
    #        super(MyModelClass, self).__init__(obs_space,
    #            action_space,
    #            num_outputs,
    #            model_config,
    #            name,
    #            policy_model_config,
    #            q_model_config,
    #            twin_q,
    #            initial_alpha,
    #            target_entropy)
    #        cell_size = 256
#
    #        
#
    #        #self.base_model = self.build(obs_space, action_space, seq_len)
    #        
    #    except Exception as e:
    #        logging.info(f"Cant create Model.  {e}", exc_info=True)

    
    


    def __init__(
        self,
        obs_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        num_outputs: Optional[int],
        model_config: ModelConfigDict,
        name: str,
        policy_model_config: ModelConfigDict = None,
        q_model_config: ModelConfigDict = None,
        twin_q: bool = False,
        initial_alpha: float = 1.0,
        target_entropy: Optional[float] = None,
    ):
        """Initialize a SACTFModel instance.
        Args:
            policy_model_config: The config dict for the
                policy network.
            q_model_config: The config dict for the
                Q-network(s) (2 if twin_q=True).
            twin_q: Build twin Q networks (Q-net and target) for more
                stable Q-learning.
            initial_alpha: The initial value for the to-be-optimized
                alpha parameter (default: 1.0).
            target_entropy (Optional[float]): A target entropy value for
                the to-be-optimized alpha parameter. If None, will use the
                defaults described in the papers for SAC (and discrete SAC).
        Note that the core layers for forward() are not defined here, this
        only defines the layers for the output heads. Those layers for
        forward() should be defined in subclasses of SACModel.
        """
        super(SACTFModel, self).__init__(
            obs_space, action_space, num_outputs, model_config, name
        )
        if isinstance(action_space, Discrete):
            self.action_dim = action_space.n
            self.discrete = True
            action_outs = q_outs = self.action_dim
        elif isinstance(action_space, Box):
            self.action_dim = np.product(action_space.shape)
            self.discrete = False
            action_outs = 2 * self.action_dim
            q_outs = 1
        else:
            assert isinstance(action_space, Simplex)
            self.action_dim = np.product(action_space.shape)
            self.discrete = False
            action_outs = self.action_dim
            q_outs = 1

        self.action_model = self.build_policy_model(
            self.obs_space, action_outs, policy_model_config, "policy_model"
        )

        self.q_net = self.build_q_model(
            self.obs_space, self.action_space, q_outs, q_model_config, "q"
        )
        if twin_q:
            self.twin_q_net = self.build_q_model(
                self.obs_space, self.action_space, q_outs, q_model_config, "twin_q"
            )
        else:
            self.twin_q_net = None

        self.log_alpha = tf.Variable(
            np.log(initial_alpha), dtype=tf.float32, name="log_alpha"
        )
        self.alpha = tf.exp(self.log_alpha)

        # Auto-calculate the target entropy.
        if target_entropy is None or target_entropy == "auto":
            # See hyperparams in [2] (README.md).
            if self.discrete:
                target_entropy = 0.98 * np.array(
                    -np.log(1.0 / action_space.n), dtype=np.float32
                )
            # See [1] (README.md).
            else:
                target_entropy = -np.prod(action_space.shape)
        self.target_entropy = target_entropy

    @override(TFModelV2)
    def forward(
        self,
        input_dict: Dict[str, TensorType],
        state: List[TensorType],
        seq_lens: TensorType,
    ) -> (TensorType, List[TensorType]):
        """The common (Q-net and policy-net) forward pass.
        NOTE: It is not(!) recommended to override this method as it would
        introduce a shared pre-network, which would be updated by both
        actor- and critic optimizers.
        """
        return input_dict["obs"], state

    
    def build_policy_model(self, obs_space, num_outputs, policy_model_config, name):
        """Builds the policy model used by this SAC.
        Override this method in a sub-class of SACTFModel to implement your
        own policy net. Alternatively, simply set `custom_model` within the
        top level SAC `policy_model` config key to make this default
        implementation of `build_policy_model` use your custom policy network.
        Returns:
            TFModelV2: The TFModelV2 policy sub-model.
        """
        print(f"build_policy_model: {policy_model_config}")
        #logging.info(f"build_policy_model {policy_model_config}")
        #input_layer = tf.keras.layers.Input(
        #    shape=(None, obs_space.shape[0] * 2))
        #output_layer1 = tf.keras.layers.Dense(256, activation='relu')(input_layer)
        #output_layer1 = tf.keras.layers.Dense(num_outputs, activation='relu')(3)
        #rnn_model = tf.keras.Model(
        #    input_layer, output_layer1
        #)
        #logging.info(rnn_model.summary())
        #logging.info("build_policy_model: Test_Model created")
        cell_size = 1024

        # Define input layers
        input_layer = tf.keras.layers.Input(
            shape=(None, obs_space.shape[0]), name="inputs"
        )
        state_in_h = tf.keras.layers.Input(shape=(cell_size,), name="h")
        state_in_c = tf.keras.layers.Input(shape=(cell_size,), name="c")
        seq_in = tf.keras.layers.Input(shape=(), name="seq_in", dtype=tf.int32)

        # Preprocess observation with a hidden layer and send to LSTM cell
        dense1 = tf.keras.layers.Dense(
            512, activation=tf.nn.relu, name="dense1", kernel_regularizer='l1_l2'
        )(input_layer)
        dense2 = tf.keras.layers.Dense(
            512, activation=tf.nn.relu, name="dense2", kernel_regularizer='l1_l2'
        )(dense1)
        dense3 = tf.keras.layers.Dense(
            512, activation=tf.nn.relu, name="dense3", kernel_regularizer='l1_l2'
        )(dense2)
        lstm_out, state_h, state_c = tf.keras.layers.LSTM(
            cell_size, return_sequences=True, return_state=True, name="lstm"
        )(
            inputs=dense3,
            mask=tf.sequence_mask(seq_in),
            initial_state=[state_in_h, state_in_c],
        )

        # Postprocess LSTM output with another hidden layer and compute values
        logits = tf.keras.layers.Dense(
            self.num_outputs, activation=tf.keras.activations.linear, name="logits"
        )(lstm_out)
        values = tf.keras.layers.Dense(1, activation="softmax", name="values")(lstm_out)

        # Create the RNN model
        rnn_model = tf.keras.Model(
            inputs=[input_layer, seq_in, state_in_h, state_in_c],
            outputs=[logits, values, state_h, state_c],
        )
        
        return rnn_model

    def build_q_model(self, obs_space, action_space, num_outputs, q_model_config, name):
        """Builds one of the (twin) Q-nets used by this SAC.
        Override this method in a sub-class of SACTFModel to implement your
        own Q-nets. Alternatively, simply set `custom_model` within the
        top level SAC `q_model_config` config key to make this default implementation
        of `build_q_model` use your custom Q-nets.
        Returns:
            TFModelV2: The TFModelV2 Q-net sub-model.
        """
        self.concat_obs_and_actions = False
        if self.discrete:
            input_space = obs_space
        else:
            orig_space = getattr(obs_space, "original_space", obs_space)
            if isinstance(orig_space, Box) and len(orig_space.shape) == 1:
                input_space = Box(
                    float("-inf"),
                    float("inf"),
                    shape=(orig_space.shape[0] + action_space.shape[0],),
                )
                self.concat_obs_and_actions = True
            else:
                input_space = gym.spaces.Tuple([orig_space, action_space])

        model = ModelCatalog.get_model_v2(
            input_space,
            action_space,
            num_outputs,
            q_model_config,
            framework="tf",
            name=name,
        )
        return model

    def get_q_values(
        self, model_out: TensorType, actions: Optional[TensorType] = None
    ) -> TensorType:
        """Returns Q-values, given the output of self.__call__().
        This implements Q(s, a) -> [single Q-value] for the continuous case and
        Q(s) -> [Q-values for all actions] for the discrete case.
        Args:
            model_out: Feature outputs from the model layers
                (result of doing `self.__call__(obs)`).
            actions (Optional[TensorType]): Continuous action batch to return
                Q-values for. Shape: [BATCH_SIZE, action_dim]. If None
                (discrete action case), return Q-values for all actions.
        Returns:
            TensorType: Q-values tensor of shape [BATCH_SIZE, 1].
        """
        return self._get_q_value(model_out, actions, self.q_net)

    def get_twin_q_values(
        self, model_out: TensorType, actions: Optional[TensorType] = None
    ) -> TensorType:
        """Same as get_q_values but using the twin Q net.
        This implements the twin Q(s, a).
        Args:
            model_out: Feature outputs from the model layers
                (result of doing `self.__call__(obs)`).
            actions (Optional[Tensor]): Actions to return the Q-values for.
                Shape: [BATCH_SIZE, action_dim]. If None (discrete action
                case), return Q-values for all actions.
        Returns:
            TensorType: Q-values tensor of shape [BATCH_SIZE, 1].
        """
        return self._get_q_value(model_out, actions, self.twin_q_net)

    def _get_q_value(self, model_out, actions, net):
        # Model outs may come as original Tuple/Dict observations, concat them
        # here if this is the case.
        if isinstance(net.obs_space, Box):
            if isinstance(model_out, (list, tuple)):
                model_out = tf.concat(model_out, axis=-1)
            elif isinstance(model_out, dict):
                model_out = tf.concat(list(model_out.values()), axis=-1)

        # Continuous case -> concat actions to model_out.
        if actions is not None:
            if self.concat_obs_and_actions:
                input_dict = {"obs": tf.concat([model_out, actions], axis=-1)}
            else:
                # TODO(junogng) : SampleBatch doesn't support list columns yet.
                #     Use ModelInputDict.
                input_dict = {"obs": (model_out, actions)}
        # Discrete case -> return q-vals for all actions.
        else:
            input_dict = {"obs": model_out}
        # Switch on training mode (when getting Q-values, we are usually in
        # training).
        input_dict["is_training"] = True

        return net(input_dict, [], None)

    def get_action_model_outputs(
        self,
        model_out: TensorType,
        state_in: List[TensorType] = None,
        seq_lens: TensorType = None,
    ) -> (TensorType, List[TensorType]):
        """Returns distribution inputs and states given the output of
        policy.model().
        For continuous action spaces, these will be the mean/stddev
        distribution inputs for the (SquashedGaussian) action distribution.
        For discrete action spaces, these will be the logits for a categorical
        distribution.
        Args:
            model_out: Feature outputs from the model layers
                (result of doing `model(obs)`).
            state_in List(TensorType): State input for recurrent cells
            seq_lens: Sequence lengths of input- and state
                sequences
        Returns:
            TensorType: Distribution inputs for sampling actions.
        """

        def concat_obs_if_necessary(obs: TensorStructType):
            """Concat model outs if they are original tuple observations."""
            if isinstance(obs, (list, tuple)):
                obs = tf.concat(obs, axis=-1)
            elif isinstance(obs, dict):
                obs = tf.concat(
                    [
                        tf.expand_dims(val, 1) if len(val.shape) == 1 else val
                        for val in tree.flatten(obs.values())
                    ],
                    axis=-1,
                )
            return obs

        if state_in is None:
            state_in = []

        if isinstance(model_out, dict) and "obs" in model_out:
            # Model outs may come as original Tuple observations
            if isinstance(self.action_model.obs_space, Box):
                model_out["obs"] = concat_obs_if_necessary(model_out["obs"])
            return self.action_model(model_out, state_in, seq_lens)
        else:
            if isinstance(self.action_model.obs_space, Box):
                model_out = concat_obs_if_necessary(model_out)
            return self.action_model({"obs": model_out}, state_in, seq_lens)

    def policy_variables(self):
        """Return the list of variables for the policy net."""

        return self.action_model.variables()

    def q_variables(self):
        """Return the list of variables for Q / twin Q nets."""

        return self.q_net.variables() + (
            self.twin_q_net.variables() if self.twin_q_net else []
        )
