from gym import spaces
from gym.envs.registration import EnvSpec
import gym
import numpy as np
import pickle
import unittest

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
from ray.rllib.evaluate import rollout
from ray.tune.registry import register_env
from ray.rllib.utils.framework import try_import_tf, try_import_torch
from ray.rllib.utils.numpy import one_hot
from ray.rllib.utils.spaces.repeated import Repeated
from ray.rllib.utils.test_utils import check
from ray.rllib.models.tf.misc import normc_initializer
import logging
import dotenv
from pathlib import Path
import os

from is3_broker_rl import model


tf1, tf, tfv = try_import_tf()
_, nn = try_import_torch()

# https://docs.ray.io/en/latest/rllib/rllib-models.html
class MyModelClass(TFModelV2):
    """Custom model for policy gradient algorithms."""

    def __init__(self, obs_space: gym.Space, action_space: gym.Space, seq_len: int, model_config = {}, *args, **kwargs):
        try:
            dotenv.load_dotenv(override=False)
            logging.basicConfig(level=logging.INFO, filename="stdout")
            self._DATA_DIR: Path = Path(os.environ.get("DATA_DIR", "data/"))
            logging.info("Test_Model created")
            super(MyModelClass, self).__init__(obs_space, action_space, seq_len, model_config ,"custom_model",  *args, **kwargs)
            cell_size = 256

            

            self.base_model = self.build(obs_space, action_space, seq_len)
        except Exception as e:
            logging.info(f"Cant create Model.  {e}", exc_info=True)


    def build(self, obs_space: gym.Space, action_space: gym.Space, seq_len: int):
        # Define input layers
        #input_layer = tf.keras.layers.Input(
        #    shape=(None, obs_space.shape[0]))
        #state_in_h = tf.keras.layers.Input(shape=(256, ))
        #state_in_c = tf.keras.layers.Input(shape=(256, ))
        #seq_in = tf.keras.layers.Input(shape=(), dtype=tf.int32)
#
        ## Send to LSTM cell
        #cell_size = 256
        #lstm_out, state_h, state_c = tf.keras.layers.LSTM(
        #    cell_size, return_sequences=True, return_state=True,
        #    name="lstm")(
        #        inputs=input_layer,
        #        mask=tf.sequence_mask(seq_in),
        #        initial_state=[state_in_h, state_in_c])
        #output_layer = tf.keras.layers.Dense(256, activation='relu')(lstm_out)
#
        ## Create the RNN model
        #rnn_model = tf.keras.Model(
        #    inputs=[input_layer, seq_in, state_in_h, state_in_c],
        #    outputs=[output_layer, state_h, state_c],
        #)
        #    #stateful=True)
        #rnn_model.summary()
        input_layer = tf.keras.layers.Input(
            shape=(None, obs_space.shape[0]))
        output_layer1 = tf.keras.layers.Dense(256, activation='relu')(input_layer)
        output_layer1 = tf.keras.layers.Dense(256, activation='relu')(output_layer1)
        rnn_model = tf.keras.Model(
            input_layer, output_layer1
        )
        logging.info(rnn_model.summary())
        return rnn_model

    def forward(self, input_dict, state, seq_lens):
        try: 
            model_out, self._value_out = self.base_model(input_dict["obs_flat"])
            logging.info(f"T in forward_model: {input_dict['T']}")
            logging.info(f"value {self._value_out}")
            return model_out, state
        except Exception as e:
            logging.info(f"Cant create Model.  {e}", exc_info=True)
        



    def value_function(self):
        return tf.reshape(self._value_out, [-1])


    # TODO: change metric.
    def metrics(self):
        logging(f"metric got called")
        return {"foo": tf.constant(42.0)}
