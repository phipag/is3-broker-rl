import gym
from ray import rllib
import numpy as np
import ray
from ray.rllib.env.policy_client import PolicyClient
import pandas as pd
from ray.rllib.agents import impala


# Class to initialize the action and observation space

class Env_config():
    def __init__(self) -> None:
        l_bounds = []
        h_bounds = []
        #l_bounds.append(np.array([-np.inf]*24))     #p_grid_imbalance = 0
        #h_bounds.append(np.array([np.inf]*24))                                       
        #l_bounds.append(np.array([-np.inf]*24))     #p_customer_prosumption = 0
        #h_bounds.append(np.array([np.inf]*24))                                           
        #l_bounds.append(np.array([-np.inf]*24))     #p_wholesale_price = 0
        #h_bounds.append(np.array([np.inf]*24))     
        #l_bounds.append(np.array([-np.inf]*24))     #p_cloud_cover = 0
        #h_bounds.append(np.array([np.inf]*24))     
        #l_bounds.append(np.array([-np.inf]*24))     #p_temperature = 0
        #h_bounds.append(np.array([np.inf]*24))     
        #l_bounds.append(np.array([-np.inf]*24))     #p_wind_speed = 0
        #h_bounds.append(np.array([np.inf]*24))        
        #l_bounds.append(np.array([-np.inf]*24))     # hour of the start with dummy. 
        #h_bounds.append(np.array([np.inf]*24))
        #l_bounds.append(np.array([-np.inf]*7))      # day of the start with dummy
        #h_bounds.append(np.array([np.inf]*7))

        l_bounds.append(np.array([-100000]*24))     #p_grid_imbalance = 0
        h_bounds.append(np.array([100000]*24))                                       
        l_bounds.append(np.array([-100000]*24))     #p_customer_prosumption = 0
        h_bounds.append(np.array([100000]*24))                                           
        l_bounds.append(np.array([-100000]*24))     #p_wholesale_price = 0
        h_bounds.append(np.array([100000]*24))     
        l_bounds.append(np.array([-100000]*24))     #p_cloud_cover = 0
        h_bounds.append(np.array([100000]*24))     
        l_bounds.append(np.array([-100000]*24))     #p_temperature = 0
        h_bounds.append(np.array([100000]*24))     
        l_bounds.append(np.array([-100000]*24))     #p_wind_speed = 0
        h_bounds.append(np.array([100000]*24))        
        l_bounds.append(np.array([-100000]*24))     # hour of the start with dummy. 
        h_bounds.append(np.array([100000]*24))
        l_bounds.append(np.array([-100000]*7))      # day of the start with dummy
        h_bounds.append(np.array([100000]*7))

        l_bound_total = np.array([])
        for j in l_bounds:
            l_bound_total = np.append(l_bound_total, j)
        r_bound_total = np.array([])
        for j in h_bounds:
            r_bound_total = np.append(r_bound_total, j)


        self.observation_space = gym.spaces.Box(
                    low=np.ravel(l_bound_total),
                    high=np.ravel(r_bound_total),
                    dtype=np.float32
                    #shape=observation_space_bounds[:, 0].shape,
                )


        #self.action_space = gym.spaces.Tuple((gym.spaces.Box(low=-np.inf, high=np.inf, shape=(2,)), gym.spaces.Discrete(24)))
        self.action_space = gym.spaces.Box(low=-1000, high=1000, shape=(48,))


    def get_gym_spaces(self):
        
        return self.observation_space, self.action_space


    def get_rl_config(self):
        
        return 