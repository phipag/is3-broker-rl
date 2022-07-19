import gym
import numpy as np

# Class to initialize the action and observation space


class Env_config:
    def __init__(self) -> None:
        l_bounds = []
        h_bounds = []
        # l_bounds.append(np.array([-np.inf]*24))     #p_grid_imbalance = 0
        # h_bounds.append(np.array([np.inf]*24))
        # l_bounds.append(np.array([-np.inf]*24))     #p_customer_prosumption = 0
        # h_bounds.append(np.array([np.inf]*24))
        # l_bounds.append(np.array([-np.inf]*24))     #p_wholesale_price = 0
        # h_bounds.append(np.array([np.inf]*24))
        # l_bounds.append(np.array([-np.inf]*24))     #p_cloud_cover = 0
        # h_bounds.append(np.array([np.inf]*24))
        # l_bounds.append(np.array([-np.inf]*24))     #p_temperature = 0
        # h_bounds.append(np.array([np.inf]*24))
        # l_bounds.append(np.array([-np.inf]*24))     #p_wind_speed = 0
        # h_bounds.append(np.array([np.inf]*24))
        # l_bounds.append(np.array([-np.inf]*24))     # hour of the start with dummy.
        # h_bounds.append(np.array([np.inf]*24))
        # l_bounds.append(np.array([-np.inf]*7))      # day of the start with dummy
        # h_bounds.append(np.array([np.inf]*7))

        l_bounds.append(np.array([-1000] * 24))  # p_grid_imbalance = 0
        h_bounds.append(np.array([1000] * 24))
        l_bounds.append(np.array([-1000] * 24))  # p_customer_prosumption = 0
        h_bounds.append(np.array([1000] * 24))
        l_bounds.append(np.array([-1000] * 24))  # p_wholesale_price = 0
        h_bounds.append(np.array([1000] * 24))
        l_bounds.append(np.array([-1000] * 24))  # p_cloud_cover = 0
        h_bounds.append(np.array([1000] * 24))
        l_bounds.append(np.array([-1000] * 24))  # p_temperature = 0
        h_bounds.append(np.array([1000] * 24))
        l_bounds.append(np.array([-1000] * 24))  # p_wind_speed = 0
        h_bounds.append(np.array([1000] * 24))
        l_bounds.append(np.array([-1000] * 24))  # cleared_orders_price = 0
        h_bounds.append(np.array([1000] * 24))
        l_bounds.append(np.array([-1000] * 24))  # cleared_orders_energy = 0
        h_bounds.append(np.array([1000] * 24))
        l_bounds.append(np.array([-1000] * 24))  # cleared_trade_price = 0
        h_bounds.append(np.array([1000] * 24))
        l_bounds.append(np.array([-1000] * 24))  # cleared_trade_energy = 0
        h_bounds.append(np.array([1000] * 24))
        l_bounds.append(np.array([-1000] * 1))  # customer_count
        h_bounds.append(np.array([1000] * 1))
        l_bounds.append(np.array([-1000000] * 1))  # total_prosumption
        h_bounds.append(np.array([1000000] * 1))
        l_bounds.append(np.array([-1000000] * 24))  # market_position = 0
        h_bounds.append(np.array([10000000] * 24))
        l_bounds.append(np.array([-1000] * 24))  # hour of the start with dummy.
        h_bounds.append(np.array([1000] * 24))
        l_bounds.append(np.array([-1000] * 7))  # day of the start with dummy
        h_bounds.append(np.array([1000] * 7))

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
            # shape=observation_space_bounds[:, 0].shape,
        )

        # self.action_space = gym.spaces.Tuple((gym.spaces.Box(low=-np.inf, high=np.inf, shape=(2,)),
        # gym.spaces.Discrete(24)))
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(48,))

    def get_gym_spaces(self):

        return self.observation_space, self.action_space

    def get_rl_config(self):

        return
