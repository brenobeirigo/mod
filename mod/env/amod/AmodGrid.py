from mod.env.car import Car, HiredCar
from mod.env.trip import Trip
from mod.env.network import Point
import mod.env.network as nw
import itertools as it
from collections import defaultdict
import numpy as np
import random
from pprint import pprint
from mod.env.config import FOLDER_EPISODE_TRACK
from functools import lru_cache
import requests
import functools
from mod.env.amod.Amod import Amod
import mod.env.decision_utils as du

# Reproducibility of the experiments
random.seed(1)

class AmodGrid(Amod):
    def __init__(self, config, car_positions=[]):
        """Grid (rows X cols) Amod environment
        
        Parameters
        ----------
        Amod : Environment parent class
            Methods to manipulate environment
        config : Config
            Simulation settings
        car_positions : list, optional
            Cars can start from predefined positions, by default []
        """
        super().__init__(config)

        # ------------------------------------------------------------ #
        # Network ######################################################
        # -------------------------------------------------------------#

        # Defining the operational map
        self.n_zones = self.config.rows * self.config.cols
        zones = np.arange(self.n_zones)
        self.zones = zones.reshape((self.config.rows, self.config.cols))

        # Defining map points with aggregation_levels
        self.points = nw.get_point_list(
            self.config.rows,
            self.config.cols,
            levels=self.config.aggregation_levels,
        )

        self.init_fleet(self.points, car_positions)
        self.init_learning()
        self.init_weighting_settings()

    @functools.lru_cache(maxsize=None)
    def get_distance(self, o, d):
        """Receives two points of a gridmap and returns
        Euclidian distance.

        Arguments:
            o {Point} -- Origin point
            d {Point} -- Destination point

        Returns:
            float -- Euclidian distance
        """
        return self.config.zone_widht * np.linalg.norm(
            np.array([o.x, o.y]) - np.array([d.x, d.y])
        )

    @functools.lru_cache(maxsize=None)
    def get_neighbors(self, center, level=0, n_neighbors=4):
        return nw.get_neighbor_zones(
            center, self.config.pickup_zone_range, self.zones
        )
