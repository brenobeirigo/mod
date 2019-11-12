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
from mod.env.adp.AdpHiredVector import AdpHired

port = 4999
url = f"http://localhost:{port}"

# Reproducibility of the experiments
random.seed(1)


class AmodNetwork(Amod):
    def __init__(self, config, car_positions=[]):
        """Street network Amod environment
        
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

        self.decision_dict = None

        # Defining map points with aggregation_levels
        self.points, distance_levels, level_count, points_level = nw.query_point_list(
            step=self.config.step_seconds,
            projection=self.config.projection,
            level_dist_list=self.config.level_dist_list,
        )

        # Set of points per level
        self.points_level = points_level

        # Levels correspond to distances queried in the server.
        # E.g., [0, 30, 60, 120, 300]
        Point.levels = sorted(distance_levels)

        # Unique ids per distance
        Point.level_count = [level_count[d] for d in Point.levels]

        self.init_fleet(self.points, car_positions)

        self.adp = AdpHired(self.points, self.config)

        self.adp.init_learning()
        self.adp.init_weighting_settings()

    # @functools.lru_cache(maxsize=None)
    def get_distance(self, o, d):
        """Receives two points referring to network ids and return the
        the distance of the shortest path between them (km).

        Parameters
        ----------
        o : Point
            Origin point
        d : Destination
            Destination point

        Returns
        -------
        float
            Shortest path
        """
        return nw.get_distance(o.id, d.id)

    # @functools.lru_cache(maxsize=None)
    def get_neighbors(self, center_point, reach=1):
        return nw.query_neighbors(center_point.id, reach=reach)

    def reachable_neighbors(self, n, t, limit):
        return nw.tenv.reachable_neighbors(n, t, limit)

    # @functools.lru_cache(maxsize=None)
    def get_zone_neighbors(self, center, explore=False):
        """Get the ids of nodes in the closest (explore=True)
        or farthest (explore=False) neighborhoods.

        When a vehicle is stoped for more than MAX_IDLE_STEP_COUNT
        steps, it is alowed to explore farther locations.

        Parameters
        ----------
        center : int
            Node id.
        explore : bool, optional
            If True, return the farthest neighbors, by default False

        Returns
        -------
        set
            Set of candidate nodes to rebalance.
        """
        targets = set()
        level_n_neighbors = self.config.n_neighbors

        # If car is parked for more than MAX_IDLE_STEP_COUNT, it can
        # explore (rebalance) to farther areas to scape low-demand
        # areas.
        if explore:
            level_n_neighbors = self.config.n_neighbors_explore

        for l, n in level_n_neighbors:
            step = Point.levels[l]
            step_targets = nw.query_neighbor_zones(
                Point.point_dict[center].level_ids_dic[step],
                step,
                n_neighbors=n,
            )
            targets.update(step_targets)
        return targets

    # @functools.lru_cache(maxsize=None)
    def get_level_neighbors(self, center, level):
        return nw.query_level_neighbors(
            center.id_level(level), Point.levels[level]
        )

    # @functools.lru_cache(maxsize=None)
    def get_region_elements(self, center, level):
        return nw.query_level_neighbors(center, Point.levels[level])

    def print_car_traces_geojson(self):
        for c in self.cars:
            for o, d in zip(c.point_list[:-1], c.point_list[1:]):
                url_neighbors = f"{url}/sp_coords/{o}/{d}"
                r = requests.get(url=url_neighbors)
                traces = r.text.split(";")
