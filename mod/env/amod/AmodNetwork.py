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
from mod.env.adp.AdpHired import AdpHired

port = 4999
url = f"http://localhost:{port}"


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
            max_levels=self.config.aggregation_levels,
            projection=self.config.projection,
            level_dist_list=self.config.level_dist_list,
        )

        # Set of points per level
        self.points_level = points_level

        # Levels correspond to distances queried in the server.
        # E.g., [0, 30, 60, 120, 300]
        Point.levels = sorted(distance_levels)

        # Unique ids per distance
        Point.level_count = [level_count[str(d)] for d in Point.levels]

        self.init_fleet(self.points, car_positions)

        self.adp = AdpHired(
            self.points,
            self.config.aggregation_levels,
            self.config.stepsize,
            self.config.harmonic_stepsize,
        )

        self.adp.init_learning()
        self.adp.init_weighting_settings()

    @functools.lru_cache(maxsize=None)
    def get_distance(self, o, d):
        """Receives two points referring to network ids and return the
        the distance of the shortest path between them (meters).

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

    @functools.lru_cache(maxsize=None)
    def get_neighbors(self, center_point, reach=1):
        return nw.query_neighbors(center_point.id, reach=reach)

    @functools.lru_cache(maxsize=None)
    def get_zone_neighbors(self, center, level=[0], n_neighbors=[4]):
        """Get the ids of "n_neighbors" neighboring region centers
        considering aggregation level around center.

        Parameters
        ----------
        center : id of region center
            [description]
        level : int, optional
            [description], by default 0
        n_neighbors : int, optional
            [description], by default 4

        Returns
        -------
        [type]
            [description]
        """
        targets = set()
        for l, n in list(zip(level, n_neighbors)):
            step = Point.levels[l]
            step_targets = nw.query_neighbor_zones(
                center.level_ids_dic[step], step, n_neighbors=n
            )
            targets.update(step_targets)
        return targets

    @functools.lru_cache(maxsize=None)
    def get_level_neighbors(self, center, level):
        return nw.query_level_neighbors(
            center.id_level(level), Point.levels[level]
        )

    @functools.lru_cache(maxsize=None)
    def get_region_elements(self, center, level):
        return nw.query_level_neighbors(center, Point.levels[level])

    def print_car_traces_geojson(self):
        for c in self.cars:
            for o, d in zip(c.point_list[:-1], c.point_list[1:]):
                url_neighbors = f"{url}/sp_coords/{o}/{d}"
                r = requests.get(url=url_neighbors)
                traces = r.text.split(";")
