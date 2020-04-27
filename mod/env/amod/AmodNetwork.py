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
        (
            self.points,
            distance_levels,
            level_count,
            point_ids_level,
        ) = nw.query_point_list(
            step=self.config.step_seconds,
            projection=self.config.projection,
            level_dist_list=self.config.level_dist_list,
        )

        # Set of points per level
        self.point_ids_level = point_ids_level

        # Levels correspond to distances queried in the server.
        # E.g., [0, 30, 60, 120, 300]
        Point.levels = sorted(distance_levels)

        # Unique ids per distance
        Point.level_count = [level_count[d] for d in Point.levels]

        # Nodes with no neighbors are not valid trips. Valid neighbors
        # can be accessed within the time increment, which is not always
        # possible when using higher up centroids.
        self.unreachable_ods, self.neighbors = self.get_unreachable_ods()

        # Points in centroid level that can be reached
        self.reachable_points = [
            self.points[v]
            for v in (
                set(self.point_ids_level[self.config.centroid_level])
                - self.unreachable_ods
            )
        ]

        # Ids of reachable points
        self.reachable_point_ids = {v.id for v in self.reachable_points}

        self.init_fleet(self.points, car_positions)

        # Point objects level
        self.points_level = {
            level: [self.points[point_id] for point_id in point_ids]
            for level, point_ids in enumerate(self.point_ids_level)
        }

        # Cars always start from set of parking lots located at the
        # center of regions of a certain level
        if self.config.cars_start_from_parking_lots:
            level_parking = self.config.level_parking_lots
            self.level_parking_ids = set(point_ids_level[level_parking])
            self.level_parking_points = self.points_level[level_parking]

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

    def reachable_neighbors(self, n, t):
        return nw.tenv.reachable_neighbors(n, t)

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
            # Example of steps that correspond to level: 0=0, 1=100, etc.
            step = Point.levels[l]
            step_targets = nw.query_neighbor_zones(
                Point.point_dict[center].level_ids_dic[step],
                step,
                n_neighbors=n,
            )

            targets.update(step_targets)

            # print("\n\n#### CENTER", center, l, step_targets)

            # When active, rebalance options are extended to neighbors
            # of the targets in "step_targets" at level "sub_level".
            # Hence, rebalance to O(step_targets*step_targets).
            # print("SUB", self.config.rebalance_sub_level)
            if (
                self.config.rebalance_sub_level is not None
                and l > self.config.rebalance_sub_level
            ):

                for target in step_targets:
                    # Select step corresponding to level
                    step = Point.levels[self.config.rebalance_sub_level]
                    sub_step_targets = nw.query_neighbor_zones(
                        Point.point_dict[target].level_ids_dic[step],
                        step,
                        n_neighbors=n,
                    )

                    # print(" -", target, sub_step_targets)

                    # All targets from sublevels
                    targets.update(sub_step_targets)

        # Guarantee all targets are at the centroid level
        targets = set(
            [
                self.points[t].id_level(self.config.centroid_level)
                for t in targets
            ]
        )

        # print("After getting the targets:", targets)

        # Cannot rebalance to itself
        targets = targets - {center}

        # print("After removing the center:", targets)

        if self.config.rebalancing_time_range_min:

            min_reb_time, max_reb_time = self.config.rebalancing_time_range_min

            # Sort rebalancing targets (farther first)
            id_dist = sorted(
                [
                    (d, self.get_travel_time(nw.get_distance(center, d)))
                    for d in targets
                ],
                key=lambda x: x[1],
                reverse=True,
            )

            # Remove targets that cannot be accessed whithn time increment
            targets = [
                d
                for d, dist in id_dist
                if dist <= max_reb_time and dist >= min_reb_time
            ]

            # print("After distance filter:", targets)
            # print(
            #     center,
            #     len(targets),
            #     targets,
            #     len(id_dist),
            #     self.config.rebalance_max_targets,
            #     id_dist,
            # )

        # Limit number of targets
        if self.config.rebalance_max_targets is not None:
            targets = targets[: self.config.rebalance_max_targets]

        # print("Cut max. targets:", targets)

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
