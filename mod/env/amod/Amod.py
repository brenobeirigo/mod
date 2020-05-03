from mod.env.car import Car, HiredCar
from mod.env.trip import Trip
from mod.env.network import Point
import mod.env.decision_utils as du
import itertools as it
from collections import defaultdict
import numpy as np
import random
from pprint import pprint
from mod.env.config import FOLDER_EPISODE_TRACK
import requests
import functools
import math
from copy import deepcopy

# Reproducibility of the experiments
random.seed(1)


class Amod:
    def __init__(self, config, car_positions=[]):
        """Start AMoD environment

        Arguments:
            rows {int} -- Number of zone rows
            cols {int} -- Number of zone columns
            fleet_size {int} -- Number of cars
            battery_size {int} -- Battery max. capacity
            actions {list} -- Possible actions from each state
            car_positions -- Where each vehicle is in the beginning.
        """
        # Nodes where cars can park unrestrictedly
        self.unrestricted_parking_node_ids = {}
        self.config = config
        self.time_steps = config.time_steps

        # ------------------------------------------------------------ #
        # Battery ######################################################
        # -------------------------------------------------------------#

        self.battery_levels = config.battery_levels
        self.battery_size_distances = config.battery_size_distances
        self.fleet_size = config.fleet_size

    def init_fleet(self, points, car_positions=[]):
        print(f"### Initializing fleet... Points: {len(points)}")
        # ------------------------------------------------------------ #
        # Fleet ########################################################
        # -------------------------------------------------------------#

        self.car_origin_points = [
            point for point in random.choices(points, k=self.fleet_size)
        ]

        # If no list predefined positions
        if not car_positions:
            # Creating random fleet starting from random points
            self.cars = [
                Car(
                    point,
                    # self.battery_levels,
                    # battery_level_miles_max=self.battery_size_distances,
                )
                for point in self.car_origin_points
            ]

        else:
            # Creating fleet starting from pre-determined positions
            self.cars = [
                Car(
                    point,
                    # self.battery_levels,
                    # battery_level_miles_max=self.battery_size_distances,
                )
                for point in car_positions
            ]

        # List of available vehicles
        self.available = self.cars

        # set of trip origins rejected in the last iteration
        self.rejected_trip_origins = []
        self.last_trip_origins = []

        self.rebalancing = []

    def get_fleet_status(self):
        """Number of cars per status and total battery level
        in miles.

        Returns:
            dict, float -- #cars per status, total battery level
        """

        status_count = defaultdict(int)
        for s in Car.status_list:
            status_count[Car.status_label_dict[s]] = 0

        total_battery_level = 0
        for c in self.cars:
            total_battery_level += c.battery_level_miles
            status_count[Car.status_label_dict[c.status]] += 1
        return status_count, total_battery_level

    def get_fleet_stats(self):

        stats = dict()
        count_status = defaultdict(int)

        # Start all car statuses with 0
        for s in Car.status_list:
            count_status[s] = 0

        # Count how many car per status
        for c in self.cars:
            count_status[c.status] += 1

        stats["Main fleet"] = dict(count_status)
        stats["Available"] = {"Cars": len(self.cars)}

        return stats

    def update_fleet_status(self, time_step):

        available = []

        for car in self.cars:
            # Check if vehicles finished their tasks
            # Where are the cars?
            # What are they doing at the current step?
            # t ----- t+1 ----- t+2 ----- t+3 ----- t+4 ------- t+5
            # --trips----trips------trips-----trips------trips-----
            car.update(time_step, time_increment=self.config.time_increment)

            # Discard busy vehicles
            if not car.busy:
                available.append(car)

        self.available = available

    def hire_cars_in_trip_origins(self, trips, step, levels=1):

        # Hire vehicles in the center of the region that can access the
        # trip origin the most quickly (i.e., first sq class level)
        hired_cars = [
            HiredCar(
                self.points[t.o.id_level(t.sq1_level)],
                self.battery_levels,
                20,
                current_step=step,
                current_arrival=step * self.config.time_increment,
                battery_level_miles_max=self.battery_size_distances,
            )
            for t in trips
        ]

        # Also hire a car for the larger region center (2nd sq class)
        # giving a bigger slack to the mip solver
        if levels == 2:
            hired_cars.extend(
                [
                    HiredCar(
                        self.points[t.o.id_level(t.sq2_level)],
                        self.battery_levels,
                        20,
                        current_step=step,
                        current_arrival=step * self.config.time_increment,
                        battery_level_miles_max=self.battery_size_distances,
                    )
                    for t in trips
                ]
            )

    # ################################################################ #
    # Cost functions ################################################# #
    # ################################################################ #

    # @functools.lru_cache(maxsize=None)
    def cost_func(self, action, o, d):

        if action == du.STAY_DECISION:
            # Stay
            return 0

        elif action == du.TRIP_DECISION:
            # Pick up
            distance_trip = self.get_distance(self.points[o], self.points[d])

            reward = self.config.calculate_fare(distance_trip)
            return reward

        elif action == du.RECHARGE_DECISION:
            # Recharge
            cost = self.config.cost_recharge_single_increment
            return -cost

        else:
            # Rebalance
            return 0

    def post_cost(self, t, decision, level=None):

        # Target attribute if decision was taken
        post_t, post_pos, post_battery = self.preview_decision(t, decision)

        # Get the value estimation considering a single level
        if level is not None:
            estimate = self.adp.get_value(
                post_t, post_pos, post_battery, level=level
            )

        else:
            # Get the post decision state estimate value based on
            # hierarchical aggregation
            estimate = self.adp.get_weighted_value(
                post_t, post_pos, post_battery
            )

        return estimate

    # ################################################################ #
    # Car actions #################################################### #
    # ################################################################ #

    def full_recharge(self, car):
        """Recharge car fully and update its parameters.
        Car will be available only when recharge is complete.

        Arguments:
            car {Car} -- Car to be recharged

        Returns:
            float -- Cost of recharge
        """
        # How many rechargeable miles vehicle have
        miles = car.get_full_recharging_miles()

        # How much time vehicle needs to recharge
        time_min, steps = self.config.get_full_recharging_time(miles)

        # Total cost of rechargin
        cost = self.config.calculate_cost_recharge(time_min)

        # Update vehicle status to recharging
        car.update_recharge(time_min, miles, cost)

        return cost

    def recharge(self, car, time_min):

        # Total cost of recharging
        cost = self.config.calculate_cost_recharge(time_min)

        # Extra kilometers car can travel after recharging
        dist = self.config.calculate_dist_recharge(time_min)

        # Update vehicle status to recharging
        car.update_recharge(
            time_min, cost, dist, time_increment=self.config.time_increment
        )

        return cost

    def rebalance(self, car, target):
        """Return

        Parameters
        ----------
        car : Car
            Car to rebalance
        target : Point
            Where car rebalance to

        Returns
        -------
        3-element tuple
            (duration, distance, reward)
        """

        # Distance car has to travel to rebalance
        distance = self.get_distance(car.point, target)

        # Duration of rebalancing trip
        duration_min, duration_step = self.get_travel_time_tuple(distance)

        # No reward for rebalancing
        reward = 0

        return duration_min, duration_step, distance, reward

    def pickup(self, trip, car):
        """Insert trip into car and update car status.

        Arguments:
            trip {Trip} -- Trip matched to vehicle
            car {Car} -- Car in which trip will be inserted

        Return:
            float -- Reward gained by car after servicing trip
        """
        # Distance car has to travel to service trip
        distance_trip = self.get_distance(trip.o, trip.d)

        # Distance to pickup passanger
        distance_pickup = self.get_distance(car.point, trip.o)

        # Total distance
        total_distance = distance_pickup + distance_trip

        revenue = self.config.calculate_fare(
            distance_trip, sq_class=trip.sq_class
        )

        duration_pickup_min, duration_pickup_step = self.get_travel_time_tuple(
            distance_pickup
        )

        duration_trip_min = self.get_travel_time(distance_trip, unit="min")

        total_duration_min = duration_pickup_min + duration_trip_min

        duration_total_step = self.get_travel_time(total_distance, unit="step")

        car.update_trip(
            duration_pickup_min,
            total_duration_min,
            total_distance,
            revenue,
            trip,
            duration_pickup_step=duration_pickup_step,
            duration_total_step=duration_total_step,
            time_increment=self.config.time_increment,
        )

        return duration_pickup_min, total_duration_min, total_distance, revenue

    # ################################################################ #
    # Decision ####################################################### #
    # ################################################################ #

    def realize_decision(self, t, decisions, a_trips, dict_a_cars):
        total_reward = 0
        serviced = list()

        for decision in decisions:

            action, point, level, o, d, times = decision

            # Trip attribute
            # od = (o, d)
            # list_trips_in_decision = trips_per_attribute[od]

            cars_with_attribute = dict_a_cars[(point, level)]

            for n, car in enumerate(cars_with_attribute):

                # Only 'times' cars will execute decision
                # determined in action 'a'
                if n >= times:
                    break

                if action == du.RECHARGE_DECISION:
                    # Recharging #######################################

                    cost_recharging = self.recharge(
                        car, self.config.recharge_time_single_level
                    )

                    # Subtract cost of recharging
                    total_reward -= cost_recharging

                elif action == du.REBALANCE_DECISION:
                    # Rebalancing ######################################

                    duration, distance, reward = self.rebalance(
                        car, self.points[d]
                    )

                    car.move(duration, distance, reward, self.points[d])

                elif action == du.STAY_DECISION:
                    # car.step += 1
                    pass

                else:
                    # Servicing ########################################
                    dist_list = [
                        self.get_distance(car.point, trip.o)
                        for trip in a_trips[(o, d)]
                    ]

                    # Get closest trip
                    iclosest_pk = np.argmin(dist_list)

                    # Get a trip to apply decision
                    trip = a_trips[(o, d)].pop(iclosest_pk)

                    duration, distance, reward = self.pickup(trip, car)

                    car.update_trip(duration, distance, reward, trip)

                    serviced.append(trip)
                    total_reward += reward

            # Remove cars already used to fulfill decisions
            cars_with_attribute = cars_with_attribute[times:]

        return (
            total_reward,
            serviced,
            list(it.chain.from_iterable(a_trips.values())),
        )

    def preview_decision(self, time_step, decision):
        """Apply decision to attribute

        Arguments:
            time_step {int} -- [description]
            decision {tuple} -- (
                 action = {rebalance, recharge, stay, service},
                  point = id car position,
                battery = attribute battery level,
                      o = decision pickup
                      d = decision delivery
            )

        Returns:
            tuple -- time_step, point, battery
        """

        action, point, battery, o, d = decision
        battery_post = battery
        if action == du.RECHARGE_DECISION:

            # Recharging ###############################################
            battery_post = min(self.battery_levels, battery + 1)
            time_step += self.config.recharge_time_single_level

        elif action == du.REBALANCE_DECISION:
            # Rebalancing ##############################################

            duration, battery_drop = self.preview_move(point, o, d)
            time_step += max(1, duration)
            battery_post = max(0, battery - battery_drop)
            point = d

        elif action == du.STAY_DECISION:
            # Staying ##################################################
            time_step += 1

        else:
            # Servicing ################################################
            duration, battery_drop = self.preview_move(point, o, d)
            time_step += max(1, duration)
            battery_post = max(0, battery - battery_drop)
            point = d

        return time_step, point, battery_post

    # @functools.lru_cache(maxsize=None)
    def preview_move(self, car_pos, o, d):

        # Distance in km
        distance = self.get_distance(self.points[o], self.points[d])

        # Car is not in the same zone, of pickup point. Therefore, it
        # has to drive there first
        if car_pos != o:
            distance += self.get_distance(self.points[car_pos], self.points[o])

        dropped_levels = int(
            round(distance / self.config.battery_distance_level)
        )

        duration = self.get_travel_time(distance, unit="step")

        return duration, dropped_levels

    # # @functools.lru_cache(maxsize=None)
    def get_travel_time_tuple(self, distance):
        """Travel time in minutes and steps given distance in meters"""

        travel_time_h = distance / self.config.speed
        travel_time_min = travel_time_h * 60
        steps = math.ceil(travel_time_min / self.config.time_increment)
        return travel_time_min, steps

    # @functools.lru_cache(maxsize=None)
    def get_travel_time(self, distance, unit="min"):
        """Travel time in minutes given distance in miles"""

        travel_time_h = distance / self.config.speed
        travel_time_min = travel_time_h * 60

        if unit == "min":
            return travel_time_min
        else:
            steps = math.ceil(travel_time_min / self.config.time_increment)
            return steps

    def get_distance_time(self, time):
        """Distance in kilometers given time in minutes"""

        distance = time * self.config.speed / 60
        return distance

    # @functools.lru_cache(maxsize=None)
    def get_travel_time_od(self, o, d, unit="min"):
        """Travel time in minutes or steps between od"""
        distance = self.get_distance(o, d)
        duration = self.get_travel_time(distance, unit=unit)
        return duration

    def get_unreachable_ods(self):
        """Unreachable ods have NO rebalancing neighbors.

        Returns
        -------
        set
            Nodes with no neighbors
        """
        unreachable_ods = set()
        neighbors = {}

        # Loop all origin nodes
        for i in self.point_ids_level[self.config.centroid_level]:

            # Get all neighbhors
            neigh = set(self.get_zone_neighbors(i))

            # No neighbors found
            if not neigh or (
                self.config.min_neighbors is not None
                and len(neigh) < self.config.min_neighbors
            ):
                unreachable_ods.add(i)

            else:
                neighbors[i] = neigh

        # Eliminate all unreacheable from all neighbors
        new_ureachable = True
        while new_ureachable:

            # print(
            #     f"## Neighbors ({len(neighbors)}) "
            #     f"- min.:{self.config.min_neighbors}"
            # )
            # pprint(neighbors)
            # print("Unreachable:", len(unreachable_ods))

            new_ureachable = False
            new_neighbors = dict()
            for n, reachable in neighbors.items():
                new_reachable = reachable - unreachable_ods

                # No neighbors found
                if not new_reachable or (
                    self.config.min_neighbors is not None
                    and len(new_reachable) < self.config.min_neighbors
                ):
                    new_ureachable = True
                    unreachable_ods.add(n)

                else:
                    new_neighbors[n] = new_reachable

            neighbors = deepcopy(new_neighbors)

        return unreachable_ods, neighbors

    # ################################################################ #
    # Prints ######################################################### #
    # ################################################################ #

    def print_environment(self):
        """Print environment zones, points, and cars"""
        print("\nZones:")
        pprint(self.zones)

        print("\nLocations:")
        pprint(self.points)

        print("\nFleet:")
        pprint(self.cars)

    def print_fleet_stats(self, filter_status=[]):
        fleet_stats_str = ""

        # Count how many car per status
        for c in self.cars:
            if filter_status and c.status not in filter_status:
                continue

            fleet_stats_str += c.status_log() + "\n"

        return fleet_stats_str

    def get_fleet_stats_summary(self):
        count_status = dict()

        # Start all car statuses with 0
        for s in Car.status_list:
            count_status[s] = 0

        # Count how many car per status
        for c in self.cars:
            count_status[c.status] += 1

        return dict(count_status)

    def print_car_traces(self):
        for c in self.cars:
            print(f"# {c}")
            pprint(c.point_list)

    # ################################################################ #
    # Save/Load ###################################################### #
    # ################################################################ #

    def reset(self, seed=None):

        # print(
        #     f"Centroid origins: {len(self.point_ids_level[self.config.centroid_level])}"
        # )

        # print(
        #     f"Centroid origins (removed unreachable): {len(self.reachable_points)}"
        # )

        new_origins = []

        if self.config.cars_start_from_initial_positions:

            new_origins = self.car_origin_points

        elif self.cars and self.config.cars_start_from_last_positions:
            # Method: https:// doi.org/10.1287/trsc.1080.0238
            # Cars start in the last visited positions:
            #  - The final resource state vector (R_Ta) is the initial
            #    resource state vector
            new_origins = [c.point for c in self.cars]

        elif self.config.cars_start_from_random_positions:
            if seed is not None:
                print(
                    f"Starting fleet throughout {len(self.reachable_points)} random points (random seed: {seed})..."
                )
                random.seed(seed + 1)

            new_origins = random.choices(
                self.reachable_points, k=self.fleet_size
            )

            # Guarantee no more than "max_cars" start from node origins
            _, counts = np.unique(
                [o.id for o in new_origins], return_counts=True
            )

            # Get random origin points until a max. number of cars
            # start from each node
            while max(counts) > self.config.max_cars_link:
                print(
                    f"Flooded area (points={len(self.reachable_points)}, "
                    f"max={max(counts)} cars), finding new origins..."
                )

                new_origins = random.choices(
                    self.reachable_points, k=self.fleet_size
                )
                _, counts = np.unique(
                    [o.id for o in new_origins], return_counts=True
                )

        elif self.config.cars_start_from_rejected_trip_origins:
            # Start from rejected trip origins

            if seed is not None:
                print(
                    f"Starting fleet at rejected trip origins"
                    f" (random seed={seed}, "
                    f"origins={len(self.rejected_trip_origins)})..."
                )
                random.seed(seed + 1)

            # Start with random origins
            new_origins = random.choices(
                self.reachable_points, k=self.fleet_size
            )

            # Slice reject trip origins (maximum size)
            rejected_trip_origins = [
                self.points[o]
                for o in self.rejected_trip_origins[: self.fleet_size]
            ]

            # Add rejected to random
            new_origins[0 : len(rejected_trip_origins)] = rejected_trip_origins

        elif self.config.cars_start_from_last_trip_origins:
            # Start from rejected trip origins

            if seed is not None:
                print(
                    f"Starting fleet from last trip origins"
                    f" (random seed={seed}, "
                    f"origins={len(self.last_trip_origins)})..."
                )
                random.seed(seed + 1)

            # Start with random origins
            new_origins = random.choices(
                self.reachable_points, k=self.fleet_size
            )

            # Slice reject trip origins (maximum size)
            # Trips are assumed to start from valid nodes
            last_trip_origins = [
                self.points[o]
                for o in self.last_trip_origins[: self.fleet_size]
            ]

            # Add rejected to random
            new_origins[0 : len(last_trip_origins)] = last_trip_origins

        elif self.config.cars_start_from_parking_lots:
            # Start from rejected trip origins

            if seed is not None:
                print(
                    f"Starting fleet from parking lots"
                    f" (random seed={seed}, "
                    f"origins={len(self.level_parking_points)})..."
                )
                random.seed(seed + 1)

            # Chose parking lots randomly
            new_origins = random.choices(
                self.level_parking_points, k=self.fleet_size
            )

        Car.count = 0

        # Cars start from centroids
        if (
            self.config.centroid_level > 0
            and not self.config.cars_start_from_parking_lots
        ):

            print(
                f"{len(new_origins)} centroid origins "
                f"(level={self.config.centroid_level})"
            )

        self.cars = [
            Car(
                point,
                # self.config.battery_levels,
                # battery_level_miles_max=self.config.battery_size_distances,
            )
            for point in [point for point in new_origins]
        ]

        self.available = self.cars

        self.rebalancing = []

    def car_neigh_stats(self):
        neigh_cars = []
        od_times = []
        for c in self.cars:

            neigh = self.neighbors[
                c.point.id_level(self.config.centroid_level)
            ]
            avg_reb_time = 0
            for n in neigh:
                od = self.get_travel_time_od(c.point, self.points[n])

                avg_reb_time += od
            avg_reb_time = avg_reb_time / len(neigh)

            # OD rebalance time
            od_times.append(avg_reb_time)
            neigh_cars.append(len(neigh))

        # self.cars = [
        #     Car(
        #         point,
        #         self.config.battery_levels,
        #         battery_level_miles_max=self.config.battery_size_distances,
        #     )
        #     for point in self.car_origin_points
        # ]

        return neigh_cars, od_times
