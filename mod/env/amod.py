from mod.env.car import Car
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

port = 4999
url = f"http://localhost:{port}"

class Amod:
    # Decision codes

    # In a zoned environment with (z1, z2) cells signals:
    #  - trip from z1 to z2
    #  - stay in zone z1 = z2
    TRIP_STAY_DECISION = 'XXX'

    TRIP_DECISION = 'T'

    STAY_DECISION = 'S'

    # In a zoned environment with (z1, z2) cells signals:
    #  - rebalance from z1 to z2
    #  - recharge in zone z1 = z2
    RECHARGE_REBALANCE_DECISION = 'YYY'

    RECHARGE_DECISION = 'P'

    REBALANCE_DECISION = 'R'

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
        self.config = config
        self.time_steps = config.time_steps

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

        # aggregation level -> point id -> point object
        # self.dict_points = defaultdict(list)
        # for p in self.points:
        #     for g in range(self.config.aggregation_levels):
        #         self.dict_points[g].append(p)

        # ------------------------------------------------------------ #
        # Battery ######################################################
        # -------------------------------------------------------------#

        self.battery_levels = config.battery_levels
        self.battery_size_distances = config.battery_size_distances
        self.fleet_size = config.fleet_size

        # ------------------------------------------------------------ #
        # Fleet ########################################################
        # -------------------------------------------------------------#

        self.car_origin_points = [
            point for point in random.choices(self.points, k=self.fleet_size)
        ]

        # If no list predefined positions
        if not car_positions:
            # Creating random fleet starting from random points
            self.cars = [
                Car(
                    point,
                    self.battery_levels,
                    battery_level_miles_max=self.battery_size_distances,
                )
                for point in self.car_origin_points
            ]
        else:
            # Creating fleet starting from pre-determined positions
            self.cars = [
                Car(
                    point,
                    self.battery_levels,
                    battery_level_miles_max=self.battery_size_distances,
                )
                for point in car_positions
            ]

        # -------------------------------------------------------------#
        # Learning #####################################################
        # -------------------------------------------------------------#

        # What is the value of a car attribute assuming aggregation
        # level and time steps
        self.values = defaultdict(
            lambda: defaultdict(lambda: defaultdict(float))
        )

        # How many times a cell was actually accessed by a vehicle in
        # a certain region, aggregation level, and time
        self.count = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))

        # Averaging weights each round
        self.counts = np.zeros(self.config.aggregation_levels)
        self.weight_track = np.zeros(self.config.aggregation_levels)

        self.current_weights = np.array([])

        # -------------------------------------------------------------#
        # Weighing #####################################################
        # -------------------------------------------------------------#

        # Transient bias
        self.transient_bias = defaultdict(
            lambda: defaultdict(lambda: defaultdict(float))
        )

        self.variance_g = defaultdict(
            lambda: defaultdict(lambda: defaultdict(float))
        )

        self.step_size_func = defaultdict(
            lambda: defaultdict(lambda: defaultdict(float))
        )

        self.lambda_stepsize = defaultdict(
            lambda: defaultdict(lambda: defaultdict(float))
        )

        # Aggregation bias
        self.aggregation_bias = defaultdict(
            lambda: defaultdict(lambda: defaultdict(float))
        )

    def reset(self):

        self.cars = [
            Car(
                point,
                self.config.battery_levels,
                battery_level_miles_max=self.config.battery_size_distances,
            )
            for point in [
                point
                for point in random.choices(self.points, k=self.fleet_size)
            ]
        ]

        # self.cars = [
        #     Car(
        #         point,
        #         self.config.battery_levels,
        #         battery_level_miles_max=self.config.battery_size_distances,
        #     )
        #     for point in self.car_origin_points
        # ]

    ####################################################################
    # Cost functions ###################################################
    ####################################################################

    @lru_cache(maxsize=None)
    def cost_func(self, action, o, d):

        if action == Amod.STAY_DECISION:
            # Stay
            return 0
        
        elif action == Amod.TRIP_DECISION:
            # Pick up
            distance_trip = self.get_distance(
                self.points[o], self.points[d]
            )

            reward = self.config.calculate_fare(distance_trip)
            return reward

        elif action == Amod.RECHARGE_DECISION:
            # Recharge
            cost = self.config.cost_recharge_sigle_increment
            return -cost
            
        else:
            # Rebalance
            return 0

    def post_cost(self, t, decision, level=None):
        if level:
            return self.get_value(t, decision, level=level)
        else:
            return self.get_weighted_value(t, decision)

    ####################################################################
    # Smoothed #########################################################
    ####################################################################

    def get_weighted_value(self, t, decision):

        # Target attribute if decision was taken
        post_t, post_pos, post_battery = self.preview_decision(t, decision)

        # Get point object associated to position
        point = self.points[post_pos]

        v_ta_0 = (
            self.values[post_t][0][(post_pos, post_battery)]
            if post_t in self.values
            and 0 in self.values[post_t]
            and (post_pos, post_battery) in self.values[post_t][0]
            else 0
        )

        weight_vector = np.zeros(self.config.aggregation_levels)
        value_vector = np.zeros(self.config.aggregation_levels)

        for g in range(self.config.aggregation_levels):

            # Find attribute at level g
            ta_g = (point.id_level(g), post_battery)

            # Current value function of attribute at level g
            value_vector[g] = (
                self.values[post_t][g][ta_g]
                if post_t in self.values
                and 0 in self.values[post_t]
                and (post_pos, post_battery) in self.values[post_t][0]
                else 0
            )

            # WEIGHTING ############################################

            # Bias due to aggregation error = v[-,a, g] - v[-, a, 0]
            aggregation_bias = value_vector[g] - v_ta_0

            # Bias due to smoothing of transient data series (value
            # function change every iteration)
            transient_bias = self.get_transient_bias(
                post_t,
                g,
                ta_g,
                v_ta_0,
                value_vector[g],
                self.step_size_func[post_t][g][ta_g],
            )

            variance_g = self.get_variance_g(
                post_t, g, ta_g, v_ta_0, value_vector[g], self.config.stepsize
            )

            # Lambda stepsize from iteration n-1
            lambda_step_size = self.lambda_stepsize[post_t][g][ta_g]

            # Estimate of the variance of observations made of state
            # s, using data from aggregation level g, after n
            # observations.
            variance_error = self.get_total_variance(
                variance_g, transient_bias, lambda_step_size
            )

            # Variance of our estimate of the mean v[-,s,g,n]
            variance = lambda_step_size * variance_error

            # Total variation (variance plus the square of the bias)
            total_variation = variance + (aggregation_bias ** 2)

            if total_variation == 0:
                weight_vector[g] = 0
            else:
                weight_vector[g] = 1 / total_variation

        if len(np.unique(value_vector)) <= 1:
            value_estimation = 0
        else:
            weight_vector = weight_vector / sum(weight_vector)
            value_estimation = sum(
                np.prod([weight_vector, value_vector], axis=0)
            )

        self.counts += 1
        self.weight_track += weight_vector

        return value_estimation

    def get_total_variance(
        self, total_variation, transient_bias, lambda_stepsize
    ):
        return (total_variation - (transient_bias ** 2)) / (
            1 + lambda_stepsize
        )

    def get_variance_g(self, t, g, a_g, v, v_g, fixed_stepsize):

        # We now need to compute s^2[a,g] which is the estimate of the
        # variance of observations (v) for states (a) for which
        # G(a) = a_g (the observations of states that aggregate up
        # to a).

        return (
            (1 - fixed_stepsize) * self.variance_g[t][g][a_g]
        ) + fixed_stepsize * ((v - v_g) ** 2)

    def get_lambda_stepsize(self, t, a_g, g, stepsize, lambda_stepsize):

        return (((1 - stepsize) ** 2) * lambda_stepsize) + (stepsize ** 2)

    def get_transient_bias(self, t, g, a_g, v, v_g, stepsize):

        # The transient bias (due to smoothing): When we smooth on past
        # observations, we obtain an estimate v[-,s,g,n-1] that tends to
        #  underestimate (or overestimate if v(^,n) tends to decrease)
        # the true mean of v[^,n].
        transient_bias = (1 - stepsize) * self.transient_bias[t][g][
            a_g
        ] + stepsize * (v - v_g)

        return transient_bias

    def get_weights(self):

        avg = self.weight_track / self.counts
        total = sum(avg)
        if total > 0:
            avg = avg / total
        self.weight_track = np.zeros(self.config.aggregation_levels)
        self.counts = np.zeros(self.config.aggregation_levels)

        return avg

    def update_values_smoothed(self, t, duals):

        for (pos, battery), v_ta in duals.items():

            # Get point object associated to position
            point = self.points[pos]

            for g in range(self.config.aggregation_levels):

                # Find attribute at level g
                a_g = (point.id_level(g), battery)

                # Step size from previous iteration
                current_stepsize = self.step_size_func[t][g][a_g]

                # Current value function of attribute at level g
                v_ta_g = self.values[t][g][a_g]

                # WEIGHTING ############################################

                # Updating

                # Account for a_g
                self.count[t][g][a_g] += 1

                # TODO what if several values aggregate up to a_g???
                new_stepsize = 1 / self.count[t][g][a_g]
                self.step_size_func[t][g][a_g] = new_stepsize

                # Bias due to smoothing of transient data series (value
                # function change every iteration)
                self.transient_bias[t][g][a_g] = self.get_transient_bias(
                    t, g, a_g, v_ta, v_ta_g, self.config.stepsize
                )

                self.variance_g[t][g][a_g] = self.get_variance_g(
                    t, g, a_g, v_ta, v_ta_g, self.config.stepsize
                )

                self.lambda_stepsize[t][g][a_g] = self.get_lambda_stepsize(
                    t, a_g, g, current_stepsize, new_stepsize
                )

                # Update value function at gth level with smoothing
                self.values[t][g][a_g] = (
                    1 - self.step_size_func[t][g][a_g]
                ) * v_ta_g + self.step_size_func[t][g][a_g] * v_ta

    ####################################################################
    # True averaging ###################################################
    ####################################################################

    def get_value(self, step, decision, level=0):

        # Target attribute if decision was taken
        d_step, d_pos, d_battery_level = self.preview_decision(step, decision)

        # Point associated to position at disaggregate level
        point = self.points[d_pos]

        # Attribute considering aggregation level
        attribute = (point.id_level(level), d_battery_level)

        # Value function
        value = self.values[d_step][level][attribute]

        return value

    def averaged_update(self, step, duals):
        """Update values without smoothing

        Arguments:
            step {int} -- Current time step
            duals {dict} -- Dictionary of attribute tuples and duals
        """

        for (pos, battery), new_vf_0 in duals.items():

            # Get point object associated to position
            point = self.points[pos]

            for g in range(self.config.aggregation_levels):

                # Find attribute at level g
                attribute_g = (point.id_level(g), battery)

                # Current value function of attribute at level g
                current_vf = self.values[step][g][attribute_g]

                # Compute attribute access
                self.count[step][g][attribute_g] += 1

                # Incremental averaging
                count = self.count[step][g][attribute_g]
                increment = (new_vf_0 - current_vf) / count

                # Update attribute mean value
                self.values[step][g][attribute_g] += increment

    ####################################################################
    # Network ##########################################################
    ####################################################################

    @lru_cache(maxsize=None)
    def get_travel_time(self, distance, unit="min"):
        """Travel time in minutes given distance in miles"""

        travel_time_h = distance / self.config.speed
        travel_time_min = travel_time_h * 60

        if unit == "min":
            return travel_time_min
        else:
            steps = int(round(travel_time_min / self.config.time_increment))
            return steps

    @lru_cache(maxsize=None)
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

    def rebalance(self, car, target):

        # Distance car has to travel to rebalance
        distance = self.get_distance(car.point, target)

        # Next arrival
        duration_min = int(round(self.get_travel_time(distance)))

        # No reward for rebalancing
        reward = 0

        return duration_min, distance, reward

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

        # Reward
        reward = self.config.calculate_fare(distance_trip)

        # Next arrival
        duration_min = int(round(self.get_travel_time(total_distance)))

        return duration_min, total_distance, reward

    @lru_cache(maxsize=None)
    def get_neighbors(self, center, level=0, n_neighbors=4):
        return nw.get_neighbor_zones(
            center, self.config.pickup_zone_range, self.zones
        )

    ####################################################################
    # Recharge #########################################################
    ####################################################################

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
        car.update_recharge(time_min, cost)

        return cost

    def recharge(self, car, time_min):

        # Total cost of recharging
        cost = self.config.calculate_cost_recharge(time_min)

        # Update vehicle status to recharging
        car.update_recharge(
            time_min,
            cost,
            time_increment=self.config.time_increment
        )

        return cost

    ####################################################################
    # Decision #########################################################
    ####################################################################

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

                if action == Amod.RECHARGE_DECISION:
                    # Recharging #######################################

                    cost_recharging = self.recharge(
                        car, self.config.recharge_time_single_level
                    )

                    # Subtract cost of recharging
                    total_reward -= cost_recharging

                elif action == Amod.REBALANCE_DECISION:
                    # Rebalancing ######################################

                    duration, distance, reward = self.rebalance(
                        car, self.points[d]
                    )

                    car.move(
                        duration,
                        distance,
                        reward,
                        self.points[d],
                        time_increment=self.config.time_increment
                    )

                elif action == Amod.STAY_DECISION:
                    # car.step += 1
                    pass

                else:
                    # Servicing ########################################

                    # Get closest trip
                    iclosest_pk = np.argmin(
                        [
                            self.get_distance(car.point, trip.o)
                            for trip in a_trips[(o, d)]
                        ]
                    )

                    # Get a trip to apply decision
                    trip = a_trips[(o, d)].pop(iclosest_pk)
                    # trip = a_trips[(o, d)].pop()

                    duration, distance, reward = self.pickup(trip, car)

                    car.update_trip(
                        duration,
                        distance,
                        reward,
                        trip,
                        time_increment=self.config.time_increment
                    )

                    serviced.append(trip)
                    total_reward += reward

            # Remove cars already used to fulfill decisions
            cars_with_attribute = cars_with_attribute[times:]

        return (
            total_reward,
            serviced,
            list(
                it.chain.from_iterable(a_trips.values())
            ),
        )

    @lru_cache(maxsize=None)
    def preview_move(self, car_pos, o, d):

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
        if action == Amod.RECHARGE_DECISION:

            # Recharging ###############################################
            battery_post = min(self.battery_levels, battery + 1)
            time_step += self.config.recharge_time_single_level

        elif action == Amod.REBALANCE_DECISION:
            # Rebalancing ##############################################

            duration, battery_drop = self.preview_move(point, o, d)
            time_step += max(1, duration)
            battery_post = max(0, battery - battery_drop)
            point = d

        elif action == Amod.STAY_DECISION:
            # Staying ##################################################
            time_step += 1

        else:
            # Servicing ################################################
            duration, battery_drop = self.preview_move(point, o, d)
            time_step += max(1, duration)
            battery_post = max(0, battery - battery_drop)
            point = d

        return time_step, point, battery_post

    ####################################################################
    # Prints ###########################################################
    ####################################################################

    def print_environment(self):
        """Print environment zones, points, and cars"""
        print("\nZones:")
        pprint(self.zones)

        print("\nLocations:")
        pprint(self.points)

        print("\nFleet:")
        pprint(self.cars)

    def print_fleet_stats(self):
        count_status = dict()

        # Start all car statuses with 0
        for s in Car.status_list:
            count_status[s] = 0

        # Count how many car per status
        for c in self.cars:
            print(c.status_log())
            count_status[c.status] += 1

        pprint(dict(count_status))

    def get_fleet_status(self):
        """Number of cars per status and total battery level
        in miles.

        Returns:
            dict, float -- #cars per status, total battery level
        """
        status_count = defaultdict(int)
        total_battery_level = 0
        for c in self.cars:
            total_battery_level += c.battery_level_miles
            status_count[c.status] += 1
        return status_count, total_battery_level

    def print_car_traces(self):
        for c in self.cars:
            print(f'# {c}')
            pprint(c.point_list)

    def print_car_traces_geojson(self):
        for c in self.cars:
            for o, d in zip(c.point_list[:-1], c.point_list[1:]):
                url_neighbors = f"{url}/sp_coords/{o}/{d}"
                r = requests.get(url=url_neighbors)
                traces = r.text.split(";")

    ####################################################################
    # Save/Load ########################################################
    ####################################################################
    def load_progress(self, progress):
        self.values = progress['values']
        self.count = progress['counts']
        self.transient_bias = progress['transient_bias']
        self.variance_g = progress['variance_g']
        self.step_size_func = progress['stepsize']
        self.lambda_stepsize = progress['lambda_stepsize']
        self.aggregation_bias = progress['aggregation_bias']

    def load_episode(self, path):
        """Load .npy dictionary containing value functions of last
        episode.

        Arguments:
            path {str} -- File with saved value functions
        """
        values_old = np.load(
            FOLDER_EPISODE_TRACK + self.config.label + ".npy"
        ).item()
        # print(values_old)
        for t, g_a in values_old.items():
            for g, a_value in g_a.items():
                for a, value in a_value.items():
                    self.values[t][g][a] = value


class AmodNetwork(Amod):
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

        super().__init__(config)

        self.config = config
        self.time_steps = config.time_steps

        # ------------------------------------------------------------ #
        # Network ######################################################
        # -------------------------------------------------------------#

        # Defining the operational map
        self.n_zones = self.config.rows * self.config.cols
        zones = np.arange(self.n_zones)
        self.zones = zones.reshape((self.config.rows, self.config.cols))

        # Defining map points with aggregation_levels
        self.points, distance_levels = nw.query_point_list(
            step=self.config.step_seconds,
            max_levels=self.config.aggregation_levels,
            projection=self.config.projection,
            level_dist_list=self.config.level_dist_list,
        )

        # Levels correspond to distances queried in the server.
        # E.g., [0, 30, 60, 120, 300]
        Point.levels = sorted(distance_levels)

        # ------------------------------------------------------------ #
        # Battery ######################################################
        # -------------------------------------------------------------#

        self.battery_levels = config.battery_levels
        self.battery_size_distances = config.battery_size_distances
        self.fleet_size = config.fleet_size

        # ------------------------------------------------------------ #
        # Fleet ########################################################
        # -------------------------------------------------------------#

        self.car_origin_points = [
            point for point in random.choices(self.points, k=self.fleet_size)
        ]

        # If no list predefined positions
        if not car_positions:
            # Creating random fleet starting from random points
            self.cars = [
                Car(
                    point,
                    self.battery_levels,
                    battery_level_miles_max=self.battery_size_distances,
                )
                for point in self.car_origin_points
            ]
        else:
            # Creating fleet starting from pre-determined positions
            self.cars = [
                Car(
                    point,
                    self.battery_levels,
                    battery_level_miles_max=self.battery_size_distances,
                )
                for point in car_positions
            ]

        # -------------------------------------------------------------#
        # Learning #####################################################
        # -------------------------------------------------------------#

        # What is the value of a car attribute assuming aggregation
        # level and time steps
        self.values = defaultdict(
            lambda: defaultdict(lambda: defaultdict(float))
        )

        # How many times a cell was actually accessed by a vehicle in
        # a certain region, aggregation level, and time
        self.count = defaultdict(
            lambda: defaultdict(lambda: defaultdict(int))
        )

        # Averaging weights each round
        self.counts = np.zeros(self.config.aggregation_levels)
        self.weight_track = np.zeros(self.config.aggregation_levels)

        self.current_weights = np.array([])

        # -------------------------------------------------------------#
        # Weighing #####################################################
        # -------------------------------------------------------------#

        # Transient bias
        self.transient_bias = defaultdict(
            lambda: defaultdict(lambda: defaultdict(float))
        )

        self.variance_g = defaultdict(
            lambda: defaultdict(lambda: defaultdict(float))
        )

        self.step_size_func = defaultdict(
            lambda: defaultdict(lambda: defaultdict(float))
        )

        self.lambda_stepsize = defaultdict(
            lambda: defaultdict(lambda: defaultdict(float))
        )

        # Aggregation bias
        self.aggregation_bias = defaultdict(
            lambda: defaultdict(lambda: defaultdict(float))
        )

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

    def get_neighbors(self, center_point, reach=1):
        return nw.query_neighbors(center_point.id, reach=reach)
    
    def get_zone_neighbors(self, center, level=0, n_neighbors=4):
        step = Point.levels[level]
        return nw.query_neighbor_zones(
            center.level_ids_dic[step],
            step,
            n_neighbors=n_neighbors,
        )

    def get_level_neighbors(self, center, level):
        return nw.query_level_neighbors(
            center.id_level(level), Point.levels[level]
        )

    def get_region_elements(self, center, level):
        return nw.query_level_neighbors(
            center, Point.levels[level]
        )

    @lru_cache(maxsize=None)
    def get_travel_time(self, distance_km, unit="min"):
        """Travel time in minutes given distance in miles"""

        travel_time_h = distance_km / self.config.speed
        travel_time_min = travel_time_h * 60

        if unit == "min":
            return travel_time_min
        else:
            steps = int(round(travel_time_min / self.config.time_increment))
            return steps

    # def realize_decision(self, t, decisions, a_trips, dict_a_cars):
    #     total_reward = 0
    #     serviced = list()

    #     for decision in decisions:

    #         action, point, level, o, d, times = decision

    #         # Trip attribute
    #         # od = (o, d)
    #         # list_trips_in_decision = trips_per_attribute[od]

    #         cars_with_attribute = dict_a_cars[(point, level)]

    #         for n, car in enumerate(cars_with_attribute):

    #             # Only 'times' cars will execute decision
    #             # determined in action 'a'
    #             if n >= times:
    #                 break

    #             if action == Amod.RECHARGE_REBALANCE_DECISION:
    #                 # Recharging #######################################
    #                 if o == d:
    #                     cost_recharging = self.recharge(
    #                         car, self.config.time_increment
    #                     )

    #                     # Subtract cost of recharging
    #                     total_reward -= cost_recharging

    #                 # Rebalancing ######################################
    #                 else:
    #                     duration, distance, reward = self.rebalance(
    #                         car, self.points[d]
    #                     )

    #                     car.move(duration, distance, reward, self.points[d])

    #             elif action == Amod.TRIP_STAY_DECISION:
    #                 # Staying ##########################################
    #                 if o == d:
    #                     car.step += 1

    #                 # Servicing ########################################
    #                 else:

    #                     # Get closest trip
    #                     iclosest_pk = np.argmin(
    #                         [
    #                             self.get_distance(car.point, trip.o)
    #                             for trip in a_trips[(o, d)]
    #                         ]
    #                     )

    #                     # Get a trip to apply decision
    #                     trip = a_trips[(o, d)].pop(iclosest_pk)

    #                     duration, distance, reward = self.pickup(trip, car)

    #                     car.update_trip(duration, distance, reward, trip)

    #                     serviced.append(trip)

    #                     total_reward += reward

    #         # Remove cars already used to fulfill decisions
    #         cars_with_attribute = cars_with_attribute[times:]

    #     return (
    #         total_reward,
    #         serviced,
    #         list(it.chain.from_iterable(a_trips.values())),
    #     )
