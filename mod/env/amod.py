from mod.env.car import Car
from mod.env.trip import Trip
from mod.env.network import Point, get_point_list, get_neighbor_zones
import itertools as it
from collections import defaultdict
import numpy as np
import random
from pprint import pprint
from mod.env.config import FOLDER_EPISODE_TRACK
from functools import lru_cache


class Amod:
    # Decision codes

    # In a zoned environment with (z1, z2) cells signals:
    #  - trip from z1 to z2
    #  - stay in zone z1 = z2
    TRIP_STAY_DECISION = 0

    # In a zoned environment with (z1, z2) cells signals:
    #  - rebalance from z1 to z2
    #  - recharge in zone z1 = z2
    RECHARGE_REBALANCE_DECISION = 1

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

        # Defining the operational map
        self.n_zones = self.config.rows * self.config.cols
        zones = np.arange(self.n_zones)
        self.zones = zones.reshape((self.config.rows, self.config.cols))

        # Defining map points with aggregation_levels
        self.points = get_point_list(
            self.config.rows,
            self.config.cols,
            levels=self.config.aggregation_levels,
        )
        # What is the value of a car attribute assuming aggregation
        # level and time steps
        self.values = defaultdict(
            lambda: defaultdict(lambda: defaultdict(float))
        )

        self.weighted_values = defaultdict(lambda: defaultdict(float))

        # How many times a cell was actually accessed by a vehicle in
        # a certain region, aggregation level, and time
        self.count = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
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

        self.squared_variation = defaultdict(
            lambda: defaultdict(lambda: defaultdict(float))
        )

        # Step size (η) - might be a constant
        self.step_size = 0.1

        # Step size (η) - might be a constant
        self.lambda_step_size = defaultdict(
            lambda: defaultdict(
                lambda: defaultdict(lambda: self.step_size ** 2)
            )
        )

        # Aggregation bias
        self.aggregation_bias = defaultdict(
            lambda: defaultdict(lambda: defaultdict(float))
        )

        # Weights of aggregated levels
        self.weight = defaultdict(
            lambda: defaultdict(lambda: defaultdict(float))
        )

        # aggregation level -> point id -> point object
        self.dict_points = defaultdict(dict)
        for p in self.points:
            for g in range(self.config.aggregation_levels):
                self.dict_points[g][p.id_level(g)] = p

        # Battery levels -- l^{d}
        self.battery_levels = config.battery_levels
        self.battery_size_miles = config.battery_size_miles
        self.fleet_size = config.fleet_size
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
                    battery_level_miles_max=self.battery_size_miles,
                )
                for point in self.car_origin_points
            ]
        else:
            # Creating fleet starting from pre-determined positions
            self.cars = [
                Car(
                    point,
                    self.battery_levels,
                    battery_level_miles_max=self.battery_size_miles,
                )
                for point in car_positions
            ]

    ####################################################################
    # Cost functions ###################################################
    ####################################################################

    @lru_cache(maxsize=None)
    def cost_func(self, action, o, d):

        if action == Amod.TRIP_STAY_DECISION:

            # Stay
            if o == d:
                return 0

            # Pick up
            else:
                distance_trip = self.get_distance(
                    self.points[o], self.points[d]
                )

                reward = self.config.calculate_fare(distance_trip)

                return reward

        elif action == Amod.RECHARGE_REBALANCE_DECISION:

            # Recharge
            if o == d:
                cost = self.config.cost_recharge_sigle_increment

                return -cost
            # Rebalance
            else:
                return 0

    def get_value(self, step, decision, level=0):

        # Target attribute if decision was taken
        d_step, d_pos, d_battery_level = self.preview_decision(step, decision)

        # Point associated to position at disaggregate level
        point = self.dict_points[0][d_pos]

        # Attribute considering aggregation level
        attribute = (point.id_level(level), d_battery_level)

        # Value function
        value = self.values[d_step][level][attribute]

        return value

    def get_weighted_value(self, t, decision):

        # Target attribute if decision was taken
        post_t, post_pos, post_battery = self.preview_decision(t, decision)

        # Get point object associated to position
        point = self.dict_points[0][post_pos]

        v_ta_0 = self.values[post_t][0][(post_pos, post_battery)]

        weight_vector = np.zeros(self.config.aggregation_levels)
        value_vector = np.zeros(self.config.aggregation_levels)

        for g in range(self.config.aggregation_levels):

            # Find attribute at level g
            ta_g = (point.id_level(g), post_battery)

            # Current value function of attribute at level g
            value_vector[g] = self.values[post_t][g][ta_g]

            # WEIGHTING ############################################

            # Bias due to aggregation error = v[-,a, g] - v[-, a, 0]
            aggregation_bias = value_vector[g] - v_ta_0

            # Bias due to smoothing of transient data series (value
            # function change every iteration)
            transient_bias = self.get_transient_bias(
                post_t, g, ta_g, v_ta_0, value_vector[g]
            )

            squared_variation = self.get_squared_variation(
                post_t, g, ta_g, v_ta_0, value_vector[g]
            )

            lambda_step_size = self.lambda_step_size[post_t][g][ta_g]

            # Estimate of the variance of observations made of state
            # s, using data from aggregation level g, after n
            # observations.
            variance_error = self.get_variance2(
                squared_variation, transient_bias, lambda_step_size
            )

            # Variance of our estimate of the mean v[-,s,g,n]
            variance = lambda_step_size * variance_error

            # Total variation (variance plus the square of the bias)
            total_variation = (variance + (aggregation_bias ** 2))

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

    def get_variance(
        self, transient_bias, estimated_value, previous_step_size_lambda
    ):
        return (estimated_value - (transient_bias ** 2)) / (
            1 + previous_step_size_lambda
        )

    def get_variance2(
        self, squared_variation, transient_bias, previous_step_size_lambda
    ):
        return (squared_variation - (transient_bias ** 2)) / (
            1 + previous_step_size_lambda
        )

    def update_lambda_step_size(self, t, attribute_g, g):

        previous_lambda = self.lambda_step_size[t][g][attribute_g]

        lambda_step_size = (
            (((1 - self.step_size) ** 2) * previous_lambda)
            + (self.step_size ** 2)
        )

        self.lambda_step_size[t][g][attribute_g] = lambda_step_size

    @lru_cache(maxsize=None)
    def get_transient_bias(self, t, g, pos_g_bat, value, value_g):

        # The transient bias (due to smoothing): When we smooth on past
        # observations, we obtain an estimate v[-,s,g,n-1] that tends to
        #  underestimate (or overestimate if v(^,n) tends to decrease)
        # the true mean of v[^,n].
        transient_bias = (1 - self.step_size) * self.transient_bias[t][g][
            pos_g_bat
        ] + self.step_size * (value - value_g)

        return transient_bias

    @lru_cache(maxsize=None)
    def get_squared_variation(self, t, g, pos_g_bat, value, value_g):

        # The transient bias (due to smoothing): When we smooth on past
        # observations, we obtain an estimate v[-,s,g,n-1] that tends to
        #  underestimate (or overestimate if v(^,n) tends to decrease)
        # the true mean of v[^,n].
        squared_variation = (
            (1 - self.step_size)
            * self.squared_variation[t][g][pos_g_bat]
        ) + self.step_size * ((value - value_g) ** 2)

        return squared_variation

    def get_weights(self):
        n_levels = self.config.aggregation_levels

        if self.weight_track.size > 0:

            n_weights = int(self.weight_track.size / n_levels)
            all_weights = np.reshape(self.weight_track, (n_weights, n_levels))

            average_weights = np.average(all_weights, axis=0)

            # print("# All weights:")
            # print(all_weights)

            # print("\n# Average:")
            # print(average_weights)

            self.weight_track = np.array([])
            self.counts = np.zeros(self.config.aggregate_levels)
            return average_weights

    def get_weights2(self):

        avg = self.weight_track/self.counts
        total = sum(avg)
        if total > 0:
            avg = avg/total
        self.weight_track = np.zeros(self.config.aggregation_levels)
        self.counts = np.zeros(self.config.aggregation_levels)

        return avg

    def update_value_functions(self, t, duals):

        for (pos, battery), v_ta in duals.items():

            # Get point object associated to position
            point = self.dict_points[0][pos]

            for g in range(self.config.aggregation_levels):

                # Find attribute at level g
                a_g = (point.id_level(g), battery)

                # Current value function of attribute at level g
                v_ta_g = self.values[t][g][a_g]

                # Update value function at gth level with smoothing
                self.values[t][g][a_g] = (
                    (1 - self.step_size) * v_ta_g
                    + self.step_size * v_ta
                )

                # WEIGHTING ############################################

                # Bias due to smoothing of transient data series (value
                # function change every iteration)
                transient_bias = self.get_transient_bias(
                    t, g, a_g, v_ta, v_ta_g
                )

                squared_variation = self.get_squared_variation(
                    t, g, a_g, v_ta, v_ta_g
                )

                # Updating
                self.transient_bias[t][g][a_g] = transient_bias
                self.squared_variation[t][g][a_g] = squared_variation
                self.update_lambda_step_size(t, a_g, g)

    def update_values(self, step, duals):

        for (pos, battery), new_vf_0 in duals.items():

            # Get point object associated to position
            point = self.dict_points[0][pos]

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

        travel_time_h = distance / self.config.speed_mph
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

    ####################################################################
    # Network ##########################################################
    ####################################################################

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
    def get_neighbors(self, center):
        return get_neighbor_zones(
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
        car.update_recharge(time_min, cost)

        return cost

    def reset(self):

        self.cars = [
            Car(
                point,
                self.config.battery_levels,
                battery_level_miles_max=self.config.battery_size_miles,
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
        #         battery_level_miles_max=self.config.battery_size_miles,
        #     )
        #     for point in self.car_origin_points
        # ]

    def realize_decision(
        self, time_step, decisions, trips_with_attribute, dict_attribute_cars
    ):

        total_reward = 0
        serviced = list()

        for decision in decisions:

            action, point, level, o, d, times = decision

            # Trip attribute
            # od = (o, d)
            # list_trips_in_decision = trips_per_attribute[od]

            cars_with_attribute = dict_attribute_cars[(point, level)]

            for n, car in enumerate(cars_with_attribute):

                # Only 'times' cars will execute decision
                # determined in action 'a'
                if n >= times:
                    break

                if action == Amod.RECHARGE_REBALANCE_DECISION:
                    # Recharging #######################################
                    if o == d:
                        cost_recharging = self.recharge(
                            car, self.config.time_increment
                        )

                        # Subtract cost of recharging
                        total_reward -= cost_recharging

                    # Rebalancing ######################################
                    else:
                        duration, distance, reward = self.rebalance(
                            car, self.points[d]
                        )

                        car.move(duration, distance, reward, self.points[d])

                elif action == Amod.TRIP_STAY_DECISION:
                    # Staying ##########################################
                    if o == d:
                        car.step += 1

                    # Servicing ########################################
                    else:

                        # Get a trip to apply decision
                        trip = trips_with_attribute[(o, d)].pop()

                        duration, distance, reward = self.pickup(trip, car)

                        car.update_trip(duration, distance, reward, trip)

                        serviced.append(trip)

                        total_reward += reward

            # Remove cars already used to fulfill decisions
            cars_with_attribute = cars_with_attribute[times:]

        return (
            total_reward,
            serviced,
            list(it.chain.from_iterable(trips_with_attribute.values())),
        )

    @lru_cache(maxsize=None)
    def preview_move(self, car_pos, o, d):

        distance = self.get_distance(self.points[o], self.points[d])

        # Car is not in the same zone, of pickup point. Therefore, it
        # has to drive there first
        if car_pos != o:
            distance += self.get_distance(self.points[car_pos], self.points[o])

        dropped_levels = int(round(distance / self.config.battery_miles_level))

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

        if action == Amod.RECHARGE_REBALANCE_DECISION:

            # Recharging ###############################################
            if o == d:
                # TODO unit in battery level does not take one time step!!!!
                battery = min(self.battery_levels, battery + 1)
                time_step += 1

            # Rebalancing ##############################################
            else:
                duration, battery_drop = self.preview_move(point, o, d)
                time_step += max(1, duration)
                battery = max(0, battery - battery_drop)
                point = d

        elif action == Amod.TRIP_STAY_DECISION:

            # Staying ##################################################
            if o == d:
                time_step += 1

            # Servicing ################################################
            else:
                duration, battery_drop = self.preview_move(point, o, d)
                time_step += max(1, duration)
                battery = max(0, battery - battery_drop)
                point = d

        return time_step, point, battery

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

    ####################################################################
    # Save/Load ########################################################
    ####################################################################

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
