from mod.env.car import Car, HiredCar
from mod.env.trip import Trip
from mod.env.network import Point
import mod.env.decision_utils as du
import mod.env.network as nw
import itertools as it
from collections import defaultdict
import numpy as np
import random
from pprint import pprint
from mod.env.config import FOLDER_EPISODE_TRACK
import requests
import functools


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
        self.config = config
        self.time_steps = config.time_steps

        # ------------------------------------------------------------ #
        # Battery ######################################################
        # -------------------------------------------------------------#

        self.battery_levels = config.battery_levels
        self.battery_size_distances = config.battery_size_distances
        self.fleet_size = config.fleet_size

    def init_fleet(self, points, car_positions=[]):
        # ------------------------------------------------------------ #
        # Fleet ########################################################
        # -------------------------------------------------------------#

        self.car_origin_points = [
            point for point in random.choices(
                points,
                k=self.fleet_size
            )
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

        # List of available vehicles
        self.available = self.cars

    def get_fleet_status(self):
        """Number of cars per status and total battery level
        in miles.

        Returns:
            dict, float -- #cars per status, total battery level
        """

        status_count = defaultdict(int)
        for s in Car.status_list:
            status_count[s] = 0

        total_battery_level = 0
        for c in self.cars:
            total_battery_level += c.battery_level_miles
            status_count[c.status] += 1
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

    ####################################################################
    # Learning #########################################################
    ####################################################################

    def init_learning(self):
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

    def init_weighting_settings(self):
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

        self.agg_weight_vectors = dict()

    ####################################################################
    # Cost functions ###################################################
    ####################################################################

    @functools.lru_cache(maxsize=None)
    def cost_func(self, action, o, d):

        if action == du.STAY_DECISION:
            # Stay
            return 0

        elif action == du.TRIP_DECISION:
            # Pick up
            distance_trip = self.get_distance(
                self.points[o], self.points[d]
            )

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

        if level:
            estimate = self.get_value(
                post_t,
                post_pos,
                post_battery,
                level=level
            )

        else:
            # Get the post decision state estimate value
            estimate = self.get_weighted_value(post_t, post_pos, post_battery)

        return estimate

    ####################################################################
    # Smoothed #########################################################
    ####################################################################

    def get_weights_and_agg_value(self, t, pos, battery):
        # Get point object associated to position
        point = self.points[pos]

        # Value function of level 0 in previous iteration
        v_ta_0 = (
            self.values[t][0][(pos, battery)]
            if t in self.values
            and 0 in self.values[t]
            and (pos, battery) in self.values[t][0]
            else 0
        )

        weight_vector = np.zeros(self.config.aggregation_levels)
        value_vector = np.zeros(self.config.aggregation_levels)

        for g in range(self.config.aggregation_levels):

            # Find attribute at level g
            ta_g = (point.id_level(g), battery)

            # Current value function of attribute at level g
            value_vector[g] = (
                self.values[t][g][ta_g]
                if t in self.values
                and 0 in self.values[t]
                and (pos, battery) in self.values[t][0]
                else 0
            )

            # WEIGHTING ############################################

            # Bias due to aggregation error = v[-,a, g] - v[-, a, 0]
            aggregation_bias = self.aggregation_bias[t][g][ta_g]

            # Bias due to smoothing of transient data series (value
            # function change every iteration)
            transient_bias = self.transient_bias[t][g][ta_g]

            variance_g = self.variance_g[t][g][ta_g]

            # Lambda stepsize from iteration n-1
            lambda_step_size = self.lambda_stepsize[t][g][ta_g]

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
            # Upate weight vector
            # self.agg_weight_vectors[(t, pos, battery)] = weight_vector

        return weight_vector, value_estimation

    def get_weight(self, t, g, a):

        # WEIGHTING ############################################

        # Bias due to aggregation error = v[-,a, g] - v[-, a, 0]
        aggregation_bias = self.aggregation_bias[t][g][a]

        # Bias due to smoothing of transient data series (value
        # function change every iteration)
        transient_bias = self.transient_bias[t][g][a]

        variance_g = self.variance_g[t][g][a]

        # Lambda stepsize from iteration n-1
        lambda_step_size = self.lambda_stepsize[t][g][a]

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
            return 0
        else:
            return 1 / total_variation

    def get_weighted_value(self, t, pos, battery):

        # Post decision attribute at time post_t
        a = (pos, battery)

        # Get point object associated to position
        point = self.points[pos]

        value_estimation = 0
        # Check if value function for disaggregate level exists
        if (
            t in self.values and
            0 in self.values[t] and
            a in self.values[t][0]
        ):
            # Return previosly defined disaggregated value function
            return self.values[t][0][a]

        # Calculate value estimation based on hierarchical aggregation
        else:

            # Get va
            weight_vector = np.zeros(self.config.aggregation_levels-1)
            value_vector = np.zeros(self.config.aggregation_levels-1)

            for g in range(1, self.config.aggregation_levels):

                pos_g = point.id_level(g)
                # Find attribute at level g
                a_g = (pos_g, battery)

                # Current value function of attribute at level g
                value_vector[g-1] = (
                    self.values[t][g][a_g]
                    if t in self.values and
                    0 in self.values[t] and
                    a_g in self.values[t][g]
                    else 0
                )

                weight_vector[g-1] = self.get_weight(t, g, a_g)

            # Normalize (weights have to sum up to one)
            weight_sum = sum(weight_vector)

            if weight_sum > 0:

                weight_vector = weight_vector / weight_sum

                # Get weighted value function
                value_estimation = sum(
                    np.prod([weight_vector, value_vector], axis=0)
                )

                # Update weight vector
                self.agg_weight_vectors[(t, pos, battery)] = weight_vector

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

    def get_lambda_stepsize(self, current_stepsize, lambda_stepsize):

        return (
            (((1 - current_stepsize) ** 2) * lambda_stepsize)
            +(current_stepsize ** 2)
        )

    def get_transient_bias(self, current_bias, v, v_g, stepsize):

        # The transient bias (due to smoothing): When we smooth on past
        # observations, we obtain an estimate v[-,s,g,n-1] that tends to
        #  underestimate (or overestimate if v(^,n) tends to decrease)
        # the true mean of v[^,n].
        transient_bias = (1 - stepsize) * current_bias + stepsize * (v - v_g)

        return transient_bias

    def get_weights(self, steps):

        # print("Calculating average weights")
        # avg_vec = np.zeros(self.config.aggregation_levels)
        # for t in range(1, steps+1):
        #     for point in self.points:
        #         p = point.id
        #         for battery in range(0,self.config.battery_levels+1):
        #             vector, value = self.get_weights_and_agg_value(t,p,battery)

        #             avg_vec += vector

        # return avg_vec/(steps*len(self.points)*self.config.battery_levels)

        try:
            avg_agg_levels = sum(self.agg_weight_vectors.values())/len(self.agg_weight_vectors)
        except:
            return np.zeros(self.config.aggregation_levels)

        self.agg_weight_vectors = dict()

        return avg_agg_levels

    def update_weights(self, t, g, a_g, sampled_v, count_g):

        # WEIGHTING ################################################## #

        # Current value function of attribute at level g
        old_v_ta_g = self.values[t][g][a_g]
        
        # Updating

        # Bias due to smoothing of transient data series (value
        # function change every iteration)
        current_transient_bias = self.transient_bias[t][g][a_g]
        self.transient_bias[t][g][a_g] = self.get_transient_bias(
            current_transient_bias, sampled_v, old_v_ta_g, self.config.stepsize
        )

        # Bias due to aggregation error = v[-,a, g] - v[-, a, 0]
        self.aggregation_bias[t][g][a_g] = old_v_ta_g - sampled_v

        self.variance_g[t][g][a_g] = self.get_variance_g(
            t, g, a_g, sampled_v, old_v_ta_g, self.config.stepsize
        )

        # Updating lambda stepsize using previous stepsizes
        self.lambda_stepsize[t][g][a_g] = self.get_lambda_stepsize(
            self.step_size_func[t][g][a_g],
            self.lambda_stepsize[t][g][a_g]
        )

        # Update the number of times state was accessed
        self.count[t][g][a_g] += count_g
        
        # Generalized harmonic stepsize
        # Notice that a_stepsize is 1 when count is zero
        a_stepsize = self.config.harmonic_stepsize
        stepsize = a_stepsize/(a_stepsize + self.count[t][g][a_g] - 1)
        self.step_size_func[t][g][a_g] = stepsize

    def update_values_smoothed(self, t, duals):
        
        # List of duals associated to tuples (level g, attribute[g])
        # The new value of an aggregate level correspond to the average
        # of these duals
        level_update_list = defaultdict(list)

        for a, v_ta in duals.items():

            pos, battery = a

            # Updating value function at disaggregate level
            self.values[t][0][a] = (
                1 - self.step_size_func[t][0][a]
            ) * v_ta + self.step_size_func[t][0][a] * v_ta

            # Update the number of times disaggregate state was accessed
            self.count[t][0][a] += 1

            # Get point object associated to position
            point = self.points[pos]

            # Append duals to all superior hierachical states
            for g in range(1, self.config.aggregation_levels):

                # Find attribute at level g
                a_g = (point.id_level(g), battery)
                
                # Value is later used to update a_g
                level_update_list[(g, a_g)].append(v_ta)

        for state_g, value_list_g in level_update_list.items():

            g, a_g = state_g

            # Number of times state g was accessed
            count_ta_g = len(value_list_g)

            # Average value function considering all elements sharing
            # the same state at level g
            v_ta_g = sum(value_list_g)/count_ta_g

            self.update_weights(t, g, a_g, v_ta_g, count_ta_g)

            # Updating value function at gth level with smoothing
            self.values[t][g][a_g] = (
                (1 - self.step_size_func[t][g][a_g]) * self.values[t][g][a_g]
                + self.step_size_func[t][g][a_g] * v_ta_g
            )

    ####################################################################
    # True averaging ###################################################
    ####################################################################

    def get_value(self, t, pos, battery, level=0):

        # Point associated to position at disaggregate level
        point = self.points[pos]

        # Attribute considering aggregation level
        attribute = (point.id_level(level), battery)

        # Value function
        value = self.values[t][level][attribute]

        return value

    def averaged_update(self, t, duals):
        """Update values without smoothing

        Arguments:
            step {int} -- Current time step
            duals {dict} -- Dictionary of attribute tuples and duals
        """

        # pprint(duals)

        for (pos, battery), new_vf_0 in duals.items():

            # Get point object associated to position
            point = self.points[pos]

            for g in range(self.config.aggregation_levels):

                # Find attribute at level g
                a_g = (point.id_level(g), battery)

                # Current value function of attribute at level g
                current_vf = self.values[t][g][a_g]

                # Increment every time attribute ta_g is accessed
                self.count[t][g][a_g] += 1

                # Incremental averaging
                count_ta_g = self.count[t][g][a_g]
                increment = (new_vf_0 - current_vf) / count_ta_g

                # Update attribute mean value
                self.values[t][g][a_g] += increment

        # Update weights using new value function estimate
        # self.update_weights(t, g, a_g, new_vf_0, 1)


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
        car.update_recharge(time_min, miles, cost)

        return cost

    def recharge(self, car, time_min):

        # Total cost of recharging
        cost = self.config.calculate_cost_recharge(time_min)

        # Extra kilometers car can travel after recharging
        dist = self.config.calculate_dist_recharge(time_min)

        # Update vehicle status to recharging
        car.update_recharge(
            time_min,
            cost,
            dist,
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

                    car.move(
                        duration,
                        distance,
                        reward,
                        self.points[d],
                        time_increment=self.config.time_increment
                    )

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

    @functools.lru_cache(maxsize=None)
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

    @functools.lru_cache(maxsize=None)
    def get_travel_time(self, distance, unit="min"):
        """Travel time in minutes given distance in miles"""

        travel_time_h = distance / self.config.speed
        travel_time_min = travel_time_h * 60

        if unit == "min":
            return travel_time_min
        else:
            steps = int(round(travel_time_min / self.config.time_increment))
            return steps

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

    def print_fleet_stats(self, filter_status=[]):
        count_status = dict()

        # Start all car statuses with 0
        for s in Car.status_list:
            count_status[s] = 0

        # Count how many car per status
        for c in self.cars:
            if filter_status and c.status not in filter_status:
                continue

            print(c.status_log())
            count_status[c.status] += 1

        
        # pprint(dict(count_status))

    def print_fleet_stats_summary(self):
        count_status = dict()

        # Start all car statuses with 0
        for s in Car.status_list:
            count_status[s] = 0

        # Count how many car per status
        for c in self.cars:
            count_status[c.status] += 1

        pprint(dict(count_status))

    def print_car_traces(self):
        for c in self.cars:
            print(f'# {c}')
            pprint(c.point_list)

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

    def reset(self, use_previous_car_positions=False):

        if not use_previous_car_positions or not self.cars:
            new_origins = random.choices(self.points, k=self.fleet_size)
        else:
            print("Using previous car positions...")
            new_origins = [c.point for c in self.cars]

        Car.count = 0

        self.cars = [
            Car(
                point,
                self.config.battery_levels,
                battery_level_miles_max=self.config.battery_size_distances,
            )
            for point in [
                point
                for point in new_origins
            ]
        ]

        self.available = self.cars

        # self.cars = [
        #     Car(
        #         point,
        #         self.config.battery_levels,
        #         battery_level_miles_max=self.config.battery_size_distances,
        #     )
        #     for point in self.car_origin_points
        # ]

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
