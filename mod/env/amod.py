from mod.env.car import Car
from mod.env.trip import Trip
from mod.env.network import Point, get_point_list, get_neighbor_zones
import itertools as it
from bidict import bidict
import matplotlib.pyplot as plt
from collections import defaultdict, Counter
from operator import itemgetter
import scipy
import numpy as np
import random
from pprint import pprint
import pickle
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

    @staticmethod
    def load(path=None, mode="rb"):
        f = open(path, mode)
        amod = pickle.load(f)
        f.close()
        return amod

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
                cost = self.config.calculate_cost_recharge(
                    self.config.time_increment
                )

                return -cost
            # Rebalance
            else:
                return 0

    def get_point_by_id(self, point_id, level=0):
        return self.dict_points[level][point_id]

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
        # How many times a cell was actually accessed by a vehicle in
        # a certain region, aggregation level, and time
        self.count = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))

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
            # Creating random fleet starting from random points
            self.cars = [
                Car(
                    point,
                    self.battery_levels,
                    battery_level_miles_max=self.battery_size_miles,
                )
                for point in car_positions
            ]

    @lru_cache(maxsize=None)
    def get_neighbors(self, center):
        return get_neighbor_zones(
            center, self.config.pickup_zone_range, self.zones
        )

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

                # print(
                #     f"{step:3} - Level: {g} - Attribute: {attribute_g} = {self.values[step][g][attribute_g]:15.2f}"
                # )
                # if self.values[step][g][attribute_g] > 100:
                #     print(step, g, attribute_g, self.values[step][g][attribute_g])

        # print(f"############# DUALS {len(duals)} #################")

        # pprint(self.values)
        # pprint(self.count)
        # count_reward = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
        # for t, g_a_count in self.count.items():
        #     for g, a_count in g_a_count.items():
        #         # print(f"\n####### Aggregation level: {g}")
        #         for a, count in a_count.items():
        #             l, b = a
        #             point = self.dict_points[g][l]
        #             attribute = (tuple(point.level_ids), b)
        #             value = np.round(self.values[t][g][attribute], decimals=2)
        #             if count >= 0:
        #                 count_reward[t][attribute][g] = {
        #                     'count': count,
        #                     'value': value
        #                 }
        # print("### Count reward ######################################")
        # pprint(
        #     {
        #         t:{ids: {g: c_r for g, c_r in g_tuple.items()}
        #         for ids, g_tuple in ids_g_tuple.items()}
        #         for t, ids_g_tuple in count_reward.items()
        #     }
        # )

        # pprint(count_reward)

    def save(self, path=None):
        f = open(path, "wb")
        pickle.dump(self.values, f)
        f.close()
    
    # @lru_cache(maxsize=None)
    def get_value(self, step, decision, level=0):

        # Target attribute if decision was taken
        d_step, d_pos, d_battery_level = self.preview_decision(step, decision)

        # Point associated to position
        point = self.dict_points[0][d_pos]

        # Attribute considering aggregation level
        attribute = (point.id_level(level), d_battery_level)

        # Value function
        value = self.values[d_step][level][attribute]
        return value

    def get_value2(self, attribute):
        # Id position at aggregation level 0
        pos_level_0 = attribute[0]

        # Point associated to position
        point = self.dict_points[0][pos_level_0]

        # Value associated to attribute
        value = 0

        # Weights
        w = [0.4, 0.3, 0.2, 0.5, 0.5]

        # Loop aggregations
        for g in range(self.config.aggregation_levels):

            # Attribute at level g
            attribute_g = (point.id_level(g), attribute[1])

            if attribute_g not in self.values[g]:
                continue

            # List of values associated to attribute of level g
            values_level = self.values[g][attribute_g][0]

            value += w[g] * values_level
        return value

    ################################################################
    # Network ######################################################
    ################################################################
    def get_travel_time(self, distance, unit="min"):
        """Travel time in minutes given distance in miles"""
        travel_time_h = distance / self.config.speed_mph
        travel_time_min = travel_time_h * 60

        if unit == "min":
            return travel_time_min
        else:
            steps = int(round(travel_time_min / self.config.time_increment))
            return steps

    def get_travel_time_od(self, o, d):
        """Travel time in minutes given Euclidean distance in miles
        between origin o and destination d"""
        distance = self.get_distance(o, d)
        return self.get_travel_time(distance)

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

    def rebalance(self, car, d):

        # Distance car has to travel to rebalance
        distance = self.get_distance(car.point, d)

        # Next arrival
        duration_min = int(round(self.get_travel_time(distance)))

        # No reward for rebalancing
        reward = 0

        return duration_min, distance, reward

    def get_duration(self, pos, o, d, unit="min"):
        """Get trip duration

        Arguments:
            pos {int} -- Position car
            o {int} -- Pickup point
            d {int} -- Delivery point

        Keyword Arguments:
            unit {str} -- Trip duration unit [min, step, mile]
            (default: {'min'})

        Returns:
            [int] -- Duration in time steps
        """

        distance = self.get_travel_distance(pos, o, d)
        duration = int(round(self.get_travel_time(distance)))

        if unit == "min":
            return duration

        elif unit == "step":
            steps = int(round(duration / self.config.time_increment))
            return steps

    def get_travel_distance(self, pos, o, d):
        """Total travel distance between three point ids.

        Arguments:
            pos {int} -- Position car
            o {int} -- Pickup point
            d {int} -- Delivery point

        Returns:
            [float] -- Distance in miles
        """

        # Distance car has to travel to service trip
        distance_trip = self.get_distance(self.points[pos], self.points[o])

        # Distance to pickup passanger
        distance_pickup = self.get_distance(self.points[o], self.points[d])

        # Total distance
        total_distance = distance_pickup + distance_trip

        return total_distance

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

        # if update:
        #     # Update car data
        #     car.update_trip(
        #         duration_min,
        #         total_distance,
        #         reward,
        #         trip
        #     )

        #     # Associate car to trip
        #     trip.picked_by = car
        # print(
        #     f'Trip total distance ({trip.o}->{trip.d}) = {total_distance:.2f}'
        #     f' miles ({distance_pickup:.2f}+{distance_trip:.2f})'
        #     f' (~{duration_min}min) - Fares: ${reward:.2f}'
        # )

        return duration_min, total_distance, reward

    def cars_idle(self):
        return [c for c in self.cars if not c.busy]

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

    def get_current_action(self, cars):

        actions = np.zeros(self.action_space.size, dtype=int)

        for c in cars:

            if c.status == Car.IDLE:
                decision = (Amod.TRIP_STAY_DECISION, c.point.id, c.point.id)

            elif c.status == Car.ASSIGN:
                decision = (Amod.TRIP_STAY_DECISION, c.trip.o.id, c.trip.d.id)

            elif c.status == Car.RECHARGING or c.status == Car.REBALANCE:
                decision = (
                    Amod.RECHARGE_REBALANCE_DECISION,
                    c.previous.id,
                    c.point.id,
                )

            action_id = self.action_space.dict[(c.attribute, decision)]
            actions[action_id] += 1

        return actions

    @staticmethod
    def get_decision(car, level):
        """Return the decision tuple derived from car.


        Arguments:
            car {Car} -- Vehicle after decision is made

        Returns:
            tuple -- decision code, from zone, to zone

        E.g.:
        decision (0, 1, 2) - Car is servicing customer (flag=0)
        going from zone 1 to zone 2
        """

        if car.status == Car.IDLE:
            decision = (
                Amod.TRIP_STAY_DECISION,
                car.point.id_level(level),
                car.point.id_level(level),
            )
        elif car.status == Car.ASSIGN:
            decision = (
                Amod.TRIP_STAY_DECISION,
                car.trip.o.id_level(level),
                car.trip.d.id_level(level),
            )
        elif car.status == Car.RECHARGING or car.status == Car.REBALANCE:
            decision = (
                Amod.RECHARGE_REBALANCE_DECISION,
                car.previous.id_level(level),
                car.point.id_level(level),
            )
        return decision

    def print_environment(self):
        """Print environment zones, points, and cars"""
        print("\nZones:")
        pprint(self.zones)

        print("\nLocations:")
        pprint(self.points)

        print("\nFleet:")
        pprint(self.cars)

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

    def print_current_stats(self):
        count_status = defaultdict(int)
        for c in self.cars:
            print(c.status_log())
            count_status[c.status] += 1

        pprint(dict(count_status))

    def reset(self):
        # Rt = resource state vector at time t
        # resource_state_vector = np.zeros(len(self.car_attributes))

        # Dt = trip state vector at time t
        # trip_state_vector = np.zeros(len(self.trip_attributes))

        # Back to initial state
        # self.state = (resource_state_vector, trip_state_vector)

        # Return cars to initial state
        # for c in self.cars:
        #     c.reset(self.battery_size)

        # Creating random fleet starting from random points
        self.cars = [
            Car(
                point,
                self.config.battery_levels,
                battery_level_miles_max=self.config.battery_size_miles,
            )
            for point in self.car_origin_points
        ]

    def update_fleet_status(self, step):
        # Update fleet status
        for car in self.cars:
            car.update(step, time_increment=self.config.time_increment)

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

    def preview_move(self, car_pos_id, o_id, d_id):

        distance = self.get_distance(self.points[o_id], self.points[d_id])

        # Car is not in the same zone, of pickup point. Therefore, it
        # has to drive there first
        if car_pos_id != o_id:
            distance += self.get_distance(
                self.points[car_pos_id], self.points[o_id]
            )

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
