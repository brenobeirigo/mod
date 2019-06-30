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
import requests
import functools
from mod.env.amod.AmodNetwork import AmodNetwork
import mod.env.decision_utils as du


class AmodNetworkHired(AmodNetwork):
    def __init__(self, config, car_positions=[]):
        """Street network Amod environment with third-party hired fleet
        
        Parameters
        ----------
        Amod : Environment parent class
            Methods to manipulate environment
        config : Config
            Simulation settings
        car_positions : list, optional
            Cars can start from predefined positions, by default []
        """

        super().__init__(config, car_positions=car_positions)

        # Third-party fleet can be hired to assist main fleet
        self.hired_cars = []
        self.available_hired = []
        self.available_hired_ids = np.zeros(len(self.points_level[0]))
        self.expired_contract_cars = []

    @functools.lru_cache(maxsize=None)
    def cost_func(self, car_type, action, o, d):

        profit_margin = 1
        if car_type == Car.TYPE_HIRED:
            profit_margin = self.config.profit_margin

        if action == du.STAY_DECISION:
            # Stay
            return 0

        elif action == du.TRIP_DECISION:
            # Pick up
            distance_trip = self.get_distance(self.points[o], self.points[d])

            reward = self.config.calculate_fare(distance_trip)

            return profit_margin * reward

        elif action == du.RECHARGE_DECISION:
            # Recharge
            cost = self.config.cost_recharge_single_increment
            return -cost

        else:
            # Rebalance
            return 0

    @functools.lru_cache(maxsize=None)
    def cost_func2(self, car_type, action, pos, o, d):
        """Return decision cost.

        Parameters
        ----------
        car_type : self owned or third party
            [description]
        action : Decision type
            STAY, TRIP, RECHARGE, REBALANCE
        pos : int
            Id car current position
        o : int
            Id trip origin
        d : int
            Id trip destination

        Returns
        -------
        float
            Decision cost
        """

        # Platform's profit margin is lower when using hired cars
        profit_margin = 1
        if car_type == Car.TYPE_HIRED:
            profit_margin = self.config.profit_margin

        if action == du.STAY_DECISION:
            # Stay
            return 0

        elif action == du.TRIP_DECISION:

            # From car's position to trip's origin
            distance_pickup = self.get_distance(
                self.points[pos], self.points[o]
            )

            # From trip's origin to trip's destination
            dist_rebalance = self.get_distance(self.points[o], self.points[d])

            # Travel cost
            cost = self.config.get_travel_cost(
                distance_pickup + dist_rebalance
            )

            # Base fare + distance cost
            revenue = self.config.calculate_fare(dist_rebalance)

            # Profit to service trip
            return profit_margin * (revenue - cost)

        elif action == du.RECHARGE_DECISION:

            # Recharge
            cost = self.config.cost_recharge_single_increment
            return -cost

        elif action == du.REBALANCE_DECISION:

            # From trip's origin to trip's destination
            dist_rebalance = self.get_distance(self.points[o], self.points[d])

            # Travel cost
            cost = self.config.get_travel_cost(dist_rebalance)
            return -cost

    def realize_decision(self, t, decisions, a_trips, dict_a_cars):
        total_reward = 0
        serviced = list()

        # Count number of times each decision was taken
        decision_dict_count = {
            du.STAY_DECISION: 0,
            du.REBALANCE_DECISION: 0,
            du.TRIP_DECISION: 0,
            du.RECHARGE_DECISION: 0,
        }

        matched_cars = set()

        for decision in decisions:

            action, point, level, o, d, car_type, contract_duration, times = (
                decision
            )

            decision_dict_count[action] += times

            cars_with_attribute = dict_a_cars[car_type][
                (point, level, contract_duration)
            ]

            n = 0

            # Main fleet cars are in the beggining of the list

            # Only 'times' cars will execute decision determined in
            # action 'a'
            while cars_with_attribute and n < times:
                n += 1
                car = cars_with_attribute.pop(0)

                # Check if car was already used. If so, try next car
                if car not in matched_cars:
                    matched_cars.add(car)

                else:
                    # Some decision was already applied to this car
                    continue

                # If car belongs to main fleet, profit margin is 100%
                profit_margin = 1

                # Start contract, if not started
                if car_type == Car.TYPE_HIRED:

                    profit_margin = self.config.profit_margin

                    # First time hired vehicle service user
                    if not car.started_contract:
                        # print(f"*** HIRING {action}!")
                        car.started_contract = True
                    # else:
                    #     print(f"*** ALREADY HIRED & {action}!")

                if action == du.RECHARGE_DECISION:
                    # Recharging ##################################### #
                    cost_recharging = self.recharge(
                        car, self.config.recharge_time_single_level
                    )

                    # Subtract cost of recharging
                    total_reward -= cost_recharging

                elif action == du.REBALANCE_DECISION:
                    # Rebalancing #################################### #
                    duration, distance, reward = self.rebalance(
                        car, self.points[d]
                    )

                    car.move(
                        duration,
                        distance,
                        reward,
                        self.points[d],
                        time_increment=self.config.time_increment,
                    )

                elif action == du.STAY_DECISION:
                    # Car settings are updated all together when time
                    # step finishes
                    pass

                elif action == du.TRIP_DECISION:
                    # Servicing ###################################### #

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
                        time_increment=self.config.time_increment,
                    )

                    serviced.append(trip)
                    total_reward += profit_margin * reward

            # Remove cars already used to fulfill decisions
            # cars_with_attribute = cars_with_attribute[times:]

        self.decision_dict = decision_dict_count

        # Remaining trips associated with trip attributes correspond to
        # users who have been denied service
        denied = list(it.chain.from_iterable(a_trips.values()))

        return (total_reward, serviced, denied)

    def update_fleet_status(self, time_step):

        # Idle company-owned cars
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

        # Idle hired cars
        available_hired = []

        # List of cars whose contracts have expired
        expired_contract = []

        for car in self.hired_cars:
            # Check if vehicles finished their tasks
            # Where are the cars?
            # What are they doing at the current step?
            # t ----- t+1 ----- t+2 ----- t+3 ----- t+4 ------- t+5
            # --trips----trips------trips-----trips------trips-----
            car.update(time_step, time_increment=self.config.time_increment)

            # Car has started the contract
            if car.started_contract:

                # Contract duration has expired
                if car.contract_duration == 0:
                    expired_contract.append(car)
                    self.available_hired_ids[car.point.id] += 1

                # Discard busy vehicles
                elif not car.busy:
                    available_hired.append(car)

        self.available = available
        self.available_hired = available_hired

        # Remove expired contract cars
        for car in expired_contract:
            self.hired_cars.remove(car)

            # Save expired contract cars
            self.expired_contract_cars.extend(expired_contract)

    def preview_decision(self, time_step, decision):
        """Apply decision to attributes

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

        action, point, battery, o, d, type_car, contract_duration = decision
        battery_post = battery
        if action == du.RECHARGE_DECISION:

            # Recharging ###############################################
            battery_post = min(self.battery_levels, battery + 1)
            time_step += self.config.recharge_time_single_level
            contract_duration -= int(
                self.config.recharge_time_single_level
                / self.config.contract_duration_level
            )

        elif action == du.REBALANCE_DECISION:
            # Rebalancing ##############################################

            duration, battery_drop = self.preview_move(point, o, d)
            time_step += max(1, duration)
            battery_post = max(0, battery - battery_drop)
            point = d
            contract_duration -= int(
                duration / self.config.contract_duration_level
            )

        elif action == du.STAY_DECISION:
            # Staying ##################################################
            time_step += 1
            contract_duration -= int(
                self.config.time_increment
                / self.config.contract_duration_level
            )

        elif action == du.TRIP_DECISION:
            # Servicing ################################################
            duration, battery_drop = self.preview_move(point, o, d)
            time_step += max(1, duration)
            battery_post = max(0, battery - battery_drop)
            point = d
            contract_duration -= int(
                duration / self.config.contract_duration_level
            )

        return time_step, point, battery_post, contract_duration

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

        for c in [c for c in self.hired_cars if c.started_contract]:
            total_battery_level += c.battery_level_miles
            status_count[c.status] += 1

        return status_count, total_battery_level

    def reset(self):

        super().reset()
        self.hired_cars = []
        self.available_hired = []

    def get_fleet_stats(self):

        stats = super().get_fleet_stats()

        count_status_sec = defaultdict(int)

        # Start all car statuses with 0
        for s in Car.status_list:
            count_status_sec[s] = 0

        # Count how many car per status
        for c in self.hired_cars:
            count_status_sec[c.status] += 1

        stats["Secondary fleet"] = dict(count_status_sec)
        stats["Available"] = {"Hired": len(self.hired_cars)}

        return stats

    def post_cost(self, t, decision, level=None):

        # Target attribute if decision was taken
        (
            post_t,
            post_pos,
            post_battery,
            post_contract_duration,
        ) = self.preview_decision(t, decision)

        # Get the value estimation considering a single level
        if level:
            estimate = self.adp.get_value(
                post_t, post_pos, post_battery, level=level
            )

        else:
            # Get the post decision state estimate value based on
            # hierarchical aggregation
            estimate = self.adp.get_weighted_value(
                post_t,
                post_pos,
                post_battery,
                contract_duration=post_contract_duration,
            )

        return estimate
