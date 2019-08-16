import itertools
from mod.env.car import Car, HiredCar
from mod.env.trip import Trip
from mod.env.network import Point
import mod.env.adp.AdpHired as adp
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
import mod.env.decisions as du
from copy import deepcopy

np.set_printoptions(precision=2)
# Reproducibility of the experiments
random.seed(1)


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
    def cost_func(self, decision):
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
        PROFIT_MARGIN = 1

        # When moving a hired car, the platform is charged 2X more
        # since this car must return to its origin point
        RETURN_FACTOR = 1

        CONGESTION_PRICE = 0

        if (
            decision[du.CAR_TYPE] == Car.TYPE_HIRED
            or decision[du.CAR_TYPE] == Car.TYPE_TO_HIRE
        ):
            PROFIT_MARGIN = self.config.profit_margin

        if decision[du.CAR_TYPE] == Car.TYPE_TO_HIRE:
            CONGESTION_PRICE = self.config.congestion_price

        if decision[du.ACTION] == du.STAY_DECISION:
            # Stay
            return 0

        elif decision[du.ACTION] == du.TRIP_DECISION:

            # From car's position to trip's origin
            distance_pickup = self.get_distance(
                self.points[decision[du.POSITION]],
                self.points[decision[du.ORIGIN]],
            )

            # From trip's origin to trip's destination
            dist_trip = self.get_distance(
                self.points[decision[du.ORIGIN]],
                self.points[decision[du.DESTINATION]],
            )

            # Travel cost
            cost = self.config.get_travel_cost(distance_pickup + dist_trip)

            # Base fare + distance cost
            revenue = self.config.calculate_fare(
                dist_trip, sq_class=decision[du.SQ_CLASS]
            )

            contribution = PROFIT_MARGIN * (revenue - cost) - CONGESTION_PRICE

            # Profit to service trip
            return contribution

        elif decision[du.ACTION] == du.RECHARGE_DECISION:

            # Recharge
            cost = self.config.cost_recharge_single_increment
            return -cost

        elif decision[du.ACTION] == du.REBALANCE_DECISION:

            # From trip's origin to trip'scar_type
            dist_trip = self.get_distance(
                self.points[decision[du.ORIGIN]],
                self.points[decision[du.DESTINATION]],
            )

            # Travel cost
            cost = self.config.get_travel_cost(dist_trip)

            reb_cost = -RETURN_FACTOR * cost - CONGESTION_PRICE
            # print(action, pos, decision[du.ORIGIN], d, car_type, sq_class, reb_cost)
            return reb_cost

    def can_move(self, pos, waypoint, target, start, remaining_hiring_slots):
        pos, waypoint, target, start = (
            Point.point_dict[pos],
            Point.point_dict[waypoint],
            Point.point_dict[target],
            Point.point_dict[start],
        )
        dist_to_origin = self.get_distance(pos, waypoint)
        dist_od = self.get_distance(waypoint, target)
        dist_d_start = self.get_distance(target, start)

        total_dist = dist_to_origin + dist_od + dist_d_start

        # Next arrival
        duration_movement = self.get_travel_time(total_dist, unit="min")

        remaining_hiring_time = (
            remaining_hiring_slots * self.config.contract_duration_level
        )

        return remaining_hiring_time > duration_movement

    def realize_decision(self, t, decisions, a_trips_dict, a_cars_dict):
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

            (
                action,
                point,
                battery,
                contract_duration,
                car_type,
                car_origin,
                o,
                d,
                sq_class,
                times,
            ) = decision

            # Track how many times a decision was taken
            decision_dict_count[action] += times

            cars_with_attribute = a_cars_dict[
                (point, battery, contract_duration, car_type, car_origin)
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

                # Ignores last element (n. times decision was applied)
                contribution_car = self.cost_func(decision[:-1])

                # print(decision, contribution_car)

                # Start contract, if not started
                if car_type == Car.TYPE_TO_HIRE:

                    # Hired car will be used by the system
                    if action != du.STAY_DECISION:
                        car.started_contract = True
                        car.type = Car.TYPE_HIRED
                    else:
                        contribution_car = 0

                if action == du.RECHARGE_DECISION:
                    # Recharging ##################################### #
                    cost_recharging = self.recharge(
                        car, self.config.recharge_time_single_level
                    )

                elif action == du.REBALANCE_DECISION:
                    # Rebalancing #################################### #
                    total_duration, total_duration_steps, total_distance, reward = self.rebalance(
                        car, self.points[d]
                    )

                    car.move(
                        total_duration,
                        total_duration_steps,
                        total_distance,
                        contribution_car,
                        self.points[d],
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
                            for trip in a_trips_dict[(o, d)]
                        ]
                    )

                    # Get a trip to apply decision
                    trip = a_trips_dict[(o, d)].pop(iclosest_pk)

                    self.pickup(trip, car)

                    serviced.append(trip)

                # Subtract cost of recharging
                total_reward += contribution_car

            # Remove cars already used to fulfill decisions
            # cars_with_attribute = cars_with_attribute[times:]

        self.decision_dict = decision_dict_count

        # Remaining trips associated with trip attributes correspond to
        # users who have been denied service
        denied = list(it.chain.from_iterable(a_trips_dict.values()))

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

    def discard_excess_hired(self):

        # Car was not used in last iteration
        active_fleet = []

        for car in self.hired_cars:
            if car.started_contract:
                active_fleet.append(car)

        self.hired_cars = active_fleet

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
            tuple: 
                time_step,
                point,
                battery_post,
                contract_duration,
                type_post,
                car_origin,
        )
        """

        action, point, battery, contract_duration, car_type, car_origin, o, d, _ = (
            decision
        )

        battery_post = battery
        type_post = car_type

        if action == du.RECHARGE_DECISION:

            # Recharging ###############################################
            battery_post = min(self.battery_levels, battery + 1)
            time_step += self.config.recharge_time_single_level
            duration = self.config.recharge_time_single_level

        elif action == du.REBALANCE_DECISION:
            # Rebalancing ##############################################

            duration, battery_drop = self.preview_move(point, o, d)
            time_step += max(1, duration)
            battery_post = max(0, battery - battery_drop)
            point = d

            # Hiring is performed when car rebalance
            if car_type == Car.TYPE_TO_HIRE:
                type_post = Car.TYPE_HIRED

        elif action == du.STAY_DECISION:
            # Staying ##################################################
            time_step += 1
            duration = self.config.time_increment

        elif action == du.TRIP_DECISION:
            # Servicing ################################################
            duration, battery_drop = self.preview_move(point, o, d)
            time_step += max(1, duration)
            battery_post = max(0, battery - battery_drop)
            point = d

            # Hiring is performed when car service user
            if car_type == Car.TYPE_TO_HIRE:
                type_post = Car.TYPE_HIRED

        # Contract duration is altered only for hired cars
        if car_type == Car.TYPE_HIRED or car_type == Car.TYPE_TO_HIRE:
            contract_duration -= int(
                duration / self.config.contract_duration_level
            )

        return (
            time_step,
            point,
            battery_post,
            contract_duration,
            type_post,
            car_origin,
        )

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
            # total_battery_level += c.battery_level_miles
            status_count[c.status] += 1

        for c in self.hired_cars:
            if c.started_contract:
                # total_battery_level += c.battery_level_miles
                status_count[c.status] += 1

        return status_count, total_battery_level

    def reset(self):

        super().reset()
        self.hired_cars = []
        self.available_hired = []
        # self.post_cost.cache_clear()
        self.adp.weighted_values.clear()

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

    def update_vf(self, duals, time_step):
        """Update value function using duals from a time step.

        Parameters
        ----------
        duals : dict
            Attribute tuple and dual value
        time_step : int
            Time step from which duals were generated
        """

        if duals:

            if self.config.update_values_averaged():
                self.adp.averaged_update(time_step, duals)

            elif self.config.update_values_smoothed():
                self.adp.update_values_smoothed(time_step, duals)

    # @functools.lru_cache(maxsize=None)
    def post_cost(self, t, decision):

        # Target attribute if decision was taken
        post_state = self.preview_decision(t, decision)

        if post_state[adp.adp.TIME] >= self.config.time_steps:
            return 0

        # Get the post decision state estimate value based on
        # hierarchical aggregation
        estimate = self.adp.get_weighted_value(post_state)

        # Penalize long rebalancing decisions
        if (
            decision[0] == du.REBALANCE_DECISION
            and self.config.penalize_rebalance
        ):

            avg_busy_stay = 0

            for busy_reb_t in range(t, post_state[adp.adp.TIME]):

                stay = (du.STAY_DECISION,) + decision[1:]

                # Target attribute if decision was taken
                stay_post_state = self.preview_decision(busy_reb_t, stay)

                estimate_stay = self.adp.get_weighted_value(stay_post_state)

                avg_busy_stay += estimate_stay

            if avg_busy_stay > 0:

                # avg_stay = avg_busy_stay / (post_t - t + 1)
                # avg_stay = avg_busy_stay
                # print(
                #     f"t:{t} - post_t={post_t} - Stay: {np.arange(t + 1, post_t+1)} = {avg_busy_stay} (avg={avg_stay:6.2f}, previous={estimate:6.2f}, new={estimate-avg_stay:6.2f}"
                # )
                # Discount the average contribution that would have
                # been gained if the car stayed still instead of
                # rebalancing
                estimate = max(0, estimate - avg_busy_stay)

        return estimate

    def get_car_status_list(self, filter_status=[]):

        count_status = dict()

        car_status_list = list()

        # Start all car statuses with 0
        for s in Car.status_list:
            count_status[s] = 0

        # Count how many car per status
        for c in itertools.chain(self.cars, self.hired_cars):
            if filter_status and c.status not in filter_status:
                continue

            car_status_list.append(c.status_log())

            count_status[c.status] += 1

        return car_status_list
