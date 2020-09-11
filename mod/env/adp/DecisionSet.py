from copy import deepcopy

from .decisions import *
from ..fleet.HiredCar import HiredCar


class DecisionSet:

    def __init__(self, env, trips):
        self.env = env
        self.trips = trips
        self.all_decisions = set()
        self.return_decisions = set()
        self.classed_decisions = defaultdict(list)
        self.rebalance_decisions = set()
        self.reachable_trips_i = set()
        self.attribute_trips_dict = defaultdict(set)
        self.attribute_trips_sq_dict = defaultdict(int)
        self.from_location = defaultdict(int)
        self.states = defaultdict(int)
        self.gen_decisions_for_available_vehicles()
        # self.gen_decisions_for_busy_vehicles()
        self.gen_decisions_for_rebalancing_vehicles()

    def gen_decisions_for_available_vehicles(self):

        for car in self.env.get_all_available_vehicles():

            if not self.state_already_visited(car):
                self.add_stay_decision_with_conditions(car)
                self.add_return_decision_for_hired_car(car)
                self.add_rebalance_decisions_for_car(car)
                self.add_recharge_decision_for_car(car)
                self.add_all_trip_decisions_car(car)

                self.account_for_visited_state_from_car(car)

    def gen_decisions_for_busy_vehicles(self):
        d = deepcopy(self.all_decisions)
        for car in self.env.get_all_busy_vehicles():
            self.add_all_trip_decisions_car(car)
            self.account_for_visited_state_from_car(car)
        # print(self.all_decisions - d)

    def state_already_visited(self, car):
        return self.states[car.attribute] > 0

    def account_for_visited_state_from_car(self, car):
        self.from_location[car.point.id] += 1
        self.states[car.attribute] += 1

    def gen_decisions_for_rebalancing_vehicles(self):

        for car in self.env.get_all_rebalancing_vehicles():
            self.add_stay_decision_for_reb_car(car)
            self.add_trip_decisions_for_reb_car(car)

    def add_stay_decision_with_conditions(self, car):
        # If idle_annealing is active (i.e., it is a number),
        # cars can decide to stay only if they haven't been parked
        # for idle_annealing steps.
        # For example, if idle_annealing = 1, and a car idle_step_count
        # is also 1, it can't park anymore.
        # The idle_annealing grows with the iterations such that in the
        # end of the experiment, cars are allowed to stay parked for
        # longer periods.
        # Notice that a car idle_step_count is zeroed after servicing
        # customer or rebalancing.
        if self.env.config.idle_annealing is not None:

            # Can stay only when idle annealing is large.
            if car.idle_step_count < self.env.config.idle_annealing:
                self.add_stay_decision_for_car(car)

        # Cars can only stay at parking lots
        elif self.env.config.cars_start_from_parking_lots and (
                car.point.id in self.env.level_parking_ids
                or car.point.id in self.env.unrestricted_parking_node_ids
        ):
            self.add_stay_decision_for_car(car)

        # Cars can stay anywhere
        else:
            self.add_stay_decision_for_car(car)

    def add_trip_decisions_for_reb_car(self, car):
        # Try matching trips departing from the closest middle point
        for i, trip in enumerate(self.trips):

            # Car cannot service trip because it cannot go back
            # to origin in time
            if isinstance(car, HiredCar):
                if not self.env.car_rebalancing_can_return_to_station_after_servicing_trip(car, trip):
                    continue

            max_pk_time = (
                    trip.max_delay - self.env.config.time_increment - trip.backlog_delay
            )

            # Time to reach trip origin
            pk_time = self.env.get_travel_time_od(
                car.middle_point, trip.o, unit="min"
            )

            # Can the car reach the trip origin?
            if pk_time + car.elapsed <= max_pk_time + trip.tolerance:
                # Setup decisions
                self.add_single_trip_decision_for_car(car, trip)
                self.reachable_trips_i.add(i)

    def add_stay_decision_for_reb_car(self, car):
        d_stay = stay_decision_reb(car)
        self.all_decisions.add(d_stay)

    def add_rebalance_decisions_for_car(self, car):
        # Rebalancing ################################################ #
        # myopic = NO
        # reactive = NO (Rebalance decisions only in 2nd round)
        # random, train, and test = YES
        if self.env.config.consider_rebalance:

            neighbors = self.env.neighbors[car.point.id]
            if self.env.config.activate_thompson:
                d_rebalance = rebalance_decisions_thompson(car, neighbors, self.env)
            else:
                if isinstance(car, HiredCar):
                    # Car can always rebalance to its home station.
                    # Makes sense when parking costs are cheaper at
                    # home station.
                    neighbors = neighbors | {car.depot.id}

                d_rebalance = rebalance_decisions(car, neighbors, self.env)

            if not d_rebalance:
                # Remove from tabu if not empty.
                # Avoid cars are corned indefinitely
                if car.tabu:
                    car.tabu.popleft()

            # TODO this is here because of a lack or rebalancing options
            # thompson selected is small 0.2
            if len(d_rebalance) == 1:
                self.add_stay_decision_for_car(car)

            # Vehicles can stay idle for a maximum number of steps.
            # If they surpass this number, they can rebalance to farther
            # areas.
            if self.env.config.max_idle_step_count:

                # Car can rebalance to farther locations besides the
                # closest after staying still for idle_step_count steps
                if car.idle_step_count >= self.env.config.max_idle_step_count:
                    farther = self.env.get_zone_neighbors(
                        car.point.id, explore=True
                    )

                    d_rebalance.update(rebalance_decisions(car, farther, self.env))

            self.rebalance_decisions.update(d_rebalance)
            self.all_decisions.update(d_rebalance)

    def add_return_decision_for_hired_car(self, car):
        if isinstance(car, HiredCar):

            # If car has moved from depot
            if car.point.id != car.depot.id:

                # Car must return when contract is about to end
                return_trip_duration = self.env.get_travel_time_od(
                    car.point, car.depot, unit="min"
                )

                if car.contract_duration <= return_trip_duration:
                    d_return = return_decision(car)
                    self.rebalance_decisions.add(d_return)
                    self.return_decisions.add(d_return)
                    self.all_decisions.add(d_return)

    def add_stay_decision_for_car(self, car):
        d_stay = stay_decision(car)
        self.all_decisions.add(d_stay)

    def add_all_trip_decisions_car(self, car):
        for trip_id, trip in enumerate(self.trips):

            if isinstance(car, HiredCar):
                if not self.env.car_can_return_to_station_after_servicing_trip(car, trip):
                    continue

            if self.env.car_can_pickup_trip(car, trip):
                self.add_single_trip_decision_for_car(car, trip)
                self.reachable_trips_i.add(trip_id)

    def add_single_trip_decision_for_car(self, car, trip):
        d = trip_decision(car, trip)
        self.all_decisions.add(d)
        self.attribute_trips_dict[(trip.o.id, trip.d.id)].add(trip)

    def add_recharge_decision_for_car(self, car):
        if self.env.car_battery_level_low(car):
            self.all_decisions.add(recharge_decision(car))

    def get_rejected_upfront(self):
        return [
            trip for i, trip in enumerate(self.trips) if i not in self.reachable_trips_i
        ]
