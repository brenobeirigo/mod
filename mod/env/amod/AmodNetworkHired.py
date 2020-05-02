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
import math
import pandas as pd
from scipy.stats import gamma, norm, truncnorm

np.set_printoptions(precision=2)
# Reproducibility of the experiments
random.seed(1)


class BetaSampler:
    def __init__(self, seed):
        self.rnd = np.random.RandomState(seed)

    def next_sample(self, a, b):
        # a, b > 0
        alpha = a + b
        beta = 0.0
        u1 = 0.0
        u2 = 0.0
        w = 0.0
        v = 0.0
        if min(a, b) <= 1.0:
            beta = max(1 / a, 1 / b)
        else:
            beta = np.sqrt((alpha - 2.0) / (2 * a * b - alpha))
        gamma = a + 1 / beta

        while True:
            u1 = self.rnd.random_sample()
            u2 = self.rnd.random_sample()
            v = beta * np.log(u1 / (1 - u1))
            w = a * np.exp(v)
            tmp = np.log(alpha / (b + w))
            if alpha * tmp + (gamma * v) - 1.3862944 >= np.log(u1 * u1 * u2):
                break
        x = w / (b + w)
        return x


class AmodNetworkHired(AmodNetwork):
    def __init__(self, config, car_positions=[], online=False):
        """Street network Amod environment with third-party hired fleet

        Parameters
        ----------
        Amod : Environment parent class
            Methods to manipulate environment
        config : Config
            Simulation settings
        car_positions : list, optional
            Cars can start from predefined positions, by default []
        online: boolean, optional
            If True, calculate costs on the fly, otherwise load costs
        """

        super().__init__(config, car_positions=car_positions)

        # Third-party fleet can be hired to assist main fleet. These are
        # hired cars whose contracts are still active.
        self.hired_cars = []

        # List of all cars hired (active and inactive contracts)
        self.overall_hired = []

        # With a depot list, we can determine a number of FAVs per node.
        # For that to work, we need a fixed order.
        self.fav_depots = self.get_fav_depots()

        # Depot set is used in flood avoidance constraint. Depot nodes
        # have no max. number o vehicles, since they are considered to
        # be parking lots.
        self.depots = set(self.fav_depots)
        # List of fav list
        self.step_favs = self.get_hired_step()
        self.available_hired = []
        self.rebalancing_hired = []
        self.available_hired_ids = np.zeros(len(self.point_ids_level[0]))
        self.expired_contract_cars = []

        # Used to execute the method in sequence for vehicles of
        # different types. For example, first execute only for PAVs, and
        # then only for FAVs.
        self.toggled_fleet = dict()
        self.toggled_fleet[Car.TYPE_FLEET] = None
        self.toggled_fleet[Car.TYPE_HIRED] = None

        # UCB
        # self.t_pos_count = defaultdict(int)

        # Thompson
        self.beta_ab = defaultdict(lambda: dict(a=1, b=1))
        self.beta_sampler = BetaSampler(1)
        self.cur_step = 0

        # Is data calculated on-the-fly?
        if online:
            self.revenue = self.online_revenue
            self.cost = self.online_costs
            self.penalty = self.online_penalty
        # Load data from dictionaries
        else:
            self.load_od_data()
            self.revenue = self.loaded_revenue
            self.cost = self.loaded_costs
            self.penalty = self.loaded_penalty
            self.od_dists_step = self.loaded_od_dists

    def online_penalty(self, car_o, trip_o, sq):

        # If tolerance is zero, there is no delay penalty
        tolerance = self.config.trip_tolerance_delay[sq]
        if tolerance == 0:
            return 0

        # Include time increment because it covers the worst case
        # scenario (user waiting since the beginning of the round)
        max_pk_delay = (
            self.config.trip_max_pickup_delay[sq] - self.config.time_increment
        )

        # Pickup travel time
        distance = nw.get_distance(car_o, trip_o)
        pk_time = self.get_travel_time(distance, unit="min")

        # If pickup travel time surpasses user max. waiting
        # 0 <= travel_time <= max_time + tolerance
        if pk_time > max_pk_delay + tolerance:
            return None

        # Delay considering 1st tier service level
        # 0 <= delay <= tolerance
        delay = max(0, pk_time - max_pk_delay)

        # Base fare is the upper bound for the penalty
        base_fare = self.config.trip_base_fare[sq]

        # Penalty is a function of the delay tolerance
        # consumed
        penalty = (base_fare / tolerance) * delay
        # print(
        #     "sq={}, distance={:6.2f}, travel_time={:6.2f}, max_time={:6.2f}, tolerance={:6.2f}, delay={:6.2f}, max_time + tolerance={:6.2f}, base_fare={:6.2f}, penalty={:6.2f}".format(
        #         sq,
        #         distance,
        #         travel_time,
        #         max_time,
        #         tolerance,
        #         delay,
        #         max_time + tolerance,
        #         base_fare,
        #         penalty,
        #     )
        # )
        return penalty

    @property
    def stats_neighbors(self):
        """Return (mean, max, min) #neighbors across all nodes"""
        n_count = [len(n) for n in self.neighbors.values()]
        return np.mean(n_count), np.max(n_count), np.min(n_count)

    def load_od_data(self):

        print("Loading od data...")
        # Use the AmoD info to create travel_cost_array
        try:

            self.od_costs_dict = np.load(
                self.config.get_path_od_costs(), allow_pickle=True
            )
            print(f' - Loaded costs from "{self.config.get_path_od_costs()}"')

            self.od_fares_dict = np.load(
                self.config.get_path_od_fares(), allow_pickle=True
            ).item()
            print(f' - Loaded fares from "{self.config.get_path_od_fares()}"')

            self.od_penalties_dict = np.load(
                self.config.get_path_od_penalties(), allow_pickle=True
            ).item()
            print(
                f' - Loaded penalties from "{self.config.get_path_od_penalties()}"'
            )

            self.od_distance_steps = np.load(
                self.config.get_path_od_distance_steps(), allow_pickle=True
            ).item()
            print(
                f' - Loaded od distance steps "{self.config.get_path_od_distance_steps()}"'
            )

        except:
            n_nodes = len(nw.tenv.distance_matrix)
            od_costs_dict = np.zeros((n_nodes, n_nodes))
            od_dists_step = np.zeros((n_nodes, n_nodes), dtype=np.int16)
            od_fares_dict = defaultdict(lambda: np.zeros((n_nodes, n_nodes)))
            od_penalties_dict = defaultdict(
                lambda: np.zeros((n_nodes, n_nodes))
            )
            rebalancing_targets = np.zeros(())
            for o in self.points:
                for d in self.points:
                    dist_trip = nw.get_distance(o.id, d.id)
                    od_dists_step = self.get_travel_time_od(o, d, unit="step")

                    # Travel cost
                    cost = self.config.get_travel_cost(dist_trip)
                    od_costs_dict[o.id][d.id] = cost

                    # Save the fare for each class
                    for sq in self.config.trip_base_fare.keys():
                        od_fares_dict[sq][o.id][
                            d.id
                        ] = self.config.calculate_fare(dist_trip, sq_class=sq)
                        od_penalties_dict[sq][o.id][
                            d.id
                        ] = self.online_penalty(o.id, d.id, sq)

            print(f"Saving OD costs at '{self.config.get_path_od_costs()}'...")
            np.save(self.config.get_path_od_costs(), od_costs_dict)

            print(
                f"Saving OD fares for each class at '{self.config.get_path_od_fares()}'..."
            )
            np.save(self.config.get_path_od_fares(), dict(od_fares_dict))

            print(
                f"Saving OD penalties for each class at '{self.config.get_path_od_penalties()}'..."
            )
            np.save(
                self.config.get_path_od_penalties(), dict(od_penalties_dict)
            )

            print(
                f"Saving OD distance steps '{self.config.get_path_od_distance_steps()}'..."
            )
            np.save(
                self.config.get_path_od_distance_steps(), dict(od_dists_step)
            )

            self.od_costs_dict = od_costs_dict
            self.od_fares_dict = od_fares_dict
            self.od_penalties_dict = od_penalties_dict
            self.od_dists_step = od_dists_step

    def online_revenue(self, o, d, sq):
        # From trip's origin to trip's destination
        dist_trip = nw.get_distance(o, d)
        # Base fare + distance cost
        revenue = self.config.calculate_fare(dist_trip, sq_class=sq)
        return revenue

    def loaded_revenue(self, o, d, sq):
        return self.od_fares_dict[sq][o][d]

    def loaded_penalty(self, o, d, sq):
        return self.od_penalties_dict[sq][o][d]

    def online_costs(self, o, d):
        """Calculate costs online using data from configuration"""

        distance_km = nw.get_distance(o, d)
        cost = self.config.get_travel_cost(distance_km)

        return cost

    def loaded_costs(self, o, d):
        return self.od_costs_dict[o][d]

    def loaded_od_dist_step(self, o, d):
        return self.od_distance_steps[o][d]

    def get_hired_step(self):

        if self.config.fav_fleet_size == 0:
            return []

        step_hire = self.get_fav_info(
            max_contract_duration=self.config.max_contract_duration
        )

        # np.save(
        #     f"{FAV_DATA_PATH}step_fav_fleet.npy",
        #     step_hire,
        #     allow_pickle=True
        # )

        hired_cars = {
            step: [
                HiredCar(
                    self.points[depot_id],
                    contract_duration_h,
                    current_step=step,
                    current_arrival=(
                        self.config.reposition_h
                        + earliest_h
                        - self.config.demand_earliest_hour
                    )
                    * 60,
                    duration_level=self.config.contract_duration_level,
                )
                for depot_id, earliest_h, contract_duration_h, deadline_h in car_info
            ]
            for step, car_info in step_hire.items()
        }

        # print(f"Finished hiring cars. Total={sum([len(cars) for s,cars in hired_cars.items()])}")

        return hired_cars

    def total_cost(self, t, d):
        return (
            self.cost_func(d)
            + self.config.discount_factor * self.post_cost(t, d)[0]
        )

    def total_cost_ucb(self, t, d):
        cost = (
            self.cost_func(d)
            + self.config.discount_factor * self.post_cost(t, d)[0]
        )
        if du.ACTION != du.TRIP_DECISION:
            # Number of times we have sampled action
            n = self.t_pos_count[
                (t, d[du.POSITION], du.ACTION, du.DESTINATION)
            ]

            decay_factor = math.sqrt(2 * math.log(self.adp.n + 1) / (n + 1))

            # TODO define proper MAXCOST (<> 2.4)
            upper_bound = cost + min(2.4, 2.4 * decay_factor)
        else:
            # Picking up is always better
            upper_bound = cost + 2.4

        return upper_bound

    def toggle_fleet(self, car_type):
        """Disable/enable all vehicles of a car_type.
        Must be executed in sequence.

        Parameters
        ----------
        car_type : int
            Type of car to enable or disable.

        Example
        -------
        >>> print("Disable PAVs")
        >>> self.toggle_fleet(self, Car.TYPE_FLEET)
        >>> print("Enable PAVs")
        >>> self.toggle_fleet(self, Car.TYPE_FLEET)

        """

        # Check if toggling is enabled
        if self.config.separate_fleets:

            # If None, remove cars of type = car_type
            if not self.toggled_fleet[car_type]:
                if car_type == Car.TYPE_FLEET:
                    self.toggled_fleet[car_type] = (
                        self.hired_cars,
                        self.available_hired,
                    )
                    self.hired_cars, self.available_hired = [], []
                else:
                    self.toggled_fleet[car_type] = self.cars, self.available
                    self.cars, self.available = [], []

            # Activate cars of type = car_type
            else:
                if car_type == Car.TYPE_FLEET:
                    self.hired_cars, self.available_hired = self.toggled_fleet[
                        car_type
                    ]
                else:
                    self.cars, self.available = self.toggled_fleet[car_type]

    # @functools.lru_cache(maxsize=None)
    def cost_func(self, decision, ignore_rebalance_costs=False):
        """Return decision cost
        
        Parameters
        ----------
        decision : tuple
            Decision tuple (decision_id, car_attribute(size=5), o, d, class)
        ignore_rebalance_costs : bool, optional
            Ignore rebalance costs for random rebalance, such that 
            moving and staying have the same costs, by default False
        
        Returns
        -------
        float
            Cost of the decision
        """

        # Platform's profit margin is lower when using hired cars
        PROFIT_MARGIN = 1

        # When moving a hired car, the platform is charged 2X more
        # since this car must return to its origin point
        RETURN_FACTOR = 1

        CONGESTION_PRICE = 0

        COST_STAY = self.config.parking_cost_step

        if decision[du.CAR_TYPE] == Car.TYPE_HIRED:
            PROFIT_MARGIN = self.config.profit_margin
            RETURN_FACTOR = 2
            # Car is parking somewhere different than its parking lot
            if decision[du.DESTINATION] != decision[du.ORIGIN]:
                COST_STAY = 2 * self.config.recharge_cost_distance

        if decision[du.ACTION] == du.STAY_DECISION:
            # Stay
            return -COST_STAY

        elif decision[du.ACTION] == du.TRIP_DECISION:

            # From car's position to trip's origin, and OD
            cost_pickup = self.cost(decision[du.POSITION], decision[du.ORIGIN])

            penalty_pk = self.penalty(
                decision[du.POSITION],
                decision[du.ORIGIN],
                decision[du.SQ_CLASS],
            )

            cost_trip = self.cost(
                decision[du.ORIGIN], decision[du.DESTINATION]
            )

            # Travel cost
            cost = cost_pickup + cost_trip + penalty_pk

            # Base fare + distance cost
            revenue = self.revenue(
                decision[du.ORIGIN],
                decision[du.DESTINATION],
                decision[du.SQ_CLASS],
            )

            contribution = PROFIT_MARGIN * (revenue - cost) - CONGESTION_PRICE

            # if decision[du.CAR_TYPE] == Car.TYPE_VIRTUAL:
            #     return -contribution

            # Profit to service trip
            return contribution

        elif decision[du.ACTION] == du.RECHARGE_DECISION:

            # Recharge
            cost = self.config.cost_recharge_single_increment
            return -cost

        elif decision[du.ACTION] in [
            du.REBALANCE_DECISION,
            du.RETURN_DECISION,
        ]:
            if ignore_rebalance_costs:
                return 0
            else:

                # From trip's origin to trip's destination
                cost = self.cost(decision[du.ORIGIN], decision[du.DESTINATION])

                reb_cost = -RETURN_FACTOR * cost - CONGESTION_PRICE
                # print(action, pos, decision[du.ORIGIN], d, car_type, sq_class, reb_cost)
                return reb_cost

    def update_middle(self, car, step):
        """Where is the car at current step?
        Update middle point of car. Used together with rebalancing
        interruption method.
        
        Parameters
        ----------
        car : Car
            Car rebalancing
        step : int
            Current step
        """

        # How long since car left previous point?
        elapsed = step * self.config.time_increment - car.previous_arrival

        # SP including o and d
        o, d = car.previous.id, car.point.id
        sp = nw.tenv.sp(o, d)
        legs = zip(sp[:-1], sp[1:])

        # print(">>> ", o, self.points[o], d, self.points[d])

        # print(sp)
        time_legs = 0
        distance_legs = 0

        # Loop legs until finds first point after elapsed time
        for leg_o, leg_d in legs:
            distance_leg = nw.get_distance(leg_o, leg_d)
            distance_legs += distance_leg
            time_leg = self.get_travel_time(distance_leg, unit="min")
            time_legs += time_leg
            # print(leg_o, self.points[leg_o], leg_d, self.points[leg_d], distance_leg, time_legs, elapsed)

            # Car takes "time_legs" to move from "previous_node" to
            # "leg_d"
            if time_legs >= elapsed:
                car.middle_point = self.points[leg_d]
                # Time from current location (middle of edge) to middle
                car.elapsed = time_legs - elapsed

                # Distance from current location (middle of edge) to middle
                car.elapsed_distance = self.get_distance_time(car.elapsed)

                # Assume car reaches middle in the current step. In fact,
                # it reaches the middle possibly "elapsed" minutes later
                car.step_m = step

                # Arrival time at middle (from previous)
                car.time_o_m = time_legs

                # Distance to middle (from previous)
                car.distance_o_m = distance_legs

                # Distance remaining to finish rebalancig
                # In case movement is stopped, discount travel costs
                # from this distance in vehicle contribution
                car.remaining_distance = (
                    nw.get_distance(car.previous.id, car.point.id)
                    - car.distance_o_m
                )

                break

    def can_move(
        self,
        pos,
        waypoint,
        target,
        start,
        remaining_hiring_slots,
        delay_offset=0,
    ):
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

        # TODO delay_offset is used when middle point is the current
        # position (it corresponds to the time to reach the middle)
        return remaining_hiring_time > duration_movement + delay_offset

    def get_fav_depots(self):

        if self.config.fav_fleet_size == 0:
            return []

        try:
            P = np.load(self.config.path_depot_list, allow_pickle=True)
            print(
                f'{len(P)} FAV depots loaded ("{self.config.path_depot_list}").'
            )

        except:

            # If FAVs start from region centers (highest level is chosen)
            if self.config.fav_depot_level or self.config.centroid_level > 0:

                level = max(
                    (
                        0
                        if self.config.fav_depot_level is None
                        else self.config.fav_depot_level
                    ),
                    self.config.centroid_level,
                )

                # Node id list
                N = list(
                    set(
                        [
                            self.points[p].id_level(level)
                            for p in range(self.config.node_count)
                        ]
                    )
                )
            else:
                N = list(np.arange(self.config.node_count))

            # If only a share of the nodes is used
            if self.config.depot_share:
                n_depots = int(self.config.depot_share * len(N))

                # If share is lower than 1 (i.e., there are not depots),
                # the entire set is considered.
                if n_depots <= 1:
                    n_depots = len(N)

            else:
                n_depots = len(N)

            # Select n depots from from depot set
            P = sorted(random.sample(set(N), n_depots))

            # Save the depots to use in future iterations. The depots
            # don't change, only the distribution of cars throughout
            # these depots change.
            np.save(self.config.path_depot_list, P, allow_pickle=True)
            print(f"Saving {len(P)} FAV depots.")

        return P

    def get_ab(self, mean, std, clip_a, clip_b, period=0.5):

        bins = int((clip_b - clip_a) * 60 / period)

        a, b = (clip_a - mean) / std, (clip_b - mean) / std

        return a, b, bins

    # TODO Regulate contract durations
    def get_availability_pattern(self):

        avail_a, avail_b, avail_bins = self.get_ab(
            *self.config.fav_availability_features,
            period=self.config.time_increment,
        )
        # print("    Contract duration:", avail_a, avail_b, avail_bins)
        return avail_a, avail_b, avail_bins

    def get_earliest_pattern(self):

        ear_a, ear_b, ear_bins = self.get_ab(
            *(self.config.fav_earliest_features[0:4]),
            period=self.config.time_increment,
        )

        # print("Earliest service time:", ear_a, ear_b, ear_bins)
        return ear_a, ear_b, ear_bins

    def earliest_features(self):
        AggLevel = namedtuple(
            "EarliestDistribution", "mean, std, clip_a, clip_b"
        )

    def get_earliest_time(self, n_favs):

        ear_a, ear_b, ear_bins = self.get_earliest_pattern()

        # Earlist times of FAVs arriving in node n
        earliest_time = (
            truncnorm.rvs(ear_a, ear_b, size=n_favs)
            + self.config.fav_earliest_features[0]
        )

        return earliest_time

    def get_contract_duration(self, n_favs):

        avail_a, avail_b, avail_bins = self.get_availability_pattern()

        # Contract durations of FAVs arriving in node n
        contract_duration = (
            truncnorm.rvs(avail_a, avail_b, size=n_favs)
            + self.config.fav_availability_features[0]
        )

        return contract_duration

    def get_fav_depot_assignment(self, P=None):

        if not P:
            P = self.fav_depots

        # How many FAVs for each depot?
        fav_count_depot = np.zeros(len(P))

        unassigned_favs = self.config.fav_fleet_size

        # Assign favs to depots randomly
        while unassigned_favs > 0:
            fav_count_depot[random.randint(0, len(P) - 1)] += 1
            unassigned_favs -= 1

        return fav_count_depot

    def get_fav_info(self, max_contract_duration=True):

        random.seed(self.config.iteration_step_seed)

        fav_count_depot = self.get_fav_depot_assignment()

        # What time favs arrive in each depot?
        earliest_times = []

        # Correspondent contract durations
        contract_durations = []

        # FAVs per time step (30s = hour*60*2)
        step_fav = defaultdict(list)

        depot_info = dict()
        for i, n in enumerate(self.fav_depots):

            n_favs = int(fav_count_depot[i])

            depot_info[n] = list()

            # Earlist times of FAVs arriving in node n
            earliest_time = self.get_earliest_time(n_favs)

            # Contract durations of FAVs arriving in node n
            contract_duration = self.get_contract_duration(n_favs)

            # Earlist times and contract durations
            for e, c in zip(earliest_time, contract_duration):

                # earliest_request = earliest_features[2]
                # latest_request = earliest_features[3]
                # rebalance_offset = earliest_features[4]
                # delivery_offset = earliest_features[5]

                # Contract duration can't surpass last time period
                if max_contract_duration:
                    # Car stay in the system from the moment
                    # it arrives until the end:
                    # earliest hour + demand interval + rebalance offset
                    c = self.config.latest_hour - e
                else:
                    c = min(c, self.config.latest_hour - e)
                # Add clipped contract duration
                contract_durations.append(c)

                # earliest, service time, deadline
                fav_data = (n, e, c, e + c)

                # FAVs per depot
                depot_info[n].append(fav_data)

                # Time step = 0.5 min (truncate seconds)
                step = self.config.get_step(e)

                # FAVs per step
                step_fav[step].append(fav_data)

            earliest_times.extend(earliest_time)

        # print(len(P), P)

        return dict(step_fav)

    def realize_decision(self, t, decisions, a_trips_dict, a_cars_dict):
        total_reward = 0
        serviced = list()

        # Count number of times each decision was taken
        decision_dict_count = {
            du.STAY_DECISION: 0,
            du.REBALANCE_DECISION: 0,
            du.TRIP_DECISION: 0,
            du.RECHARGE_DECISION: 0,
            du.RETURN_DECISION: 0,
        }

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

            if car_type == Car.TYPE_VIRTUAL:
                continue

            # Track how many times a decision was taken
            decision_dict_count[action] += times

            # Track summary decision for UCB
            # self.t_pos_count[(t, point, action, d)] += times

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

                # Ignores last element (n. times decision was applied)
                contribution_car = self.decision_info[decision[:-1]][0]

                # print(decision, contribution_car)

                # Start contract, if not started
                if car_type == Car.TYPE_HIRED:

                    # Hired car will be used by the system
                    car.started_contract = True

                if action == du.RECHARGE_DECISION:
                    # Recharging ##################################### #
                    cost_recharging = self.recharge(
                        car, self.config.recharge_time_single_level
                    )

                elif action in [du.REBALANCE_DECISION, du.RETURN_DECISION]:
                    # Rebalancing #################################### #
                    (
                        total_duration,
                        total_duration_steps,
                        total_distance,
                        reward,
                    ) = self.rebalance(car, self.points[d])

                    if self.config.activate_thompson:
                        # update beta distribution
                        self.beta_ab[(t, point, d)]["a"] += times
                        self.beta_ab[(t, point, d)]["b"] -= 1

                        if self.beta_ab[(t, point, d)]["b"] <= 0:
                            self.beta_ab[(t, point, d)]["b"] = 1

                    car.move(
                        total_duration,
                        total_duration_steps,
                        total_distance,
                        contribution_car,
                        self.points[d],
                        return_trip=(action == du.RETURN_DECISION),
                    )

                elif action == du.STAY_DECISION:
                    # Car does nothing to alter its state ############ #

                    if car.status == Car.IDLE:
                        car.idle_step_count += 1

                    # Notice that if a rebalancing vehicle cannot pick
                    # up trips, it will be assigned to a STAY decision.
                    # Hence, it will continue the rebalance.

                elif action == du.TRIP_DECISION:
                    # Servicing ###################################### #

                    if car.status == Car.REBALANCE:
                        # Car was previously rebalancing. Thus, the
                        # rebalancing movement has to be interrupted.
                        car.interrupt_rebalance()

                        # Car did not finish rebalance. Thus, return to
                        # platform the costs not yet charged regarding the
                        # rest of the rebalancing path.
                        total_reward += self.config.get_travel_cost(
                            car.remaining_distance
                        )

                    # How long each request has been waiting?
                    trip_delays = [
                        trip.max_delay_from_placement + trip.backlog_delay
                        for trip in a_trips_dict[(o, d)]
                    ]

                    # Request waiting the longest first
                    i_max_wait = np.argmax(trip_delays)

                    # Get a trip to apply decision
                    trip = a_trips_dict[(o, d)].pop(i_max_wait)

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

    def get_neighbors(self, car_id):

        # Rebalancing ################################################ #
        if self.config.reachable_neighbors:

            neighbors = self.reachable_neighbors(
                car_id, self.config.time_increment * 60
            )
        else:
            # Filter targets farther than "time_increment" and limit
            # the max. number of targets. If rebalance_max_targers is
            # None, does not filter.
            neighbors = self.get_zone_neighbors(car_id)

        return neighbors

    @property
    def level_inbound_dict(self):
        """Dictionary of current cars inbound to each position"""
        return self.level_step_inbound_cars[self.config.centroid_level]

    def update_fleet_status(self, time_step, use_rebalancing_cars=False):
        """Update the status of the fleet (PAVs and FAVs) at time step.
         - List of cars per attribute
         - List of cars per location
         - List of cars returning to home station
         - List of cars inbound to each node

        Depending on the method, update can also add rebalancing cars
        to the list of available cars.

        Parameters
        ----------
        time_step : int
            Current time step of the simulation
        use_rebalancing_cars: bool
            Rebalancing cars are added to the list of available vehicles.
            We consider they can be rerouted from their closest node
            they are at in their shortest path, accordind to the current
            time step.
        """

        # List of cars per attribute
        self.attribute_cars_dict = defaultdict(list)

        # If rebalancing targets are not removed (due to tabu list)
        # the dictionary can be used again

        # if self.config.car_size_tabu > 0:
        # Set of reachable neighbors from each car position
        # self.attribute_rebalance = dict()

        # List of cars per location
        self.cars_location = defaultdict(lambda: defaultdict(list))

        # List of hired cars inbound to their on station
        self.cars_stationed = defaultdict(list)

        # List of cars inbound to a position. Do not account for FAVs
        # inbound to their stations.
        self.cars_inbound_to = defaultdict(list)

        self.level_step_inbound_cars = defaultdict(
            lambda: defaultdict(lambda: defaultdict(list))
        )

        # Vehicles stopped at location do not visit tabu list
        self.cars_location_tabu = defaultdict(set)

        # List of cars per region center
        self.count_car_region = defaultdict(lambda: defaultdict(int))

        # List of free cars per region center
        self.count_available_car_region = defaultdict(lambda: defaultdict(int))

        # Avaialable carss

        # Idle company-owned cars
        available = []
        rebalancing = []
        for car in self.cars:
            # print(" -1 ", car.__repr__())
            # Check if vehicles finished their tasks
            # Where are the cars?
            # What are they doing at the current step?
            # t ----- t+1 ----- t+2 ----- t+3 ----- t+4 ------- t+5
            # --trips----trips------trips-----trips------trips-----
            car.update(time_step, time_increment=self.config.time_increment)
            # print(" -2 ", car.__repr__())

            # Inbound location of car
            self.cars_location[car.point.id][car.type].append(car)

            # Discard busy vehicles
            if not car.busy:
                available.append(car)

                # Free car count per region center
                for g in range(len(self.config.level_dist_list)):
                    self.count_available_car_region[g][
                        car.point.id_level(g)
                    ] += 1

                self.attribute_cars_dict[car.attribute].append(car)

                # Get accessible neighbors from each car position
                # if car.point.id not in self.attribute_rebalance:
                #     self.attribute_rebalance[car.point.id] = self.neighbors[
                #         car.point.id
                #     ]

                # Get union set of tabu locations to visit
                if self.config.car_size_tabu > 0:
                    self.cars_location_tabu[car.point.id] |= set(car.tabu)
            else:
                # Car is moving (rebalancing or servicing)
                if use_rebalancing_cars and car.status == Car.REBALANCE:

                    # Find car's current position
                    self.update_middle(car, time_step)

                    # print(car.m_data())

                    rebalancing.append(car)

                    # print("REBALANCE FROM MIDDLE")
                    self.attribute_cars_dict[car.attribute].append(car)

                # Busy cars arriving at each location
                self.cars_inbound_to[car.point.id].append(car)

            # Car count per region center
            for g in range(len(self.config.level_dist_list)):
                self.count_car_region[g][car.point.id_level(g)] += 1

                # When each car will become available
                self.level_step_inbound_cars[g][car.point.id_level(g)][
                    car.step
                ].append(car)

        # Idle hired cars
        available_hired = []
        rebalancing_hired = []

        # List of cars whose contracts have expired
        expired_contract = []

        for car in self.hired_cars:

            # Check if vehicles finished their tasks
            # Where are the cars?
            # What are they doing at the current step?
            # t ----- t+1 ----- t+2 ----- t+3 ----- t+4 ------- t+5
            # --trips----trips------trips-----trips------trips-----
            car.update(time_step, time_increment=self.config.time_increment)

            # Inbound location of car
            self.cars_location[car.point.id][car.type].append(car)

            if car.point.id == car.origin.id:
                self.cars_stationed[car.point.id].append(car)

            # Car has started the contract
            # if car.started_contract:

            # Contract duration has expired
            if car.contract_duration == 0:
                expired_contract.append(car)
                # self.hired_cars.remove(car)
                self.available_hired_ids[car.point.id] += 1

            # Discard busy vehicles
            elif not car.busy:
                available_hired.append(car)

                # Free car count per region center
                for g in range(len(self.config.level_dist_list)):
                    self.count_available_car_region[g][
                        car.point.id_level(g)
                    ] += 1

                self.attribute_cars_dict[car.attribute].append(car)

                # TODO Remove from logic, it is necessary to use if
                # tabu list is on (new_attribute_rebalance)
                # Get accessible neighbors from each car position
                # if car.point.id not in self.attribute_rebalance:
                #     self.attribute_rebalance[car.point.id] = self.neighbors[
                #         car.point.id
                #     ]

                if self.config.car_size_tabu > 0:
                    self.cars_location_tabu[car.point.id] |= set(car.tabu)

            else:
                # Only account for FAVs moving to positions different
                # than their own stations
                if car.point.id != car.origin.id:
                    self.cars_inbound_to[car.point.id].append(car)

                if use_rebalancing_cars and car.status == Car.REBALANCE:

                    self.update_middle(car, time_step)

                    # print(car.m_data())

                    rebalancing_hired.append(car)

                    self.attribute_cars_dict[car.attribute].append(car)

                # print(f"{car.previous.id} -> {car.point.id} - timestep=({time_step}) ({car.elapsed}) - middle={car.middle_point} - elapsed={car.elapsed}")

            # Car count per region center
            for g in range(len(self.config.level_dist_list)):
                self.count_car_region[g][car.point.id_level(g)] += 1

        self.available = available
        self.available_hired = available_hired
        self.rebalancing = rebalancing
        self.rebalancing_hired = rebalancing_hired

        # Remove expired contract cars
        for car in expired_contract:
            self.hired_cars.remove(car)

        # Save expired contract cars
        self.expired_contract_cars.extend(expired_contract)

        # Vehicles learn together fresh neighbors to explore
        if self.config.car_size_tabu > 0:
            new_attribute_rebalance = dict()
            for p, reb in self.attribute_rebalance.items():
                fresh_neighbors = reb - self.cars_location_tabu[p]
                new_attribute_rebalance[p] = fresh_neighbors

                # Print tabu operations
                # if len(self.cars_location_tabu[p]) > 1:
                #     print(
                #         f"{p:04}: neighbors({len(reb)})={reb}, "
                #         f"tabu({len(self.cars_location_tabu[p])})="
                #         f"{self.cars_location_tabu[p]}, "
                #         f"fresh neighbors({len(new_attribute_rebalance[p])})="
                #         f"{new_attribute_rebalance[p]}"
                #     )

            self.attribute_rebalance = new_attribute_rebalance
            # pprint(self.cars_location_tabu)

    def show_count_vehicles_top(self, step, n):
        count_tuples = [
            (
                pos,
                len(type_car[Car.TYPE_FLEET]),
                len(type_car[Car.TYPE_HIRED]),
                len(type_car[Car.TYPE_FLEET]) + len(type_car[Car.TYPE_HIRED]),
            )
            for pos, type_car in self.cars_location.items()
        ]

        count_pav = sorted(
            count_tuples, reverse=True, key=lambda x: (x[1], x[2], x[3])
        )[:10]

        count_fav = sorted(
            count_tuples, reverse=True, key=lambda x: (x[2], x[1], x[3])
        )[:10]

        count_total = sorted(
            count_tuples, reverse=True, key=lambda x: (x[3], x[1], x[2])
        )[:10]

        count_location_stationed = sorted(
            [
                len(cars) + len(self.cars_stationed.get(pos, []))
                for pos, cars in self.cars_location.items()
            ][:n],
            reverse=True,
        )

        print(
            f"{step} -     PAV={count_pav}\n"
            f"{step} -     FAV={count_fav}\n"
            f"{step} -   Total={count_total}\n"
            f"{step} - PAV-FAV={count_location_stationed}\n"
        )
        pprint(self.cars_stationed)

    def discard_excess_hired(self):

        # Car was not used in last iteration
        active_fleet = []

        for car in self.hired_cars:
            if car.started_contract:
                active_fleet.append(car)

        self.hired_cars = active_fleet

    # @functools.lru_cache(maxsize=None)
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
            tuple: (
                time_step,
                point,
                battery_post,
                contract_duration,
                type_post,
                car_origin
            )
        """

        (
            action,
            point,
            battery,
            contract_duration,
            car_type,
            car_origin,
            o,
            d,
            _,
        ) = decision

        battery_post = battery
        type_post = car_type

        if action == du.RECHARGE_DECISION:

            # Recharging ###############################################
            battery_post = min(self.battery_levels, battery + 1)
            time_step += self.config.recharge_time_single_level
            duration = self.config.recharge_time_single_level

        elif action in [du.REBALANCE_DECISION, du.RETURN_DECISION]:
            # Rebalancing ##############################################

            duration, battery_drop = self.preview_move(point, o, d)
            time_step += max(1, duration)
            battery_post = max(0, battery - battery_drop)
            point = d

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

        # Contract duration is altered only for hired cars
        if car_type == Car.TYPE_HIRED:
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
        fav_status_count = defaultdict(int)
        pav_status_count = defaultdict(int)
        for s in Car.status_list:
            status_count[s] = 0
            pav_status_count[s] = 0
            fav_status_count[s] = 0

        total_battery_level = 0
        for c in self.cars:
            # total_battery_level += c.battery_level_miles
            status_count[c.status] += 1
            pav_status_count[c.status] += 1

        for c in self.hired_cars:
            if c.started_contract:
                # total_battery_level += c.battery_level_miles
                status_count[c.status] += 1
                fav_status_count[c.status] += 1

        return (
            status_count,
            pav_status_count,
            fav_status_count,
            total_battery_level,
        )

    def reset(self, seed=1):

        super().reset(seed=seed)
        np.random.seed(seed=seed)
        self.hired_cars = []
        self.overall_hired = []
        self.step_favs = self.get_hired_step()
        self.expired_contract_cars = []
        self.available_hired = []
        self.rebalancing_hired = []
        # self.post_cost.cache_clear()
        self.adp.weighted_values.clear()
        self.cur_step = 0

    @property
    def available_fleet_size(self):
        """Sum of PAV and FAV fleet sizes"""
        return len(self.available) + len(self.available_hired)

    def get_fleet_df(self):
        print(
            f"Saving fleet data - #PAVs={len(self.cars)}"
            f" - #FAVs={len(self.overall_hired)}."
        )
        d = defaultdict(list)

        for car in itertools.chain(self.cars, self.overall_hired):

            d["id"].append(car.id)
            d["type"].append(car.type)
            # Current node or destination
            d["point"].append(car.point)
            d["waypoint"].append(car.waypoint)
            # Last point visited
            d["previous"].append(car.previous)
            # Starting point
            d["origin"].append(car.origin)

            # Middle point data
            d["middle_point"].append(car.middle_point)
            d["elapsed_distance"].append(car.elapsed_distance)
            d["time_o_m"].append(car.time_o_m)
            d["distance_o_m"].append(car.distance_o_m)
            d["elapsed"].append(car.elapsed)
            d["remaining_distance"].append(car.remaining_distance)
            d["step_m"].append(car.step_m)

            d["idle_step_count"].append(car.idle_step_count)
            d["interrupted_rebalance_count"].append(
                car.interrupted_rebalance_count
            )

            d["tabu"].append(car.tabu)

            d["battery_level"].append(car.battery_level)

            d["trip"].append(car.trip)
            d["point_list"].append(car.point_list)

            d["arrival_time"].append(car.arrival_time)
            d["previous_arrival"].append(car.previous_arrival)
            d["previous_step"].append(car.previous_step)
            d["step"].append(car.step)
            d["revenue"].append(car.revenue)
            d["n_trips"].append(car.n_trips)
            d["distance_traveled"].append(car.distance_traveled)

            # Vehicle starts free to operate
            d["status"].append(car.status)
            d["curret_trip"].append(car.current_trip)

            d["time_status"].append(car.time_status)

            # Regular cars are always available
            d["contract_duration"].append(car.contract_duration)

        df = pd.DataFrame.from_dict(dict(d))

        return df

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

    def post_cost(self, t, decision):

        # Target attribute if decision was taken
        post_state = self.preview_decision(t, decision)

        if post_state[du.CAR_TYPE] == Car.TYPE_VIRTUAL:
            edit_post = list(post_state)
            edit_post[du.CAR_TYPE] = Car.TYPE_FLEET
            post_state = tuple(edit_post)

        if post_state[adp.adp.TIME] >= self.config.time_steps:
            return 0, post_state

        # Get the post decision state estimate value based on
        # hierarchical aggregation
        estimate = self.adp.get_weighted_value(post_state)

        # Penalize long rebalancing decisions
        if (
            decision[0] == du.REBALANCE_DECISION
            and self.config.penalize_rebalance
        ):

            avg_busy_stay = 0

            post_time = post_state[adp.adp.TIME]

            # Rebalancing is longer than one time step
            # t + 1 is allowed because the resource is guaranteed to
            # be available in the next period
            if post_time > t + 1:

                for busy_reb_t in range(t + 1, post_state[adp.adp.TIME]):

                    stay = (du.STAY_DECISION,) + decision[1:]

                    # Target attribute if decision was taken
                    stay_post_state = self.preview_decision(busy_reb_t, stay)

                    estimate_stay = self.adp.get_weighted_value(
                        stay_post_state
                    )

                    avg_busy_stay += estimate_stay

                if avg_busy_stay > 0:

                    # avg_stay = avg_busy_stay / (post_t - t + 1)
                    # avg_stay = avg_busy_stay
                    # print(
                    #     f"t:{t} - post_t={post_t} - "
                    #     f"Stay: {np.arange(t + 1, post_t+1)} = "
                    #     f"{avg_busy_stay} (avg={avg_stay:6.2f}, "
                    #     f"previous={estimate:6.2f}, "
                    #     f"new={estimate-avg_stay:6.2f}"
                    # )
                    # Discount the average contribution that would have
                    # been gained if the car stayed still instead of
                    # rebalancing
                    estimate = max(0, estimate - avg_busy_stay)

        return estimate, post_state

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
