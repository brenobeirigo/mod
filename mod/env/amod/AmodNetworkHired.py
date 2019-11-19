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
from mod.env.config import FOLDER_EPISODE_TRACK, FAV_DATA_PATH
import requests
import functools
from mod.env.amod.AmodNetwork import AmodNetwork
import mod.env.decisions as du
from copy import deepcopy
import math

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
            if alpha * tmp + (gamma * v) - 1.3862944 >= \
                np.log(u1 * u1 * u2):
                break
        x = w / (b + w)
        return x

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
        self.available_hired_ids = np.zeros(len(self.points_level[0]))
        self.expired_contract_cars = []

        # Used to execute the method in sequence for vehicles of
        # different types. For example, first execute only for PAVs, and
        # then only for FAVs.
        self.toggled_fleet = dict()
        self.toggled_fleet[Car.TYPE_FLEET] = None
        self.toggled_fleet[Car.TYPE_HIRED] = None

        # UCB
        #self.t_pos_count = defaultdict(int)

        # Thompson
        self.beta_ab = defaultdict(lambda: dict(a=1, b=1))
        self.beta_sampler = BetaSampler(1)
        self.cur_step = 0

        # Set of reachable neighbors from each car position
        self.attribute_rebalance = dict()

    def get_hired_step(self):

        if self.config.fav_fleet_size == 0:
            return []

        step_hire = self.get_fav_info()

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
                    current_arrival=(self.config.reposition_h + earliest_h - self.config.demand_earliest_hour) * 60,
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
            n = self.t_pos_count[(t, d[du.POSITION], du.ACTION, du.DESTINATION)]

            decay_factor = (
                math.sqrt(2 * math.log(self.adp.n + 1)/(n+1))
            )

            # TODO define proper MAXCOST (<> 2.4)
            upper_bound = cost + min(2.4, 2.4*decay_factor)
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
                    self.toggled_fleet[car_type] = self.hired_cars, self.available_hired
                    self.hired_cars, self.available_hired = [], []
                else:
                    self.toggled_fleet[car_type] = self.cars, self.available
                    self.cars, self.available = [], []

            # Activate cars of type = car_type
            else:
                if car_type == Car.TYPE_FLEET:
                    self.hired_cars, self.available_hired = self.toggled_fleet[car_type]
                else:
                    self.cars, self.available = self.toggled_fleet[car_type]

    # @functools.lru_cache(maxsize=None)
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

        COST_STAY = self.config.parking_cost_step

        if decision[du.CAR_TYPE] == Car.TYPE_HIRED:
            PROFIT_MARGIN = self.config.profit_margin
            RETURN_FACTOR = 2
            # Car is parking somewhere different than its parking lot
            if decision[du.DESTINATION] != decision[du.ORIGIN]:
                COST_STAY = 2*self.config.recharge_cost_distance

        if decision[du.ACTION] == du.STAY_DECISION:
            # Stay
            return -COST_STAY

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

            # if decision[du.CAR_TYPE] == Car.TYPE_VIRTUAL:
            #     return -contribution

            # Profit to service trip
            return contribution

        elif decision[du.ACTION] == du.RECHARGE_DECISION:

            # Recharge
            cost = self.config.cost_recharge_single_increment
            return -cost

        elif decision[du.ACTION] in [du.REBALANCE_DECISION, du.RETURN_DECISION]:

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

    def get_fav_depots(self):

        if self.config.fav_fleet_size == 0:
            return []

        try:
            P = np.load(self.config.path_depot_list, allow_pickle=True)
            print(f"{len(P)} FAV depots loaded.")

        except:

            # If FAVs start from region centers
            if self.config.fav_depot_level:

                # Node id list
                N = list(set([self.points[p].id_level(self.config.fav_depot_level) for p in range(self.config.node_count)]))
            else:
                N = list(np.arange(self.config.node_count))

            # If only a share of the nodes is used
            if self.config.depot_share:
                n_depots = int(self.config.depot_share*len(N))

                # If share is lower than 1 (i.e., there are not depots),
                # the entire set is considered.
                if  n_depots <= 1:
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

    def get_fav_depot_assignment(self, P=None):

        if not P:
            P = self.fav_depots

        # How many FAVs for each depot?
        fav_count_depot = np.zeros(len(P))

        unassigned_favs = self.config.fav_fleet_size

        # Assign favs to depots randomly
        while unassigned_favs > 0:
            fav_count_depot[random.randint(0, len(P)-1)] += 1
            unassigned_favs -= 1

        return fav_count_depot

    def get_fav_info(self, max_contract_duration=True):

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
            earliest_time = self.config.get_earliest_time(n_favs)

            # Contract durations of FAVs arriving in node n
            contract_duration = self.config.get_contract_duration(n_favs)

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
                contribution_car = self.cost_func(decision[:-1])

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
                    # Car settings are updated all together when time
                    # step finishes
                    car.idle_step_count += 1

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

    def get_neighbors(self, car_id):

        # Rebalancing ################################################ #
        if self.config.reachable_neighbors:

            neighbors = self.reachable_neighbors(
                car_id, self.config.time_increment*60
            )
        else:
            neighbors = self.get_zone_neighbors(car_id)

        return neighbors

    def update_fleet_status(self, time_step):

        # List of cars per attribute
        self.attribute_cars_dict = defaultdict(list)

        # If rebalancing targets are not removed (due to tabu list)
        # the dictionary can be used again
        if self.config.car_size_tabu > 0:
            # Set of reachable neighbors from each car position
            self.attribute_rebalance = dict()

        # List of cars per location
        self.cars_location = defaultdict(list)

        # Vehicles stopped at location do not visit tabu list
        self.cars_location_tabu = defaultdict(set)

        # List of cars per region center
        self.count_car_region = defaultdict(lambda: defaultdict(int))

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

                self.attribute_cars_dict[car.attribute].append(car)

                # Get accessible neighbors from each car position
                if car.point.id not in self.attribute_rebalance:
                    self.attribute_rebalance[car.point.id] = self.get_neighbors(car.point.id)

                # Get union set of tabu locations to visit
                if self.config.car_size_tabu > 0:
                    self.cars_location_tabu[car.point.id] |= set(car.tabu)
            else:
                # Busy cars arriving at each location
                self.cars_location[car.point.id].append(car)

            # # Car count per region center
            # for g in range(len(self.config.level_dist_list)):
            #     self.count_car_region[g][self.points[car.point.id].id_level(g)]+= 1

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
            # if car.started_contract:

            # Contract duration has expired
            if car.contract_duration == 0:
                # expired_contract.append(car)
                expired_contract.append(car)
                # self.hired_cars.remove(car)
                self.available_hired_ids[car.point.id] += 1

            # Discard busy vehicles
            elif not car.busy:
                available_hired.append(car)

                self.attribute_cars_dict[car.attribute].append(car)

                # Get accessible neighbors from each car position
                if car.point.id not in self.attribute_rebalance:
                    self.attribute_rebalance[car.point.id] = self.get_neighbors(car.point.id)

                if self.config.car_size_tabu > 0:
                    self.cars_location_tabu[car.point.id] |= set(car.tabu)
            
            else:
                # Busy cars arriving at each location
                self.cars_location[car.point.id].append(car)

            # # Car count per region center
            # for g in range(len(self.config.level_dist_list)):
            #     self.count_car_region[g][self.points[car.point.id].id_level(g)] += 1

        self.available = available
        self.available_hired = available_hired

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

        return status_count, pav_status_count, fav_status_count, total_battery_level

    def reset(self):

        super().reset()
        self.hired_cars = []
        self.step_favs = self.get_hired_step()
        self.expired_contract_cars = []
        self.available_hired = []
        # self.post_cost.cache_clear()
        self.adp.weighted_values.clear()
        self.cur_step = 0
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
