from collections import defaultdict
import mod.env.network as nw
from gurobipy import tuplelist, GRB, Model, quicksum
from pprint import pprint
from mod.env.amod import Amod
from mod.env.trip import ClassedTrip
from mod.env.car import Car
import time
import numpy as np
import itertools

# Decisions are tuples following the format
# (ACTION, POSITION, BATTERY, ORIGIN, DESTINATION)

# Labels for decision tuples
ACTION = 0
POSITION = 1
BATTERY = 2
ORIGIN = 3
DESTINATION = 4
CAR_TYPE = 5

# #################################################################### #
# Manipulate solutions ############################################### #
# #################################################################### #


def extract_duals(flow_cars, car_attributes):
    duals = dict()
    for o, l in car_attributes:
        c = flow_cars[o, l]
        if c.pi > 0:
            # pi = The constraint dual value in the current solution
            # (also known as the shadow price).
            duals[(o, l)] = c.pi
        # print(f'The dual value of {c.constrName} : {c.pi}')
    return duals


def extract_duals_point(flow_cars, car_attributes):
    duals = dict()
    for point, battery_level in car_attributes:
        c = flow_cars[point, battery_level]
        if c.pi > 0:
            # pi = The constraint dual value in the current solution
            # (also known as the shadow price).
            duals[(point, battery_level)] = c.pi
        # print(f'The dual value of {c.constrName} : {c.pi}')
    return duals


def extract_solution(var_list, flow_cars=None):

    # list of decision tuples (action, point, level, o, d)
    decisions = list()

    # Dual values for car attributes (point, level)
    duals = dict()

    # Loop (decision tuple, var) pairs
    for decision, var in var_list.items():

        if var.x > 0.9:
            decisions.append(decision + (round(var.x),))

        if flow_cars:
            # FIll out duals
            car_attribute = (decision[POSITION], decision[BATTERY])

            if car_attribute not in duals:
                # pi = The constraint dual value in the current solution
                # (also known as the shadow price).
                c = flow_cars[car_attribute].pi
                if c > 0:
                    duals[car_attribute] = c

    return decisions, duals


# #################################################################### #
# Manipulate decisions ############################################### #
# #################################################################### #


def print_decisions(x_rebalance, x_stay, x_pickup, x_recharge, all_decisions):
    print("#### Decision variables - (action, point, level, o, d) ####")
    print("# REBALANCE")
    pprint(x_rebalance)

    print("# STAY")
    pprint(x_stay)

    print("# PICKUP")
    pprint(x_pickup)

    print("# RECHARGE")
    pprint(x_recharge)

    print(
        "\n#### Count ####\n"
        f"#Pickup: {len(x_pickup)}"
        f" - #Stay: {len(x_stay)}"
        f" - #Recharge: {len(x_recharge)}"
        f" - #Rebalance: {len(x_rebalance)}"
        f" - #Total: {len(all_decisions)}"
    )


def get_decision_tuples(car_attributes, neighbors, trip_ods):

    x_stay = [
        (Amod.TRIP_STAY_DECISION,) + c + (c[0],) + (c[0],)
        for c in car_attributes
    ]

    x_pickup = [
        (Amod.TRIP_STAY_DECISION,) + c + (o,) + (d,)
        for o, d in trip_ods
        for c in car_attributes
        if o in neighbors[c[0]]
    ]

    x_recharge = [
        (Amod.RECHARGE_REBALANCE_DECISION,) + c + (c[0],) + (c[0],)
        for c in car_attributes
    ]

    x_rebalance = [
        (Amod.RECHARGE_REBALANCE_DECISION,) + c + (c[0],) + (z,)
        for c in car_attributes
        for z in neighbors[c[0]]
        if c[0] != z
    ]

    # Enumerated list of decisions
    all_decisions = tuplelist(x_pickup + x_stay + x_rebalance + x_recharge)

    # print_decisions(x_rebalance, x_stay, x_pickup, x_recharge, all_decisions)

    return all_decisions


def get_decision_set(
    car_attributes,
    neighbors,
    trip_ods,
    rebalance_neighbors,
    max_battery_level=20,
):
    decisions = set()

    for (pos_level, battery), car_list in car_attributes.items():
        for car in car_list:

            # Stay #####################################################
            decisions.add(
                (Amod.STAY_DECISION,)
                + (car.point.id, battery)
                + (car.point.id,)
                + (car.point.id,)
            )

            # Trip #####################################################
            for (o, d), trip_list in trip_ods.items():
                for trip in trip_list:
                    if o in neighbors[pos_level]:
                        decisions.add(
                            (Amod.TRIP_DECISION,)
                            + (car.point.id, battery)
                            + (trip.o.id,)
                            + (trip.d.id,)
                        )

            # Recharge #################################################
            if battery < max_battery_level:
                decisions.add(
                    (Amod.RECHARGE_DECISION,)
                    + (car.point.id, battery)
                    + (car.point.id,)
                    + (car.point.id,)
                )

            # Rebalancing ##############################################
            for neighbor in rebalance_neighbors[car.point.id]:
                decisions.add(
                    (Amod.REBALANCE_DECISION,)
                    + (car.point.id, battery)
                    + (car.point.id,)
                    + (neighbor,)
                )

    return decisions


# #################################################################### #
# Methods ############################################################ #
# #################################################################### #


def adp_network(
    env,
    trips,
    time_step,
    charge=True,
    agg_level=None,
    myopic=False,
    neighborhood_level=0,
    n_neighbors=4,
):
    """Assign trips to available vehicles optimally at the current
        time step.

    Parameters
    ----------
    env : Environment
        AMoD environment
    trips : list
        List of trips
    time_step : int
        Time step after receiving trips

    charge : bool, optional
        Apply the charging constraint, by default True
    agg_level : [type], optional
        Attributes are queried according to an aggregation level, 
        by default None
    myopic : bool, optional
        If True, does not learn between iterations, by default False
    neighborhood_level : int, optional
        How large are region centers (
            e.g., 1 = reachable in 1min,
                  2 = reachable in 2min
            ), by default 1
    n_neighbors : int, optional
        Max. neighbors of region centers, by default 4

    Returns
    -------
    float, list, list
        total contribution, serviced trips, rejected trips
    """

    # Starting assignment model
    m = Model("assignment")

    # Disables all logging (file and console)
    m.setParam("OutputFlag", 0)

    # if log_path:
    #     m.Params.LogFile = "{}/region_centers_{}.log".format(
    #         log_path, max_delay
    #     )

    #     m.Params.ResultFile = "{}/region_centers_{}.lp".format(
    #         log_path, max_delay
    #     )

    # ##################################################################
    # SORT CARS ########################################################
    # ##################################################################

    (
        reachable_points,
        cars_with_attribute_neighborhood,
        dict_attribute_neighbors,
        dict_attribute_rebalance,
        cars_with_attribute_zero,
    ) = sortout_fleet(
        env,
        time_step,
        neighborhood_level,
        env.config.rebalance_level,
        n_neighbors
    )

    # ##################################################################
    # SORT TRIPS #######################################################
    # ##################################################################

    #  Dictionary of #trips per trip attribute,i.e., (o.id, d.id)
    (
        trips_with_attribute_neighborhood,
        trips_with_attribute_zero,
        rejected
    ) = sortout_trips(trips, neighborhood_level, reachable_points)

    # ##################################################################
    # VARIABLES ########################################################
    # ##################################################################

    decisions = get_decision_set(
        cars_with_attribute_neighborhood,
        dict_attribute_neighbors,
        trips_with_attribute_neighborhood,
        dict_attribute_rebalance,
        max_battery_level=env.config.battery_levels,
    )

    # Adding variables
    x_var = m.addVars(tuplelist(decisions), name="x")

    # ##################################################################
    # MODEL ############################################################
    # ##################################################################

    # ---------------------------------------------------------------- #
    # COST FUNCTION ####################################################
    # ---------------------------------------------------------------- #

    if myopic:
        post_decision_costs = 0

    # Model has learned shadow costs from previous iterations and can
    # use them to determine post decision costs.
    else:
        post_decision_costs = quicksum(
            (env.post_cost(time_step, d, level=agg_level) * x_var[d])
            for d in x_var
        )

    # Cost of current decision
    current_costs = quicksum(
        env.cost_func(d[ACTION], d[ORIGIN], d[DESTINATION]) * x_var[d]
        for d in x_var
    )

    m.setObjective(current_costs + post_decision_costs, GRB.MAXIMIZE)

    # ---------------------------------------------------------------- #
    # CONSTRAINTS ######################################################
    # ---------------------------------------------------------------- #

    # Car flow conservation
    flow_cars = m.addConstrs(
        (
            x_var.sum("*", point, level, "*", "*")
            == len(cars_with_attribute_zero[(point, level)])
            for point, level in cars_with_attribute_zero.keys()
        ),
        "CAR_FLOW",
    )

    # Trip flow conservation
    flow_trips = m.addConstrs(
        (
            x_var.sum(Amod.TRIP_DECISION, "*", "*", o, d)
            <= len(trips_with_attribute_zero[(o, d)])
            for o, d in trips_with_attribute_zero
        ),
        "TRIP_FLOW",
    )

    # Car is obliged to charged if battery reaches minimum level
    if charge:
        recharge = m.addConstrs(
            (
                x_var[(action, pos, level, o, d)]
                == len(cars_with_attribute_zero[(pos, level)])
                for action, pos, level, o, d in x_var
                if level <= env.config.min_battery_level
                and action == Amod.RECHARGE_DECISION
            ),
            "RECHARGE",
        )

    # Optimize
    m.optimize()
    m.write(f"mip/adp{time_step:04}.lp")

    if m.status == GRB.Status.UNBOUNDED:
        print("The model cannot be solved because it is unbounded")

    if m.status == GRB.Status.OPTIMAL:

        # c = time.time()
        # best_decisions2 = extract_duals(flow_cars, cars_with_attribute.keys())
        # decisions = extract_decisions(x_var)
        # a = time.time()
        best_decisions, duals = extract_solution(x_var, flow_cars=flow_cars)

        # Update shadow prices to be used in the next iterations
        if not myopic:
            if agg_level:
                env.averaged_update(time_step, duals)
            else:
                env.update_values_smoothed(time_step, duals)

        reward, serviced, denied = env.realize_decision(
            time_step,
            best_decisions,
            trips_with_attribute_zero,
            cars_with_attribute_zero,
        )
        # print(f"Objective Function - {m.objVal:6.2f} X
        # {reward:6.2f} - Decision reward")

        # Update list of rejected orders
        rejected.extend(denied)

        return reward, serviced, rejected

    if (
        m.status != GRB.Status.INF_OR_UNBD
        and m.status != GRB.Status.INFEASIBLE
    ):
        print("Optimization was stopped with status %d" % m.status)

    if m.status == GRB.Status.INFEASIBLE:
        # do IIS
        print("The model is infeasible; computing IIS")

        # Save model
        m.write("myopic.lp")

        m.computeIIS()

        if m.IISMinimal:
            print("IIS is minimal\n")
        else:
            print("IIS is not minimal\n")
            print("\nThe following constraint(s) cannot be satisfied:")
        for c in m.getConstrs():
            if c.IISConstr:
                print("%s" % c.constrName)



def sortout_trips(trips, neighborhood_level, reachable_points):

    #  Dictionary of #trips per trip attribute,i.e., (o.id, d.id)
    trips_with_attribute_neighborhood = defaultdict(list)
    trips_with_attribute_zero = defaultdict(list)

    # Trips that cannot be picked up
    rejected = list()
    for t in trips:

        if t.o.id_level(neighborhood_level) in reachable_points:

            od_level_neighborhood = (
                t.o.id_level(neighborhood_level),
                t.d.id_level(neighborhood_level),
            )

            od_level_zero = (t.o.id, t.d.id)

            trips_with_attribute_neighborhood[od_level_neighborhood].append(t)
            trips_with_attribute_zero[od_level_zero].append(t)

        # If no vehicle can reach the trip, it is immediately rejected
        else:
            rejected.append(t)
    return trips_with_attribute_neighborhood, trips_with_attribute_zero, rejected


def sortout_fleet(
        env,
        time_step,
        neighborhood_level,
        rebalance_level,
        n_neighbors
    ):
    # ##################################################################
    # SORT CARS ########################################################
    # ##################################################################

    # How many cars per attribute
    cars_with_attribute_neighborhood = defaultdict(list)
    cars_with_attribute_zero = defaultdict(list)

    # Which positions are surrounding each car position
    dict_attribute_neighbors = dict()
    dict_attribute_rebalance = dict()

    # Reachable points
    rechable_points = set()
    for car in env.available:

        # List of cars with the same attribute (pos, battery level)
        a_level = car.attribute_level(neighborhood_level)
        cars_with_attribute_neighborhood[a_level].append(car)
        cars_with_attribute_zero[car.attribute].append(car)

        # Each car can rebalance to its immediate neighbors. This
        # prevents vehicles are busy rebalancing to far away zones.
        if car.attribute not in dict_attribute_rebalance:
            dict_attribute_rebalance[car.point.id] = env.get_zone_neighbors(
                car.point, level=rebalance_level, n_neighbors=n_neighbors
            )

            # dict_attribute_rebalance[car.point.id] = env.get_neighbors(
            #     car.point,
            #     reach=2
            # )

        # Was this position already processed?
        car_pos_id = car.point.id_level(neighborhood_level)
        if car_pos_id not in dict_attribute_neighbors:

            # Get zones around current car regions
            # nearby_zones = env.get_zone_neighbors(
            #     car.point, level=neighborhood_level, n_neighbors=n_neighbors
            # )

            # Get zones around current car regions
            nearby_zones = [car_pos_id]

            # Update set of points cars can reach
            rechable_points.update(nearby_zones)

            dict_attribute_neighbors[car_pos_id] = nearby_zones

    return (
        rechable_points,
        cars_with_attribute_neighborhood,
        dict_attribute_neighbors,
        dict_attribute_rebalance,
        cars_with_attribute_zero,
    )


def adp_grid(env, trips, time_step, charge=True, agg_level=None, myopic=False):
    """Assign trips to available vehicles optimally at the current
    time step.

    Arguments:
        env {Environment} -- AMoD environment
        trips {list} -- List of trips
        time_step {int} -- Current time step

    Keyword Arguments:
        charge {bool} -- Apply the charging constraint (default: {True})
        agg_level {int} -- Attributes are queried according to an
        aggregation level (default: {None})
        myopic {bool} -- If True, does not learn between iterations
            (default: {True})

    Returns:
        float, list, list -- total_contribution, serviced trips,
            rejected trips
    """

    # Starting assignment model
    m = Model("assignment")

    # Disables all logging (file and console)
    m.setParam("OutputFlag", 0)

    # if log_path:
    #     m.Params.LogFile = "{}/region_centers_{}.log".format(
    #         log_path, max_delay
    #     )

    #     m.Params.ResultFile = "{}/region_centers_{}.lp".format(
    #         log_path, max_delay
    #     )

    # ##################################################################
    # SORT CARS ########################################################
    # ##################################################################

    # How many cars per attribute
    cars_with_attribute = defaultdict(list)

    # Which positions are surrounding each car position
    dict_attribute_neighbors = dict()

    # Reachable points
    rechable_points = set()

    for car in env.cars:

        # Check if vehicles finished their tasks
        # Where are the cars? What are they doing at the current step?
        car.update(time_step, time_increment=env.config.time_increment)

        # Discard busy vehicles
        if car.busy:
            continue

        # List of cars with the same attribute (pos, battery level)
        cars_with_attribute[car.attribute].append(car)

        # Was this position already processed?
        car_pos_id = car.point.id
        if car_pos_id not in dict_attribute_neighbors:

            # Get zones around current car regions
            nearby_zones = env.get_neighbors(car.point)

            # Update set of points cars can reach
            rechable_points.update(nearby_zones)

            dict_attribute_neighbors[car_pos_id] = nearby_zones

    # ##################################################################
    # SORT TRIPS #######################################################
    # ##################################################################

    #  Dictionary of #trips per trip attribute,i.e., (o.id, d.id)
    trips_with_attribute = defaultdict(list)

    # Trips that cannot be picked up
    rejected = list()
    for t in trips:

        if t.o.id in rechable_points:
            trips_with_attribute[(t.o.id, t.d.id)].append(t)

        # If no vehicle can reach the trip, it is immediately rejected
        else:
            rejected.append(t)

    # ##################################################################
    # VARIABLES ########################################################
    # ##################################################################

    # Enumerate list of decision variables
    # time1 = time.time()
    all_decisions = get_decision_tuples(
        cars_with_attribute.keys(),
        dict_attribute_neighbors,
        trips_with_attribute,
    )

    # print("\n## Time to get decisions:", time.time() - time1)

    # Adding variables
    x_var = m.addVars(all_decisions, name="x")

    # ##################################################################
    # MODEL ############################################################
    # ##################################################################

    # ---------------------------------------------------------------- #
    # COST FUNCTION ####################################################
    # ---------------------------------------------------------------- #

    if myopic:
        post_decision_costs = 0

    # Model has learned shadow costs from previous iterations and can
    # use them to determine post decision costs.
    else:
        post_decision_costs = quicksum(
            (env.post_cost(time_step, d, level=agg_level) * x_var[d])
            for d in x_var
        )

    # Cost of current decision
    current_costs = quicksum(
        env.cost_func(d[ACTION], d[ORIGIN], d[DESTINATION]) * x_var[d]
        for d in x_var
    )

    m.setObjective(current_costs + post_decision_costs, GRB.MAXIMIZE)

    # ---------------------------------------------------------------- #
    # CONSTRAINTS ######################################################
    # ---------------------------------------------------------------- #

    # Car flow conservation
    flow_cars = m.addConstrs(
        (
            x_var.sum("*", point, level, "*", "*")
            == len(cars_with_attribute[(point, level)])
            for point, level in cars_with_attribute.keys()
        ),
        "CAR_FLOW",
    )

    # Trip flow conservation
    flow_trips = m.addConstrs(
        (
            x_var.sum(Amod.TRIP_STAY_DECISION, "*", "*", o, d)
            <= len(trips_with_attribute[(o, d)])
            for o, d in trips_with_attribute.keys()
        ),
        "TRIP_FLOW",
    )

    # Car is obliged to charged if battery reaches minimum level
    if charge:
        recharge = m.addConstrs(
            (
                x_var[(action, pos, level, o, d)]
                == len(cars_with_attribute[(pos, level)])
                for action, pos, level, o, d in x_var
                if level <= env.config.min_battery_level
                and action == Amod.RECHARGE_REBALANCE_DECISION
                and o == d
            ),
            "RECHARGE",
        )

    # Optimize
    m.optimize()

    if m.status == GRB.Status.UNBOUNDED:
        print("The model cannot be solved because it is unbounded")

    if m.status == GRB.Status.OPTIMAL:

        # c = time.time()
        # best_decisions2 = extract_duals(flow_cars, cars_with_attribute.keys())
        # decisions = extract_decisions(x_var)
        # a = time.time()
        best_decisions, duals = extract_solution(x_var, flow_cars=flow_cars)

        # Update shadow prices to be used in the next iterations
        if not myopic:
            if agg_level:
                env.averaged_update(time_step, duals)
            else:
                env.update_values_smoothed(time_step, duals)

        reward, serviced, denied = env.realize_decision(
            time_step,
            best_decisions,
            trips_with_attribute,
            cars_with_attribute,
        )
        # print(f"Objective Function - {m.objVal:6.2f} X
        # {reward:6.2f} - Decision reward")

        # Update list of rejected orders
        rejected.extend(denied)

        return reward, serviced, rejected

    if (
        m.status != GRB.Status.INF_OR_UNBD
        and m.status != GRB.Status.INFEASIBLE
    ):
        print("Optimization was stopped with status %d" % m.status)

    if m.status == GRB.Status.INFEASIBLE:
        # do IIS
        print("The model is infeasible; computing IIS")

        # Save model
        m.write("myopic.lp")

        m.computeIIS()

        if m.IISMinimal:
            print("IIS is minimal\n")
        else:
            print("IIS is not minimal\n")
            print("\nThe following constraint(s) cannot be satisfied:")
        for c in m.getConstrs():
            if c.IISConstr:
                print("%s" % c.constrName)


def fcfs(env, trips, time_step, charge=True):
    """First Come First Serve

    Arguments:
        env {Environment} -- AMoD environment
        trips {list} -- List of trips
        time_step {int} -- Current time step

    Returns:
        float, list, list -- total_contribution, serviced trips,
            rejected trips

    """

    # trips_zone_dict = get_trips_zone_dict(trips)
    # print("Trip zone dict:")
    # print(trips_zone_dict)

    total_contribution = 0

    # Available cars to service passengers
    available_cars = set()

    dict_zone_trips = defaultdict(set)

    for tt in trips:
        dict_zone_trips[tt.o.id].add(tt)

    serviced = set()
    rejected = set()

    for car in env.cars:

        # Check if vehicles finished their tasks
        # print("Updating status all vehicles")
        car.update(time_step, time_increment=env.config.time_increment)

        # Only available vehicles can be recharged or assigned
        if car.busy:
            # print(f'car {car.id} is busy until {car.arrival_time}')
            continue

        # print(f'car {car.id} is free')

        # Recharge vehicles about to run out of power
        if charge and car.need_recharge(env.config.recharge_threshold):

            # print(f' - it needs RECHARGE')
            # Recharge vehicle
            # cost_recharging = env.full_recharge(car)
            cost_recharging = env.recharge(
                car, env.config.recharge_time_single_level
            )

            # Subtract cost of recharging
            total_contribution -= cost_recharging

        else:
            # Get trips car can pickup

            available_cars.add(car)

            # Get zones around current car regions
            nearby_zones = nw.get_neighbor_zones(
                car.point, env.config.pickup_zone_range, env.zones
            )

            best_trip = None
            best_reward = 0
            duration_min = None
            total_distance = None
            z_best = None

            for z in nearby_zones:
                for trip in dict_zone_trips[z]:
                    d, t, r = env.pickup(trip, car)

                    if best_reward < r:
                        if not car.has_power(t):
                            continue

                        duration_min, total_distance, best_reward = d, t, r
                        best_trip = trip
                        z_best = z

            if best_trip is not None:

                # Update car data
                car.update_trip(
                    duration_min, total_distance, best_reward, best_trip
                )

                serviced.add(best_trip)

                total_contribution += best_reward

                dict_zone_trips[z_best].remove(best_trip)

    for rejected_trip_set in dict_zone_trips.values():
        rejected.update(rejected_trip_set)

    return total_contribution, serviced, rejected


def myopic(env, trips, time_step, charge=True):

    # Starting assignment model
    m = Model("assignment")

    # Disables all logging (file and console)
    m.setParam("OutputFlag", 0)

    # ##################################################################
    # SORT CARS ########################################################
    # ##################################################################

    # How many cars per attribute
    cars_with_attribute = defaultdict(list)

    # Which positions are surrounding each car position
    dict_attribute_neighbors = dict()

    # Reachable points
    rechable_points = set()

    for car in env.cars:

        # Check if vehicles finished their tasks
        # Where are the cars? What are they doing at the current step?
        car.update(time_step, time_increment=env.config.time_increment)

        # Discard busy vehicles
        if car.busy:
            continue

        # List of cars with the same attribute (pos, battery level)
        cars_with_attribute[car.attribute].append(car)

        # Was this position already processed?
        car_pos_id = car.point.id
        if car_pos_id not in dict_attribute_neighbors:

            # Get zones around current car regions
            nearby_zones = env.get_neighbors(car.point)

            # Update set of points cars can reach
            rechable_points.update(nearby_zones)

            dict_attribute_neighbors[car_pos_id] = nearby_zones

    # ##################################################################
    # SORT TRIPS #######################################################
    # ##################################################################

    #  Dictionary of #trips per trip attribute,i.e., (o.id, d.id)
    trips_with_attribute = defaultdict(list)

    # Trips that cannot be picked up
    rejected = list()
    for t in trips:

        if t.o.id in rechable_points:
            trips_with_attribute[(t.o.id, t.d.id)].append(t)

        # If no vehicle can reach the trip, it is immediately rejected
        else:
            rejected.append(t)

    # ##################################################################
    # VARIABLES ########################################################
    # ##################################################################

    # Enumerate list of decision variables
    # time1 = time.time()
    all_decisions = get_decision_tuples(
        cars_with_attribute.keys(),
        dict_attribute_neighbors,
        trips_with_attribute,
    )

    # print("\n## Time to get decisions:", time.time() - time1)

    # Adding variables
    x_var = m.addVars(all_decisions, name="x")

    # ##################################################################
    # MODEL ############################################################
    # ##################################################################

    # if log_path:
    #         m.Params.LogFile = "{}/region_centers_{}.log".format(
    #             log_path, max_delay
    #         )

    #         m.Params.ResultFile = "{}/region_centers_{}.lp".format(
    #             log_path, max_delay
    #         )

    # Cost of current decision
    cost = quicksum(
        env.cost_func(d[ACTION], d[ORIGIN], d[DESTINATION]) * x_var[d]
        for d in x_var
    )

    m.setObjective(cost, GRB.MAXIMIZE)

    # Car flow conservation
    flow_cars = m.addConstrs(
        (
            x_var.sum("*", point, level, "*", "*")
            == len(cars_with_attribute[(point, level)])
            for point, level in cars_with_attribute.keys()
        ),
        "CAR_FLOW",
    )

    # Trip flow conservation
    flow_trips = m.addConstrs(
        (
            x_var.sum(Amod.TRIP_STAY_DECISION, "*", "*", o, d)
            <= len(trips_with_attribute[(o, d)])
            for o, d in trips_with_attribute.keys()
        ),
        "TRIP_FLOW",
    )

    if charge:
        # Battery
        recharge = m.addConstrs(
            (
                x_var[(action, pos, level, o, d)]
                == len(cars_with_attribute[(pos, level)])
                for action, pos, level, o, d in x_var
                if level <= env.config.min_battery_level
                and action == Amod.RECHARGE_REBALANCE_DECISION
                and o == d
            ),
            "RECHARGE",
        )

    # Optimize
    m.optimize()

    if m.status == GRB.Status.UNBOUNDED:
        print("The model cannot be solved because it is unbounded")

    if m.status == GRB.Status.OPTIMAL:

        best_decisions, _ = extract_solution(x_var)

        reward, serviced, denied = env.realize_decision(
            time_step,
            best_decisions,
            trips_with_attribute,
            cars_with_attribute,
        )
        # print(f"Objective Function - {m.objVal:6.2f} X
        # {reward:6.2f} - Decision reward")

        # Update list of rejected orders
        rejected.extend(denied)

        return reward, serviced, rejected

    if (
        m.status != GRB.Status.INF_OR_UNBD
        and m.status != GRB.Status.INFEASIBLE
    ):
        print("Optimization was stopped with status %d" % m.status)

    if m.status == GRB.Status.INFEASIBLE:
        # do IIS
        print("The model is infeasible; computing IIS")

        # Save model
        m.write("myopic.lp")

        m.computeIIS()

        if m.IISMinimal:
            print("IIS is minimal\n")
        else:
            print("IIS is not minimal\n")
            print("\nThe following constraint(s) cannot be satisfied:")
        for c in m.getConstrs():
            if c.IISConstr:
                print("%s" % c.constrName)


def adp_low_resolution(
    env,
    trips,
    time_step,
    charge=True,
    agg_level=None,
    myopic=False,
    neighborhood_level=0,
    n_neighbors=4,
):
    """Assign trips to available vehicles optimally at the current
        time step.

    Parameters
    ----------
    env : Environment
        AMoD environment
    trips : list
        List of trips
    time_step : int
        Time step after receiving trips

    charge : bool, optional
        Apply the charging constraint, by default True
    agg_level : [type], optional
        Attributes are queried according to an aggregation level, 
        by default None
    myopic : bool, optional
        If True, does not learn between iterations, by default False
    neighborhood_level : int, optional
        How large are region centers (
            e.g., 1 = reachable in 1min,
                  2 = reachable in 2min
            ), by default 1
    n_neighbors : int, optional
        Max. neighbors of region centers, by default 4

    Returns
    -------
    float, list, list
        total contribution, serviced trips, rejected trips
    """

    # Starting assignment model
    m = Model("assignment")

    # Disables all logging (file and console)
    m.setParam("OutputFlag", 0)

    # if log_path:
    #     m.Params.LogFile = "{}/region_centers_{}.log".format(
    #         log_path, max_delay
    #     )

    #     m.Params.ResultFile = "{}/region_centers_{}.lp".format(
    #         log_path, max_delay
    #     )

    # ##################################################################
    # SORT CARS ########################################################
    # ##################################################################

    # How many cars per attribute
    cars_with_attribute = defaultdict(list)

    # Which positions are surrounding each car position
    dict_attribute_neighbors = dict()

    # Reachable points
    rechable_points = set()

    for car in env.cars:

        # Check if vehicles finished their tasks
        # Where are the cars? What are they doing at the current step?
        # t ------- t+1 ------- t+2 ------- t+3 ------- t+4 ------- t+5
        # ---trips-------trips-------trips-------trips-------trips-----
        car.update(time_step, time_increment=env.config.time_increment)

        # Discard busy vehicles
        if car.busy:
            continue

        # List of cars with the same attribute (pos, battery level)
        a_level = car.attribute_level(neighborhood_level)
        cars_with_attribute[a_level].append(car)

        # Was this position already processed?
        car_pos_id = car.point.id_level(neighborhood_level)
        if car_pos_id not in dict_attribute_neighbors:

            # Get zones around current car regions
            nearby_zones = env.get_neighbors(
                car.point, level=neighborhood_level, n_neighbors=n_neighbors
            )

            # Update set of points cars can reach
            rechable_points.update(nearby_zones)

            dict_attribute_neighbors[car_pos_id] = nearby_zones

    # ##################################################################
    # SORT TRIPS #######################################################
    # ##################################################################

    #  Dictionary of #trips per trip attribute,i.e., (o.id, d.id)
    trips_with_attribute = defaultdict(list)

    # Trips that cannot be picked up
    rejected = list()
    for t in trips:

        if t.o.id_level(neighborhood_level) in rechable_points:

            od_level_tuple = (
                t.o.id_level(neighborhood_level),
                t.d.id_level(neighborhood_level),
            )
            trips_with_attribute[od_level_tuple].append(t)

        # If no vehicle can reach the trip, it is immediately rejected
        else:
            rejected.append(t)

    # ##################################################################
    # VARIABLES ########################################################
    # ##################################################################

    # Enumerate list of decision variables
    # time1 = time.time()
    # all_decisions = get_decision_tuples(
    #     cars_with_attribute.keys(),
    #     dict_attribute_neighbors,
    #     trips_with_attribute,
    # )

    all_decisions = get_decision_set(
        cars_with_attribute,
        dict_attribute_neighbors,
        trips_with_attribute,
        max_battery_level=env.config.max_battery_level,
    )

    # print("\n## Time to get decisions:", time.time() - time1)

    # Adding variables
    x_var = m.addVars(all_decisions, name="x")

    # ##################################################################
    # MODEL ############################################################
    # ##################################################################

    # ---------------------------------------------------------------- #
    # COST FUNCTION ####################################################
    # ---------------------------------------------------------------- #

    if myopic:
        post_decision_costs = 0

    # Model has learned shadow costs from previous iterations and can
    # use them to determine post decision costs.
    else:
        post_decision_costs = quicksum(
            (env.post_cost(time_step, d, level=agg_level) * x_var[d])
            for d in x_var
        )

    # Cost of current decision
    current_costs = quicksum(
        env.cost_func(d[ACTION], d[ORIGIN], d[DESTINATION]) * x_var[d]
        for d in x_var
    )

    m.setObjective(current_costs + post_decision_costs, GRB.MAXIMIZE)

    # ---------------------------------------------------------------- #
    # CONSTRAINTS ######################################################
    # ---------------------------------------------------------------- #

    # Car flow conservation
    flow_cars = m.addConstrs(
        (
            x_var.sum("*", point, level, "*", "*")
            == len(cars_with_attribute[(point, level)])
            for point, level in cars_with_attribute.keys()
        ),
        "CAR_FLOW",
    )

    # Trip flow conservation
    flow_trips = m.addConstrs(
        (
            x_var.sum(Amod.TRIP_STAY_DECISION, "*", "*", o, d)
            <= len(trips_with_attribute[(o, d)])
            for o, d, trip_list in trips_with_attribute
        ),
        "TRIP_FLOW",
    )

    # Car is obliged to charged if battery reaches minimum level
    if charge:
        recharge = m.addConstrs(
            (
                x_var[(action, pos, level, o, d)]
                == len(cars_with_attribute[(pos, level)])
                for action, pos, level, o, d in x_var
                if level <= env.config.min_battery_level
                and action == Amod.RECHARGE_REBALANCE_DECISION
                and o == d
            ),
            "RECHARGE",
        )

    # Optimize
    m.optimize()

    if m.status == GRB.Status.UNBOUNDED:
        print("The model cannot be solved because it is unbounded")

    if m.status == GRB.Status.OPTIMAL:

        # c = time.time()
        # best_decisions2 = extract_duals(flow_cars, cars_with_attribute.keys())
        # decisions = extract_decisions(x_var)
        # a = time.time()
        best_decisions, duals = extract_solution(x_var, flow_cars=flow_cars)

        # Update shadow prices to be used in the next iterations
        if not myopic:
            if agg_level:
                env.averaged_update(time_step, duals)
            else:
                env.update_values_smoothed(time_step, duals)

        reward, serviced, denied = env.realize_decision(
            time_step,
            best_decisions,
            trips_with_attribute,
            cars_with_attribute,
        )
        # print(f"Objective Function - {m.objVal:6.2f} X
        # {reward:6.2f} - Decision reward")

        # Update list of rejected orders
        rejected.extend(denied)

        return reward, serviced, rejected

    if (
        m.status != GRB.Status.INF_OR_UNBD
        and m.status != GRB.Status.INFEASIBLE
    ):
        print("Optimization was stopped with status %d" % m.status)

    if m.status == GRB.Status.INFEASIBLE:
        # do IIS
        print("The model is infeasible; computing IIS")

        # Save model
        m.write("myopic.lp")

        m.computeIIS()

        if m.IISMinimal:
            print("IIS is minimal\n")
        else:
            print("IIS is not minimal\n")
            print("\nThe following constraint(s) cannot be satisfied:")
        for c in m.getConstrs():
            if c.IISConstr:
                print("%s" % c.constrName)


def get_decision_set_classed(
        cars,
        hired_cars,
        level_id_cars_dict,
        level_id_trips_dict,
        rebalance_targets_dict,
        max_battery_level=20,
    ):

    def stay_decision(car):
        return (
            (Amod.STAY_DECISION,)
            + (car.point.id, car.battery_level)
            + (car.point.id,)
            + (car.point.id,)
            + (car.type,)
        )

    def end_contract_decision(car):
        return (
            (Amod.END_CONTRACT_DECISION,)
            + (car.point.id, car.battery_level)
            + (car.point.id,)
            + (car.start_end_point.id,)
            + (car.type,)
        )

    def recharge_decision(car):
        return (
            (Amod.RECHARGE_DECISION,)
            + (car.point.id, car.battery_level)
            + (car.point.id,)
            + (car.point.id,)
            + (car.type,)
        )

    def rebalance_decision(car, neighbor, hire=False):
        return (
            ((Amod.HIRE_DECISION if hire else Amod.REBALANCE_DECISION),)
            + (car.point.id, car.battery_level)
            + (car.point.id,)
            + (neighbor,)
            + (car.type,)
        )

    def rebalance_decisions(car, targets, hire=False):
        rebalance_decisions = set()
        for t in targets:
            rebalance_decisions.add(rebalance_decision(car, t, hire=hire))
        return rebalance_decisions

    def hire_decisions(car, targets, hire=False):
        hire_decisions = set()
        for t in targets:
            hire_decisions.add(rebalance_decision(car, t, hire=hire))
        return hire_decisions

    def trip_decision(car, trip, hire=True):
        return (
            (Amod.TRIP_DECISION,)
            + (car.point.id, car.battery_level)
            + (trip.o.id,)
            + (trip.d.id,)
            + (car.type,)
        )

    decisions = defaultdict(set)
    decision_class = defaultdict(list)

    for car in cars:
        # Stay ####################################################### #
        decisions[car.type].add(stay_decision(car))

        # Recharge ################################################### #
        if car.battery_level < max_battery_level:
            decisions[car.type].add(recharge_decision(car))

        # Rebalancing ################################################ #
        rebalance_targets = rebalance_targets_dict[car.type][car.point.id]
        decisions[car.type].update(rebalance_decisions(car, rebalance_targets))

    for car in hired_cars:

        # if car.started_contract:

        # Rebalancing ############################################ #
        rebalance_targets = rebalance_targets_dict[car.type][car.point.id]
        decisions[car.type].update(
            rebalance_decisions(car, rebalance_targets)
        )

        # Stay ################################################### #
        decisions[car.type].add(stay_decision(car))

        # Recharge ############################################### #
        if car.battery_level < max_battery_level:
            decisions[car.type].add(recharge_decision(car))

        # else:

        #     # Hire new vehicles
        #     decisions[car.type].update(
        #         hire_decisions(car, rebalance_targets, hire=True)
        #     )

        #     # End contract
        #     decisions[car.type].add(end_contract_decision(car))

    # ################################################################ #
    # TRIP X CARS #################################################### #
    # ################################################################ #

    # Trips sorted out by level and level and level_id
    for (level, level_id), trip_list in level_id_trips_dict.items():

        # Matching trips to cars
        for trip in trip_list:
            
            # A trip is matched to a car a single time. If the car was
            # already matched before, a decision was already created
            # for it.
            set_matched = set()

            for car_type in level_id_cars_dict:

                # Check if there are cars to match to trips. Hired
                # vehicles always match trips.
                if (level, level_id) in level_id_cars_dict[car_type]:

                    car_list = level_id_cars_dict[car_type][(level, level_id)]

                    for car in car_list:
                        
                        # If car not previosly matched (other levels)
                        if car not in set_matched:
                            
                            # A car is matched to a trip a single time
                            set_matched.add(car)

                            # Setup decisions
                            d = trip_decision(car, trip)
                            
                            # ---------------------------------------- #
                            # DECISIONS ASSOCIATED TO EACH SQ CLASS ## #
                            # ---------------------------------------- #
                            
                            # There might be repeated decisions
                            # associated to the same class since
                            # several trips can depart from the same
                            # place.
                            if level == trip.sq1_level:
                                decision_class[trip.sq_class].append(d)

                            decisions[car.type].add(d)
                        else:
                            pass

                else:
                    # Car and Trip ids can't match
                    pass
                    # print(
                    #     f'{(level, level_id)} not in level_id_cars_dict.'
                    #     f'\ntrips={trip_list}.'
                    # )

    return decisions, decision_class


def adp_network_hired(
        env,
        trips,
        time_step,
        sq_guarantee=True,
        charge=True,
        agg_level=None,
        myopic=False,
        log_path=None,
    ):
    """Assign trips to available vehicles optimally at the current
        time step.

    Parameters
    ----------
    env : Environment
        AMoD environment
    trips : list
        List of trips
    time_step : int
        Time step after receiving trips
    charge : bool, optional
        Apply the charging constraint, by default True
    agg_level : [type], optional
        Attributes are queried according to an aggregation level, 
        by default None
    myopic : bool, optional
        If True, does not learn between iterations, by default False
    neighborhood_level : int, optional
        How large are region centers (
            e.g., 1 = reachable in 1min,
                  2 = reachable in 2min
            ), by default 1
    n_neighbors : int, optional
        Max. neighbors of region centers, by default 4

    Returns
    -------
    float, list, list
        total contribution, serviced trips, rejected trips
    """

    # Starting assignment model
    m = Model("assignment")

    # Log steps of current episode
    if log_path:
        m.setParam("LogToConsole", 0)
        m.Params.LogFile = f"{log_path}/log/mip_{time_step:04}.log"
        m.Params.ResultFile = f"{log_path}/lp/mip_{time_step:04}.lp"

    else:
        # Disables all logging (file and console)
        m.setParam("OutputFlag", 0)

    # ##################################################################
    # SORT CARS ########################################################
    # ##################################################################

    (
        level_id_cars_dict, rebalance_targets_dict, attribute_cars_dict,
    ) = sortout_fleet2(
        env,
        time_step,
    )

    # ##################################################################
    # SORT TRIPS #######################################################
    # ##################################################################

    #  Dictionary of #trips per trip attribute,i.e., (o.id, d.id)
    (
        level_id_trips_dict, attribute_trips_dict, class_count_dict
    ) = sortout_trips2(trips)

    # ##################################################################
    # VARIABLES ########################################################
    # ##################################################################

    decision_cars, decision_class = get_decision_set_classed(
        env.available,
        env.available_hired,
        level_id_cars_dict,
        level_id_trips_dict,
        rebalance_targets_dict,
        max_battery_level=env.config.battery_levels,
    )

    # Adding variables
    all_decisions = list(itertools.chain.from_iterable(decision_cars.values()))

    x_var = m.addVars(tuplelist(all_decisions), name="x")

    # ##################################################################
    # MODEL ############################################################
    # ##################################################################

    # ---------------------------------------------------------------- #
    # COST FUNCTION ####################################################
    # ---------------------------------------------------------------- #

    if myopic:
        post_decision_costs = 0

    # Model has learned shadow costs from previous iterations and can
    # use them to determine post decision costs.
    else:
        post_decision_costs = quicksum(
            (env.post_cost(time_step, d, level=agg_level) * x_var[d])
            for d in x_var
        )

    # Cost of current decision
    current_costs = quicksum(
        env.cost_func(
            d[CAR_TYPE],
            d[ACTION],
            d[ORIGIN],
            d[DESTINATION]
        ) * x_var[d]
        for d in x_var
    )

    m.setObjective(current_costs + post_decision_costs, GRB.MAXIMIZE)

    # ---------------------------------------------------------------- #
    # CONSTRAINTS ######################################################
    # ---------------------------------------------------------------- #

    # Join attributes of both fleets
    attribute_cars = dict()
    for car_type in attribute_cars_dict:
        attribute_cars.update(attribute_cars_dict[car_type])

    # Car flow conservation
    flow_cars = m.addConstrs(
        (
            x_var.sum("*", point, level, "*", "*", "*")
            == len(attribute_cars[(point, level)])
            for point, level in attribute_cars.keys()
        ),
        "CAR_FLOW",
    )

    # Trip flow conservation
    flow_trips = m.addConstrs(
        (
            x_var.sum(Amod.TRIP_DECISION, "*", "*", o, d, "*")
            <= len(attribute_trips_dict[(o, d)])
            for o, d in attribute_trips_dict
        ),
        "TRIP_FLOW",
    )

    # ################################################################ #
    # SERVICE QUALITY CONSTRAINTS #################################### #
    # ################################################################ #

    if sq_guarantee:

        # Minimum service rate for users of each class
        for sq_class, s_rate in ClassedTrip.sq_classes.items():

            # List of decisions associated to a user class
            var_list_class = []

            # Adding decisions to user class list
            for single_decision in set(decision_class[sq_class]):
                var_list_class.append(x_var[single_decision])

            # Add constraints
            if len(var_list_class) > 0:
                m.addConstr(
                    quicksum(var_list_class) >= np.ceil(
                        s_rate * class_count_dict[sq_class]
                    ),
                    f"TRIP_FLOW_CLASS_{sq_class}"
                )

    # Car is obliged to charged if battery reaches minimum level
    if charge:
        recharge = m.addConstrs(
            (
                x_var[(action, pos, level, o, d, car_type)]
                == len(attribute_cars[(pos, level)])
                for action, pos, level, o, d, car_type in x_var
                if level <= env.config.min_battery_level
                and action == Amod.RECHARGE_DECISION
            ),
            "RECHARGE",
        )

    # Optimize
    m.optimize()
    # m.write(f"adp{time_step:04}.lp")

    if m.status == GRB.Status.UNBOUNDED:
        print("The model cannot be solved because it is unbounded")

    if m.status == GRB.Status.OPTIMAL:

        # c = time.time()
        # best_decisions2 = extract_duals(flow_cars, cars_with_attribute.keys())
        # decisions = extract_decisions(x_var)
        # a = time.time()
        best_decisions, duals = extract_solution(x_var, flow_cars=flow_cars)

        # Update shadow prices to be used in the next iterations
        if not myopic:
            if agg_level:
                env.averaged_update(time_step, duals)
            else:
                env.update_values_smoothed(time_step, duals)

        reward, serviced, denied = env.realize_decision(
            time_step,
            best_decisions,
            attribute_trips_dict,
            attribute_cars_dict,
        )
        # print(f"Objective Function - {m.objVal:6.2f} X
        # {reward:6.2f} - Decision reward")

        rejected = []

        # Update list of rejected orders
        rejected.extend(denied)

        return reward, serviced, rejected

    if (
        m.status != GRB.Status.INF_OR_UNBD
        and m.status != GRB.Status.INFEASIBLE
    ):
        print("Optimization was stopped with status %d" % m.status)

    if m.status == GRB.Status.INFEASIBLE:
        # do IIS
        print("The model is infeasible; computing IIS")

        # Save model
        m.write("myopic.lp")

        m.computeIIS()

        if m.IISMinimal:
            print("IIS is minimal\n")
        else:
            print("IIS is not minimal\n")
            print("\nThe following constraint(s) cannot be satisfied:")
        for c in m.getConstrs():
            if c.IISConstr:
                print("%s" % c.constrName)


def sortout_fleet2(env, time_step):

    rebalance_level = env.config.rebalance_level
    n_neighbors = env.config.n_neighbors


    # ##################################################################
    # SORT CARS ########################################################
    # ##################################################################

    # How many cars per attribute
    type_attribute_cars_dict = defaultdict(lambda:defaultdict(list))

    # Which positions are surrounding each car position
    dict_attribute_rebalance = defaultdict(lambda:defaultdict(list))

    # Which positions are surrounding each car position
    dict_level_id = defaultdict(lambda:defaultdict(list))

    # Cars can explore levels corresponding to the largest region center
    # considered by users.
    class_levels = list(range(1, env.config.neighborhood_level+1))

    for car in env.available + env.available_hired:

        # List of cars with the same attribute (pos, battery level)
        type_attribute_cars_dict[car.type][car.attribute].append(car)

        # Each car can rebalance to its immediate neighbors. This
        # prevents vehicles are busy rebalancing to far away zones.
        rebalance_targets = env.get_zone_neighbors(
            car.point,
            level=rebalance_level,
            n_neighbors=n_neighbors
        )

        dict_attribute_rebalance[car.type][car.point.id] = rebalance_targets

        for level in class_levels:
            id_level = car.point.id_level(level)
            dict_level_id[car.type][(level, id_level)].append(car)

    return (
        dict(dict_level_id),
        dict_attribute_rebalance,
        dict(type_attribute_cars_dict),
    )


def sortout_trips2(trips):

    trips_with_attribute_zero = defaultdict(list)
    trips_level_id = defaultdict(list)
    class_count_dict = defaultdict(int)

    for t in trips:
        # Adding levels BEFORE sq level 1 (preferred)
        for level in range(1, t.sq1_level):
            trips_level_id[(level, t.o.id_level(level))].append(t)
        
        trips_level_id[(t.sq1_level, t.id_sq1_level)].append(t)
        trips_level_id[(t.sq2_level, t.id_sq2_level)].append(t)

        class_count_dict[t.sq_class] += 1

        od_level_zero = (t.o.id, t.d.id)
        trips_with_attribute_zero[od_level_zero].append(t)

    return trips_level_id, trips_with_attribute_zero, class_count_dict


def extract_solution(var_list, flow_cars=None):

    # list of decision tuples (action, point, level, o, d)
    decisions = list()

    # Dual values for car attributes (point, level)
    duals = dict()

    # Loop (decision tuple, var) pairs
    for decision, var in var_list.items():

        if var.x > 0.9:
            decisions.append(decision + (round(var.x),))

        if flow_cars:
            # FIll out duals
            car_attribute = (decision[POSITION], decision[BATTERY])

            if car_attribute not in duals:
                # pi = The constraint dual value in the current solution
                # (also known as the shadow price).
                c = flow_cars[car_attribute].pi
                if c > 0:
                    duals[car_attribute] = c

    return decisions, duals
