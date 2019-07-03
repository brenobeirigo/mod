from collections import defaultdict
import mod.env.network as nw
from gurobipy import tuplelist, GRB, Model, quicksum
from pprint import pprint
from mod.env.amod.Amod import Amod
from mod.env.trip import ClassedTrip
from mod.env.car import Car, HiredCar
import time
import numpy as np
import itertools
import mod.env.decision_utils as du
import os
import sys

# Decisions are tuples following the format
# (ACTION, POSITION, BATTERY, ORIGIN, DESTINATION)

# Labels for decision tuples
ACTION = 0
POSITION = 1
BATTERY = 2
ORIGIN = 3
DESTINATION = 4
CAR_TYPE = 5
CONTRACT_DURATION = 6
SQ_CLASS = 7
N_DECISIONS = 8


AVERAGED_UPDATE = "averaged_update"
WEIGHTED_UPDATE = "weighted_update"

# #################################################################### #
# Manipulate solutions ############################################### #
# #################################################################### #


def extract_duals_relaxed(m, flow_cars_dict):
    """[summary]

    Parameters
    ----------
    m : [type]
        [description]
    flow_cars_dict : [type]
        [description]

    Returns
    -------
    dict(dict())
        Dual value for each car type and attribute (point, battery)
    """
    duals = dict()

    try:
        # Relax model (LP) to get duals
        fixed = m.fixed()
        fixed.Params.presolve = 0
        fixed.optimize()

        # if fixed.status != GRB.Status.OPTIMAL:
        #     print("Error: fixed model isn't optimal")

        # diff = m.objVal - fixed.objVal
        # a = fixed.getConstrs()

        for car_type, flow_cars in flow_cars_dict.items():
            # Shadow associated to all car types
            duals[car_type] = dict()

            for pos, battery, contract_duration, car_type in flow_cars:

                try:
                    constr = fixed.getConstrByName(
                        f"CAR_FLOW_{car_type}[{pos},{battery},{contract_duration},{car_type}]"
                    )

                    # pi = The constraint dual value in the current solution
                    shadow_price = constr.pi

                    # print(f'The dual value of {constr.constrName} : {shadow_price}')
                except:
                    shadow_price = 0

                duals[car_type][(pos, battery, contract_duration, car_type)] = shadow_price

    except:
        print("Can't create relaxed model.")

    return duals


def extract_duals(flow_cars):
    duals = dict()
    for pos, battery in flow_cars:

        try:
            constr = flow_cars[pos, battery]
            # pi = The constraint dual value in the current solution
            shadow_price = constr.pi
            print(f"The dual value of {constr.constrName} : {shadow_price}")
        except:
            shadow_price = 0

        duals[(pos, battery)] = shadow_price

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


def extract_decisions(var_list):

    # list of decision tuples (action, point, level, o, d)
    decisions = list()

    # Dual values for car attributes (point, level)
    duals = dict()

    # Loop (decision tuple, var) pairs
    for decision, var in var_list.items():

        if var.x > 0.9:
            decisions.append(decision + (round(var.x),))

    return decisions


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
    value_function_update=AVERAGED_UPDATE,
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

    # ################################################################ #
    # SORT CARS ###################################################### #
    # ################################################################ #

    (
        reachable_points,
        cars_with_attribute_neighborhood,
        dict_attribute_neighbors,
        dict_attribute_rebalance,
        cars_with_attribute_zero,
    ) = sortout_fleet(
        env,
        time_step,
        env.config.neighborhood_level,
        env.config.rebalance_level,
        env.config.n_neighbors,
    )

    # ##################################################################
    # SORT TRIPS #######################################################
    # ##################################################################

    #  Dictionary of #trips per trip attribute,i.e., (o.id, d.id)
    (
        trips_with_attribute_neighborhood,
        trips_with_attribute_zero,
        rejected,
    ) = sortout_trips(trips, neighborhood_level, reachable_points)

    # ##################################################################
    # VARIABLES ########################################################
    # ##################################################################

    decisions = du.get_decision_set(
        cars_with_attribute_neighborhood,
        dict_attribute_neighbors,
        trips_with_attribute_neighborhood,
        dict_attribute_rebalance,
        max_battery_level=env.config.battery_levels,
    )

    # Adding variables
    x_var = m.addVars(tuplelist(decisions), name="x", vtype=GRB.INTEGER, lb=0)

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
            x_var.sum(du.TRIP_DECISION, "*", "*", o, d)
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
                and action == du.RECHARGE_DECISION
            ),
            "RECHARGE",
        )

    # Optimize
    m.optimize()
    m.write(f"mip/adp{time_step:04}.lp")

    if m.status == GRB.Status.UNBOUNDED:
        print("The model cannot be solved because it is unbounded")

    if m.status == GRB.Status.OPTIMAL:

        best_decisions = extract_decisions(x_var)

        # Update shadow prices to be used in the next iterations
        if not myopic:
            duals = extract_duals_relaxed(m, flow_cars)

            # Are there any shadow prices to update?
            if duals:
                if value_function_update == AVERAGED_UPDATE:
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
    return (
        trips_with_attribute_neighborhood,
        trips_with_attribute_zero,
        rejected,
    )


def sortout_fleet(
    env,
    time_step,
    neighborhood_level,
    rebalance_level,
    n_neighbors,
    rebalance_multilevel=False,
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
                car.point,
                level=rebalance_level,
                n_neighbors=n_neighbors,
                multi_level=rebalance_multilevel,
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


def adp_grid(
    env,
    trips,
    time_step,
    charge=True,
    agg_level=None,
    myopic=False,
    value_function_update=AVERAGED_UPDATE,
):
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
    all_decisions = du.get_decision_tuples(
        cars_with_attribute.keys(),
        dict_attribute_neighbors,
        trips_with_attribute,
    )

    # print("\n## Time to get decisions:", time.time() - time1)

    # Adding variables
    x_var = m.addVars(all_decisions, name="x", vtype=GRB.INTEGER, lb=0)

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
            x_var.sum(du.TRIP_DECISION, "*", "*", o, d)
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
                and action == du.RECHARGE_DECISION
                and o == d
            ),
            "RECHARGE",
        )

    # Optimize
    m.optimize()

    if m.status == GRB.Status.UNBOUNDED:
        print("The model cannot be solved because it is unbounded")

    if m.status == GRB.Status.OPTIMAL:

        best_decisions = extract_decisions(x_var)

        # Update shadow prices to be used in the next iterations
        if not myopic:
            duals = extract_duals_relaxed(m, flow_cars)

            # Are there any shadow prices to update?
            if duals:
                if value_function_update == AVERAGED_UPDATE:
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


def get_denied_ids(decisions, attribute_trips_dict):
    
    # Start denied trip count with all trips
    denied = defaultdict(int)

    # Denied trip count per attribute (start with all trips)
    denied_count_dict = {
        trip_a: len(trip_list)
        for trip_a, trip_list in attribute_trips_dict.items()
    }

    # Loop decisions and discount fulfilled trips
    for d in decisions:

        if d[ACTION] == du.TRIP_DECISION:

            trip_a = (d[ORIGIN], d[DESTINATION])

            # Subtract trips fulfilled
            denied_count_dict[trip_a] -= d[N_DECISIONS]

    # Count the 
    for trip_a, n_denied in denied_count_dict.items():
        if n_denied >= 0:
            o, d = trip_a
            denied[o] += n_denied
    return denied


def adp_network_hired(
    env,
    trips,
    time_step,
    sq_guarantee=True,
    charge=True,
    agg_level=None,
    myopic=False,
    log_path=None,
    value_function_update=AVERAGED_UPDATE,
    episode=None,
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
        folder_epi_log = f"{env.config.folder_mip_log}episode_{episode:04}/"
        folder_epi_lp = f"{env.config.folder_mip_lp}episode_{episode:04}/"

        if not os.path.exists(folder_epi_log):
            os.makedirs(folder_epi_log)
            os.makedirs(folder_epi_lp)

        m.Params.LogFile = f"{folder_epi_log}mip_{time_step:04}.log"
        m.Params.ResultFile = f"{folder_epi_lp}mip_{time_step:04}.lp"

    else:
        # Disables all logging (file and console)
        m.setParam("OutputFlag", 0)

    # ##################################################################
    # SORT CARS ########################################################
    # ##################################################################
    (
        # Dictionary of cars per tuple (g, G(position))
        level_id_cars_dict,
        # Dictionary of target positions for each position
        rebalance_targets_dict,
        # [TYPE_HIRED|TYPE_FLEET] -> (position, battery) -> list of cars
        type_attribute_cars_dict,
    ) = sortout_fleets(env)

    # ##################################################################
    # SORT TRIPS #######################################################
    # ##################################################################
    (
        #  Dictionary of #trips per trip attribute,i.e., (o.id, d.id)
        attribute_trips_dict,
        # (level, id_level(origin)) -> trips
        level_id_trips_dict,
        # Number of trips per class
        class_count_dict,
    ) = sortout_trip_list(trips)

    # ##################################################################
    # VARIABLES ########################################################
    # ##################################################################

    decision_cars, decision_class = du.get_decision_set_classed(
        env.available,
        env.available_hired,
        level_id_cars_dict,
        level_id_trips_dict,
        rebalance_targets_dict,
        max_battery_level=env.config.battery_levels,
    )

    # Join decision tuples of both fleets (hired and self owned)
    all_decisions = list(itertools.chain.from_iterable(decision_cars.values()))

    # Create variables
    x_var = m.addVars(
        tuplelist(all_decisions), name="x", vtype=GRB.INTEGER, lb=0
    )

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
    present_contribution = quicksum(
        env.cost_func(
            d[CAR_TYPE],
            d[ACTION],
            d[POSITION],
            d[ORIGIN],
            d[DESTINATION]
        ) * x_var[d]
        for d in x_var
    )

    # Maximize present and future outcome
    m.setObjective(
        present_contribution
        + env.config.discount_factor*post_decision_costs,
        GRB.MAXIMIZE
    )

    # ---------------------------------------------------------------- #
    # CONSTRAINTS ######################################################
    # ---------------------------------------------------------------- #

    # Car flow conservation
    flow_cars_dict = car_flow_constrs(
        m, x_var, type_attribute_cars_dict
    )

    # Trip flow conservation
    flow_trips = trip_flow_constrs(
        m, x_var, attribute_trips_dict
    )

    # Service quality constraints
    if sq_guarantee:
        sq_flow_dict = sq_constrs(
            m, x_var, decision_class, class_count_dict
        )

    # Car is obliged to charged if battery reaches minimum level
    # Car flow conservation
    if charge:
        max_battery = env.config.battery_levels
        car_recharge_dict = recharge_constrs(
            m, x_var, type_attribute_cars_dict, max_battery
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
        best_decisions = extract_decisions(x_var)

        # Number of customers rejected per origin id
        denied_count_dict = get_denied_ids(
            best_decisions,
            attribute_trips_dict
        )

        # Hired fleet is appearing in trip origins
        # hired_cars = [
        #     HiredCar(
        #         env.points[env.points[pos].id_level(3)],
        #         env.battery_levels,
        #         20,
        #         current_step=time_step,
        #         current_arrival=time_step * env.config.time_increment,
        #         battery_level_miles_max=env.battery_size_distances,
        #     )
        #     for pos, count in denied_count_dict.items()
        # ]

        # # Add hired fleet to model
        # env.hired_cars.extend(hired_cars)
        # env.available_hired.extend(hired_cars)

        # (
        #     # Dictionary of cars per tuple (g, G(position))
        #     level_id_cars_dict,
        #     # Dictionary of target positions for each position
        #     rebalance_targets_dict,
        #     # [TYPE_HIRED|TYPE_FLEET] -> (position, battery) -> list of cars
        #     type_attribute_cars_dict,
        # ) = sortout_fleets(env)

        # Update shadow prices to be used in the next iterations
        if not myopic:

            duals_dict = extract_duals_relaxed(m, flow_cars_dict)

            for car_type, duals in duals_dict.items():

                # Are there any shadow prices to update?
                if duals:
                    if value_function_update == AVERAGED_UPDATE:
                        env.adp.averaged_update(time_step, duals)
                    else:
                        env.adp.update_values_smoothed(time_step, duals)

        reward, serviced, denied = env.realize_decision(
            time_step,
            best_decisions,
            attribute_trips_dict,
            type_attribute_cars_dict,
        )
        # print(f"Objective Function - {m.objVal:6.2f} X
        # {reward:6.2f} - Decision reward")

        rejected = []

        # Update list of rejected orders
        rejected.extend(denied)

        return reward, serviced, rejected

    if (
        m.status != GRB.Status.INF_OR_UNBD and
        m.status != GRB.Status.INFEASIBLE
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


def adp_network_hired2(
    env,
    trips,
    time_step,
    sq_guarantee=True,
    charge=True,
    agg_level=None,
    myopic=False,
    log_path=None,
    value_function_update=AVERAGED_UPDATE,
    episode=None,
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
        folder_epi_log = f"{env.config.folder_mip_log}episode_{episode:04}/"
        folder_epi_lp = f"{env.config.folder_mip_lp}episode_{episode:04}/"

        if not os.path.exists(folder_epi_log):
            os.makedirs(folder_epi_log)
            os.makedirs(folder_epi_lp)

        m.Params.LogFile = f"{folder_epi_log}mip_{time_step:04}.log"
        m.Params.ResultFile = f"{folder_epi_lp}mip_{time_step:04}.lp"

    else:
        # Disables all logging (file and console)
        m.setParam("OutputFlag", 0)

    # ##################################################################
    # SORT CARS ########################################################
    # ##################################################################
    (
        # Dictionary of cars per tuple (g, G(position))
        level_id_cars_dict,
        # Dictionary of target positions for each position
        rebalance_targets_dict,
        # [TYPE_HIRED|TYPE_FLEET] -> (position, battery) -> list of cars
        type_attribute_cars_dict,
    ) = sortout_fleets(env)

    # ##################################################################
    # SORT TRIPS #######################################################
    # ##################################################################
    (
        #  Dictionary of #trips per trip attribute,i.e., (o.id, d.id)
        attribute_trips_dict,
        # (level, id_level(origin)) -> trips
        level_id_trips_dict,
        # Number of trips per class
        class_count_dict,
    ) = sortout_trip_list(trips)

    # ##################################################################
    # VARIABLES ########################################################
    # ##################################################################

    # Get all decision tuples, and trip decision tuples per service 
    # quality class. If max. battery level is defined, also includes
    # recharge decisions.
    decision_cars, decision_class = du.get_decision_set_classed(
        env.available,
        env.available_hired,
        level_id_cars_dict,
        level_id_trips_dict,
        rebalance_targets_dict,
        # max_battery_level=env.config.battery_levels,
    )

    # Join decision tuples of both fleets (hired and self owned)
    all_decisions = list(itertools.chain.from_iterable(decision_cars.values()))

    # Create variables
    x_var = m.addVars(
        tuplelist(all_decisions), name="x", vtype=GRB.INTEGER, lb=0
    )

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
    present_contribution = quicksum(
        env.cost_func(
            d[CAR_TYPE],
            d[ACTION],
            d[POSITION],
            d[ORIGIN],
            d[DESTINATION],
            d[SQ_CLASS]
        ) * x_var[d]
        for d in x_var
    )

    # Maximize present and future outcome
    m.setObjective(
        present_contribution
        + env.config.discount_factor*post_decision_costs,
        GRB.MAXIMIZE
    )

    # ---------------------------------------------------------------- #
    # CONSTRAINTS ######################################################
    # ---------------------------------------------------------------- #

    # Car flow conservation
    flow_cars_dict = car_flow_constrs(m, x_var, type_attribute_cars_dict)

    # Trip flow conservation
    flow_trips = trip_flow_constrs(m, x_var, attribute_trips_dict)

    # Service quality constraints
    if sq_guarantee:
        sq_flow_dict = sq_constrs(
            m, x_var, decision_class, class_count_dict
        )

    # Car is obliged to charged if battery reaches minimum level
    # Car flow conservation
    if charge:
        max_battery = env.config.battery_levels
        car_recharge_dict = recharge_constrs(
            m, x_var, type_attribute_cars_dict, max_battery
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
        best_decisions = extract_decisions(x_var)

        # Number of customers rejected per origin id
        denied_count_dict = get_denied_ids(
            best_decisions,
            attribute_trips_dict
        )

        # Hired fleet is appearing in trip origins
        # hired_cars = [
        #     HiredCar(
        #         env.points[env.points[pos].id_level(3)],
        #         env.battery_levels,
        #         20,
        #         current_step=time_step,
        #         current_arrival=time_step * env.config.time_increment,
        #         battery_level_miles_max=env.battery_size_distances,
        #     )
        #     for pos, count in denied_count_dict.items()
        # ]

        # # Add hired fleet to model
        # env.hired_cars.extend(hired_cars)
        # env.available_hired.extend(hired_cars)

        # (
        #     # Dictionary of cars per tuple (g, G(position))
        #     level_id_cars_dict,
        #     # Dictionary of target positions for each position
        #     rebalance_targets_dict,
        #     # [TYPE_HIRED|TYPE_FLEET] -> (position, battery) -> list of cars
        #     type_attribute_cars_dict,
        # ) = sortout_fleets(env)

        # Update shadow prices to be used in the next iterations
        if not myopic:

            duals_dict = extract_duals_relaxed(m, flow_cars_dict)

            for car_type, duals in duals_dict.items():

                # Are there any shadow prices to update?
                if duals:
                    if value_function_update == AVERAGED_UPDATE:
                        env.adp.averaged_update(time_step, duals)
                    else:
                        env.adp.update_values_smoothed(time_step, duals)

        reward, serviced, denied = env.realize_decision(
            time_step,
            best_decisions,
            attribute_trips_dict,
            type_attribute_cars_dict,
        )
        # print(f"Objective Function - {m.objVal:6.2f} X {reward:6.2f} - Decision reward")

        rejected = []

        # Update list of rejected orders
        rejected.extend(denied)

        return reward, serviced, rejected

    if (
        m.status != GRB.Status.INF_OR_UNBD and
        m.status != GRB.Status.INFEASIBLE
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

# #################################################################### #
# Sortout resources and trips ######################################## #
# #################################################################### #


def sortout_fleets(env):
    """Associate vehicles from both fleets to its region center levels
    and ids, find the rebalance targets from each position, and list the
    cars per attribute (point, battery) and car type (hired or fleet).
    
    Parameters
    ----------
    env : Amod environment
        Amod 
    
    Returns
    -------
    [type]
        [description]
    """

    # Tuple of region center levels cars can rebalance to
    rebalance_levels = env.config.rebalance_level

    # Number of targets can reach at each level
    n_targets_level = env.config.n_neighbors

    # If not None, access immediate neighbors (intersections)
    rebalance_reach = env.config.rebalance_reach

    # ##################################################################
    # SORT CARS ########################################################
    # ##################################################################

    # How many cars per attribute
    type_attribute_cars_dict = defaultdict(lambda: defaultdict(list))

    # Which positions are surrounding each car position
    attribute_rebalance = defaultdict(lambda: defaultdict(list))

    # Which positions are surrounding each car position
    dict_level_id = defaultdict(lambda: defaultdict(list))

    # Cars can explore levels corresponding to the largest region center
    # considered by users.
    class_levels = ClassedTrip.get_levels()

    for car in env.available + env.available_hired:

        # List of cars with the same attribute (pos, battery level)
        type_attribute_cars_dict[car.type][car.attribute].append(car)

        # Cars in the same positions can rebalance to the same places.
        # Check if rebalance targets were previously determined
        if car.point.id not in attribute_rebalance[car.type]:

            # Get immediate neighbors (intersections) at reach degrees
            if rebalance_reach:
                rebalance_targets = env.get_neighbors(
                    car.point,
                    reach=rebalance_reach
                )
            # Get region center neighbors
            else:
                rebalance_targets = env.get_zone_neighbors(
                    car.point,
                    level=rebalance_levels,
                    n_neighbors=n_targets_level,
                )

            # All points a car can rebalance to from its corrent point
            attribute_rebalance[car.type][car.point.id] = rebalance_targets

        # Associate each car to superior aggregation levels and ids,
        # up until the largest region centers requests can be matched.
        for level in range(max(class_levels) + 1):
            id_level = car.point.id_level(level)
            dict_level_id[car.type][(level, id_level)].append(car)

    return (
        dict(dict_level_id),
        attribute_rebalance,
        dict(type_attribute_cars_dict),
    )


def sortout_trip_list(trips):
    """
    1 - Associate to each level g and level id g_id lists of trips whose
    G(origin id) is g_id at level g;
    2 - Count the number of trips per service quality class;
    3 - Associate trips to od tuples (trip attributes).

    Parameters
    ----------
    trips : list
        List of trips to sort.

    Returns
    -------
    dict(dict(list)) and dict()
        level, level id and list of trips association
        trip count per service quality class.

    Example
    -------
    >>> t1 = Trip(o=[1,10,100], d=[2,20,100], sq=A(0,1))
    >>> t2 = Trip(o=[3,30,100], d=[2,20,100], sq=B(1,2))
    >>> trips = [t1,t2]
    >>> sortout_trip_list(trips)
    >>> {
            0:{1:[t1], 3:[t2]},
            1:{10:[t1], 30:[t2]},
            2:{100:[t1,t2]}
        }, {A:1, B:1}
    """
    trip_origins_level_id = defaultdict(list)
    class_count_dict = defaultdict(int)
    od_trip_list = defaultdict(list)

    # Create a dictionary associate
    for t in trips:

        # Trip count per class
        class_count_dict[t.sq_class] += 1

        # Group trips with the same ods
        od_trip_list[(t.o.id, t.d.id)].append(t)

        # Trips can be accessed from any point covered by all its levels
        # up to to the last service quality level.
        for level in range(0, t.sq2_level + 1):
            id_level = t.o.id_level(level)
            trip_origins_level_id[(level, id_level)].append(t)

    return od_trip_list, trip_origins_level_id, class_count_dict

# #################################################################### #
# CONSTRAINTS ######################################################## #
# #################################################################### #

def car_flow_constrs(m, x_var, type_attribute_cars_dict):

    flow_cars_dict = dict()

    for car_type, attribute_cars in type_attribute_cars_dict.items():
        flow_cars_dict[car_type] = m.addConstrs(
            (
                x_var.sum("*", point, battery, "*", "*", car_type, contract_duration, '*') ==
                len(attribute_cars[(point, battery, contract_duration, car_type)])
                for point, battery, contract_duration, car_type in attribute_cars.keys()
            ),
            f"CAR_FLOW_{car_type}",
        )

    return flow_cars_dict


def trip_flow_constrs(m, x_var, attribute_trips_dict):

    flow_trips = m.addConstrs(
        (
            x_var.sum(du.TRIP_DECISION, "*", "*", o, d, "*", "*", "*") <=
            len(attribute_trips_dict[(o, d)])
            for o, d in attribute_trips_dict
        ),
        "TRIP_FLOW",
    )
    return flow_trips


def recharge_constrs(m, x_var, type_attribute_cars_dict, battery_levels):

    # Car flow conservation
    car_recharge_dict = dict()

    for car_type, attribute_cars in type_attribute_cars_dict.items():  

        car_recharge_dict[car_type] = m.addConstrs(
            (
                x_var[(action, pos, level, o, d, car_type)] ==
                len(attribute_cars[(pos, level)])
                for action, pos, level, o, d, car_type in x_var
                if level <= battery_levels and
                action == du.RECHARGE_DECISION
            ),
            f"RECHARGE_{car_type}",
        )

    return car_recharge_dict


def sq_constrs(m, x_var, decision_class, class_count_dict):

    constr_sq_class = dict()
    # Minimum service rate for users of each class
    for sq_class, s_rate in ClassedTrip.sq_classes.items():

        # List of decisions associated to a user class
        var_list_class = []

        # Adding decisions to user class list
        for single_decision in set(decision_class[sq_class]):
            var_list_class.append(x_var[single_decision])

        # Add constraints
        if len(var_list_class) > 0:
            constr_sq_class[sq_class] = m.addConstr(
                quicksum(var_list_class) >=
                np.ceil(s_rate * class_count_dict[sq_class]),
                f"TRIP_FLOW_CLASS_{sq_class}",
            )

    return constr_sq_class

# #################################################################### #
# Matching methods ################################################### #
# #################################################################### #

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
    all_decisions = du.get_decision_tuples(
        cars_with_attribute.keys(),
        dict_attribute_neighbors,
        trips_with_attribute,
    )

    # print("\n## Time to get decisions:", time.time() - time1)

    # Adding variables
    x_var = m.addVars(all_decisions, name="x", vtype=GRB.INTEGER, lb=0)

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
            x_var.sum(du.TRIP_STAY_DECISION, "*", "*", o, d)
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
                and action == du.RECHARGE_REBALANCE_DECISION
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
