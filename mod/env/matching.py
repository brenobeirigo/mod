import os
import sys

import numpy as np
from collections import defaultdict
from gurobipy import tuplelist, GRB, Model, quicksum

from mod.env.trip import ClassedTrip
from mod.env.car import Car, HiredCar
import mod.util.log_util as la
import mod.env.decisions as du
import mod.env.adp.adp as adp
import itertools
import pandas as pd
import time
from copy import deepcopy
from pprint import pprint

# Decisions are tuples following the format
# (ACTION, POSITION, BATTERY, ORIGIN, DESTINATION, SQ_CLASS)

# Labels for decision tuples
ACTION = 0
POSITION = 1
BATTERY = 2
CONTRACT_DURATION = 3
CAR_TYPE = 4
CAR_ORIGIN = 5
ORIGIN = 6
DESTINATION = 7
SQ_CLASS = 8
N_DECISIONS = 9

# #################################################################### #
# CONSTRAINTS ######################################################## #
# #################################################################### #


def is_optimal(m):
    if m.status == GRB.Status.OPTIMAL:
        return True
    elif m.status == GRB.Status.UNBOUNDED:
        print("The model cannot be solved because it is unbounded")
        return False
    elif (
        m.status != GRB.Status.INF_OR_UNBD
        and m.status != GRB.Status.INFEASIBLE
    ):
        print("Optimization was stopped with status %d" % m.status)
        return False

    elif m.status == GRB.Status.INFEASIBLE:
        print(f"Model infeasible (status={m.status}")
        return False


def optimize_and_fix_fractional_vars(m, logger=None):
    def sortkey(v1):
        """Key function used to sort variables based on relaxation
        fractionality"""

        sol = v1.x
        return abs(sol - int(sol + 0.5))

    m.optimize()

    if is_optimal(m):

        for i in range(1000):

            # Create a list of fractional variables, sorted in order of
            # increasing distance from the relaxation solution to the
            # nearest integer value

            fractional = []
            for v in m.getVars():
                sol = v.x
                if abs(sol - int(sol + 0.5)) > 1e-5:
                    fractional += [v]

            if len(fractional) == 0:
                break

            fractional.sort(key=sortkey)

            # Fix the first quartile to the nearest integer value
            logger.debug(
                f"Iteration {i}, obj {m.objVal}, fractional {len(fractional)}"
            )

            nfix = max(int(len(fractional) / 4), 1)
            for i in range(nfix):
                v = fractional[i]
                fixval = int(v.x + 0.5)
                v.lb = fixval
                v.ub = fixval
                logger.debug(f"  Fix {v.varName} to {fixval} (rel {v.x})")

            m.optimize()
            if not is_optimal(m):
                break


def car_flow_constr(m, x_var, attribute_cars_dict):

    flow_cars_dict = m.addConstrs(
        (
            x_var.sum("*", *car_attribute, "*", "*", "*")
            == len(attribute_cars_dict[car_attribute])
            for car_attribute in attribute_cars_dict
        ),
        f"CAR_FLOW",
    )

    return flow_cars_dict


def trip_flow_constrs(m, x_var, attribute_trips_dict, universal_service=False):

    if universal_service:
        flow_trips = m.addConstrs(
            (
                x_var.sum(du.TRIP_DECISION, "*", "*", "*", "*", "*", o, d, "*")
                == len(attribute_trips_dict[(o, d)])
                for o, d in attribute_trips_dict
            ),
            "TRIP_FLOW",
        )

    else:
        flow_trips = m.addConstrs(
            (
                x_var.sum(du.TRIP_DECISION, "*", "*", "*", "*", "*", o, d, "*")
                <= len(attribute_trips_dict[(o, d)])
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
                x_var[(action, pos, level, o, d, car_type, car_origin)]
                == len(attribute_cars[(pos, level)])
                for action, pos, level, o, d, car_type, car_origin in x_var
                if level <= battery_levels and action == du.RECHARGE_DECISION
            ),
            f"RECHARGE_{car_type}",
        )

    return car_recharge_dict


def max_cars_node_constrs(
    m, decisions, vehicles_arriving_at, max_cars_node=5, unrestricted=[]
):
    """Restrict the number of cars arriving at each node.

    Parameters
    ----------
    m : Model
        Gurobi model.
    decisions : dict
        Dictionary associating each position p to the decisions in which
        p is the destination (STAY, REBALANCE, and TRIP).
    vehicles_arriving_at : dict
        Number of vehicles already arriving in each positions from
        previous assignments.
    max_cars_node : int, optional
        Max. number of cars per node, by default 5
    unrestricted : list, optional
        List of nodes where the maximum number of cars constraint
        is not applied (e.g., parking lots), by default []

    Returns
    -------
    dict
        Constraints associated to each destination p.
    """

    flood_avoidance_constrs = dict()

    for pos, constrs in decisions.items():

        # Depots are unrestricted (unlimited number of vehicles)
        if pos not in unrestricted:

            n_cars_link = max(
                0, max_cars_node - len(vehicles_arriving_at[pos])
            )

            flood_avoidance_constrs[pos] = m.addConstr(
                constrs <= n_cars_link, f"MAX_CARS_LINK[{pos}]"
            )

    return flood_avoidance_constrs


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
                quicksum(var_list_class)
                >= np.ceil(s_rate * class_count_dict[sq_class]),
                f"TRIP_FLOW_CLASS_{sq_class}",
            )

    return constr_sq_class


def return_to_station_constrs(m, x_var, decision_return):
    """FAVs must return to stations before their contract ends.

    Parameters
    ----------
    m : Model
        Gurobi model.
    x_var : tuplelist
        All decision variables.
    decision_return : list
        Decision tuples concerning return trips to FAV origin.
    """

    for d in decision_return:
        m.addConstr(
            x_var[d] == 1,
            f"RETURN_DEPOT_[{d[du.POSITION]},{d[du.CAR_ORIGIN]}]",
        )


# #################################################################### #
# UTILS ############################################################## #
# #################################################################### #


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

    # Number of denied trips per origin
    for trip_a, n_denied in denied_count_dict.items():
        if n_denied > 0:
            o, d = trip_a
            denied[o] += n_denied
    return denied


def linearize(model):
    """Get LP from integer model and appy former solution. For MIP
    problems, no dual information is ever available.

    Parameters
    ----------
    model : Gurobi model
        MIP already optimized (integer variables)

    Returns
    -------
    Gurobi Model
        MIP relaxed (re-optimized)
    """
    linear = model.fixed()

    for x in linear.getVars():
        x.vtype = GRB.CONTINUOUS

    # Turn off presolve
    linear.Params.presolve = 0

    linear.optimize()

    if linear.status != GRB.Status.OPTIMAL:
        raise Exception("Error: fixed model isn't optimal.")

    return linear


def extract_duals(m, flow_cars, ignore_zeros=False, logger=None):
    """Extract duals from car flow constraints.

    Parameters
    ----------
    m : Gurobi model
        MIP already optimized
    flow_cars: dict
        Dictionary of car flow attributes with key tuples:
        (pos, battery, contract_duration, car_type)

    Returns
    -------
    dict(dict())
        Dual value for each car type and attribute (point, battery)

    Details
    -------
    Gurobi cannot extract duals from multi-objective models:

    "We haven't attempted to generalize the notions of dual solutions
    or simplex bases for continuous multi-objective models, so you can't
    query attributes such as Pi, RC, VBasis, or CBasis for
    multi-objective solutions."
    """

    # Shadow prices associated to car attributes
    duals = dict()

    for pos, battery, contract_duration, car_type, car_origin in flow_cars:

        if car_type == Car.TYPE_VIRTUAL:
            print("VIRTUAL CAR!")

        try:
            constr = m.getConstrByName(
                f"CAR_FLOW[{pos},{battery},{contract_duration},{car_type},{car_origin}]"
            )

            # pi = The constraint dual value in the currsent solution
            shadow_price = constr.pi

            # if logger:
            #     logger.debug(
            #         f"The dual value of {constr.constrName} : {shadow_price}"
            #     )

        except Exception as e:
            if car_type == Car.TYPE_VIRTUAL:
                print("deu merda")
            if logger:
                logger.debug(
                    f"Can't extract dual from constraint '{constr}'."
                    f" Exception '{e}'."
                )
            shadow_price = 0

        # Should zero value functions be updated?
        if ignore_zeros and shadow_price == 0:
            continue

        if car_type == Car.TYPE_VIRTUAL:
            car_type = Car.TYPE_FLEET
            print(
                (pos, battery, contract_duration, car_type, car_origin),
                shadow_price,
            )

        duals[
            (pos, battery, contract_duration, car_type, car_origin)
        ] = shadow_price

    return duals


def extract_decisions(var_list):

    # list of decision tuples (action, point, level, o, d)
    decisions = list()

    # Loop (decision tuple, var) pairs
    for decision, var in var_list.items():

        if var.x > 0.1:
            decisions.append(decision + (round(var.x),))

    return decisions


def extract_decision_compare(var_list1, m2):

    diff = list()
    # Loop (decision tuple, var) pairs
    for decision, m1_var in var_list1.items():
        m2_var = m2.getVarByName(f'x[{",".join(map(str,list(decision)))}]')
        diff.append((decision, m1_var.x, m2_var.x))

    return diff


def get_total_cost(env, decision, time_step):
    post_cost, _ = env.post_cost(time_step, decision)
    return env.cost_func(decision) + env.config.discount_factor * post_cost


def get_artificial_duals(env, time_step, attribute_trips_dict):
    """Use rejected trips to create artificial shadow prices from 
    estimated lost profits.

    Virtual cars are created in all region centers covering the trip
    origin. To each car attribute, is associated a list of lost 
    profits.

    The average lost profits across car states belonging to all virtual
    cars is an estimate of the value function.

    Parameters
    ----------
    env : Amod
        Amod environment
    time_step : int
        Current time step (used to calculate post decision costs)
    attribute_trips_dict : dict(list)
        List of trips per od pair

    Returns
    -------
    dict(float)
        Lost profits per car_flow tuple:
        (id, battery_level, contract_duration, car_type, car_origin)
    """

    duals_dict = defaultdict(list)
    trip_decisions = list()

    # Loop lists of rejected trips per od
    for (o, _), trips in attribute_trips_dict.items():

        for t in trips:
            for g in range(0, len(env.config.level_dist_list)):

                # Region center at level g
                rc_g = env.points[o].id_level(g)
                rc_point_g = env.points[rc_g]

                # Create a virtual car in region center
                virtual_car = Car(rc_point_g)

                # Time to reach trip origin
                travel_time = env.get_travel_time_od(
                    virtual_car.point, t.o, unit="min"
                )

                # Can the car reach the trip origin?
                if travel_time <= t.max_delay:

                    # Decision of car servicing
                    decision = du.trip_decision(virtual_car, t)

                    trip_decisions.append(decision)

                    cost_artificial_trip = get_total_cost(
                        env, decision, time_step
                    )

                    # Create artificial car flow tuple
                    car_flow_state = (
                        rc_g,
                        virtual_car.battery_level,
                        Car.INFINITE_CONTRACT_DURATION,
                        virtual_car.type,
                        Car.DISCARD,
                    )

                    duals_dict[car_flow_state].append(cost_artificial_trip)

    # Average all estimates associated to virtual states
    duals_avg_dict = {
        car_flow_state: np.average(duals_list)
        for (car_flow_state, duals_list) in duals_dict.items()
    }

    logger_name = env.config.log_path(env.adp.n)

    # Logging cost calculus
    la.log_costs(
        logger_name,
        trip_decisions,
        env.cost_func,
        env.post_cost,
        time_step,
        env.config.discount_factor,
        msg="ARTIFICIAL TRIPS",
        post_opt=False,
    )

    # Log duals
    la.log_duals(logger_name, duals_avg_dict, msg="(ARTIFICIAL)")

    return duals_avg_dict


# #################################################################### #
# MIP ################################################################ #
# #################################################################### #


def service_trips(
    env,
    trips,
    time_step,
    charge=True,
    iteration=None,
    log_mip=False,
    universal_service=False,
    use_artificial_duals=True,
    log_times=True,
    car_type_hide=None,
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
    log_iteration : int, optional
        Iteration number tp log
    sq_guarantee : bool, optional
        If True, users are serviced accoring to their class
    universal_service : bool, optional
        If True, All users must be serviced 
    use_artificial_duals : bool, optional
        If True, insert arficial dual in all region centers associated
        to the rejected trips.

    Returns
    -------
    float, list, list
        total contribution, serviced trips, rejected trips
    """
    if log_times:
        t_decisions = 0
        t_duals = 0
        t_realize_decision = 0
        t_update = 0
        t_setup_costs = 0
        t_setup_constraints = 0
        t_optimize = 0
        t_total = 0

    # Updating current time step in the environment
    env.cur_step = time_step

    # Starting time and logs
    t1_total = time.time()
    logger_name = env.config.log_path(env.adp.n)
    logger = la.get_logger(logger_name)

    # Disable fleet
    env.toggle_fleet(car_type_hide)

    # Starting assignment model
    m = Model("assignment")

    # Log steps of current episode
    if log_mip:

        m.setParam("LogToConsole", 0)
        folder_epi_log = f"{env.config.folder_mip_log}episode_{iteration:04}/"
        folder_epi_lp = f"{env.config.folder_mip_lp}episode_{iteration:04}/"

        if not os.path.exists(folder_epi_log):
            os.makedirs(folder_epi_log)
            os.makedirs(folder_epi_lp)

        m.Params.LogFile = f"{folder_epi_log}mip_{time_step:04}.log"
        m.Params.ResultFile = f"{folder_epi_lp}mip_{time_step:04}.lp"

        logger.debug(f"Logging MIP execution in '{m.Params.LogFile}'")
        logger.debug(f"Logging MIP model in '{m.Params.ResultFile}'")

    else:
        # Disables all logging (file and console)
        m.setParam("OutputFlag", 0)

        # Model is deterministic (usefull for testing)
        m.setParam("Seed", 1)

    # ##################################################################
    # SORT CARS ########################################################
    # ##################################################################

    la.log_attribute_cars_dict(logger_name, env.attribute_cars_dict)

    # ##################################################################
    # SORT TRIPS #######################################################
    # ##################################################################

    # Number of trips per class
    class_count_dict = defaultdict(int)

    # List of trips per OD
    attribute_trips_dict = defaultdict(list)

    # List of trips per OD
    attribute_trips_sq_dict = defaultdict(list)

    # TODO Car productivity
    # How many trips in each region
    # count_trips_region = defaultdict(
    #     lambda: defaultdict(lambda: {"o": 0, "d": 0})
    # )

    # Create a dictionary associate
    for trip in trips:

        # Trip count per class
        class_count_dict[trip.sq_class] += 1

        # Group trips with the same ods
        attribute_trips_dict[(trip.o.id, trip.d.id)].append(trip)

        # Group trips with the same ods
        attribute_trips_sq_dict[trip.attribute].append(trip)

        # TODO Rebalance based on car productivity (trips/cars/area)
        # Trip count per region center
        # for g in range(len(env.config.level_dist_list)):
        #     count_trips_region[g][env.points[trip.o.id].id_level(g)]['o']+=1
        #     count_trips_region[g][env.points[trip.d.id].id_level(g)]['d']+=1

    # print("### Count car region")
    # pprint(env.count_car_region)

    # print("\n### Count trip region")
    # pprint(count_trips_region)

    # ##################################################################
    # VARIABLES ########################################################
    # ##################################################################

    # Get all decision tuples, and trip decision tuples per service
    # quality class. If max. battery level is defined, also includes
    # recharge decisions.
    logger.debug(
        f"  - Getting decisions  "
        f"(trips={len(trips)}, "
        f"available cars={len(env.available)+len(env.available_hired)})"
    )

    t1_decisions = time.time()
    decision_cars, decision_return, decision_class = du.get_decisions(
        env,
        trips
        # max_battery_level=env.config.battery_levels,
    )

    # virtual_decisions = du.get_virtual_decisions(env, trips)

    # logger.debug("\n ###### Virtual vehicles:")
    # for v in virtual_decisions:
    #     logger.debug(f' - {v}')

    # decision_cars.update(virtual_decisions)

    t_decisions = time.time() - t1_decisions

    logger.debug(f"  - Decision count: {len(decision_cars)}")

    # Logging cost calculus
    la.log_costs(
        logger_name,
        decision_cars,
        env.cost_func,
        env.post_cost,
        time_step,
        env.config.discount_factor,
        msg="TRIP DECISIONS",
        # filter_decisions=set([du.TRIP_DECISION]),
        post_opt=False,
    )

    # Create variables
    x_var = m.addVars(
        tuplelist(decision_cars), name="x", vtype=GRB.CONTINUOUS, lb=0
    )

    # ##################################################################
    # MODEL ############################################################
    # ##################################################################

    # ---------------------------------------------------------------- #
    # COST FUNCTION ####################################################
    # ---------------------------------------------------------------- #

    # Time to setup post decision costs
    t1_setup_costs = time.time()

    # If myopic, do not include post decision costs
    # If random, discard rebalance costs
    if env.config.myopic or env.config.policy_random:

        contribution = quicksum(
            env.cost_func(d, ignore_rebalance_costs=True) * x_var[d]
            for d in x_var
        )

    else:
        # Model has learned shadow costs from previous iterations and
        # can use them to determine post decision costs.
        contribution = quicksum(
            env.total_cost(time_step, d) * x_var[d] for d in x_var
        )

    penalty = 0
    # pprint(attribute_trips_sq_dict)
    if env.config.trip_rejection_penalty is not None:
        penalty = quicksum(
            (
                env.config.trip_rejection_penalty[sq]
                * (
                    len(tp_list)
                    - x_var.sum(
                        du.TRIP_DECISION, "*", "*", "*", "*", "*", o, d, sq
                    )
                )
            )
            for (o, d, sq), tp_list in attribute_trips_sq_dict.items()
        )

    # for (o, d, sq), tp_list in attribute_trips_sq_dict.items():
    #     print((o, d, sq), len(tp_list), env.config.trip_rejection_penalty[sq], x_var.sum(du.TRIP_DECISION, "*", "*", "*", "*", "*", o, d, sq))

    m.setObjective(contribution - penalty, GRB.MAXIMIZE)

    t_setup_costs = time.time() - t1_setup_costs

    # ---------------------------------------------------------------- #
    # CONSTRAINTS ######################################################
    # ---------------------------------------------------------------- #
    t1_setup_constraints = time.time()
    # Car flow conservation
    flow_cars_dict = car_flow_constr(m, x_var, env.attribute_cars_dict)

    # FAVs return to their origins before contract deadlines
    return_to_station_constrs(m, x_var, decision_return)

    # Trip flow conservation
    flow_trips = trip_flow_constrs(
        m, x_var, attribute_trips_dict, universal_service=universal_service
    )

    # Service quality constraints
    if env.config.sq_guarantee:
        sq_flow_dict = sq_constrs(m, x_var, decision_class, class_count_dict)

    # Car is obliged to charged if battery reaches minimum level
    # Car flow conservation
    if charge:
        max_battery = env.config.battery_levels
        car_recharge_dict = recharge_constrs(
            m, x_var, env.attribute_cars_dict, max_battery
        )

    # Limit the number of cars per node
    if env.config.max_cars_link:

        # decisions_time_pos = defaultdict(list)
        decisions_destination = defaultdict(int)

        for d in x_var:

            # FAVs inbound to their origin are not considered
            if d[du.CAR_TYPE] == Car.TYPE_HIRED:
                if d[du.CAR_ORIGIN] == d[du.DESTINATION]:
                    continue

            # Cars arriving at destination
            decisions_destination[d[du.DESTINATION]] += x_var[d]

        # Set up constraint
        # TODO previously env.depots were unrestricted
        max_cars_node_constr = max_cars_node_constrs(
            m,
            decisions_destination,
            env.cars_inbound_to,
            max_cars_node=env.config.max_cars_link,
            unrestricted=[],
        )

    t_setup_constraints = time.time() - t1_setup_constraints

    t1_optimize = time.time()

    # Try finding integer values for the fractional variables
    optimize_and_fix_fractional_vars(m, logger=logger)

    t_optimize = time.time() - t1_optimize

    if m.status == GRB.Status.UNBOUNDED:
        print("The model cannot be solved because it is unbounded")

    elif m.status == GRB.Status.OPTIMAL:

        la.log_solution(logger_name, x_var)

        # Decision tuple + (n. of times decision was taken)
        best_decisions = extract_decisions(x_var)

        # Logging cost calculus
        la.log_costs(
            logger_name,
            best_decisions,
            env.cost_func,
            env.post_cost,
            time_step,
            env.config.discount_factor,
            post_opt=True,
        )

        # Number of customers rejected per origin id
        denied_count_dict = get_denied_ids(
            best_decisions, attribute_trips_dict
        )

        logger.debug("Denied")
        logger.debug(denied_count_dict)

        # Update shadow prices to be used in the next iterations
        if env.config.train:

            try:

                t1_duals = time.time()

                # Extracting shadow prices from car flow constraints
                duals = extract_duals(
                    m,
                    flow_cars_dict,
                    ignore_zeros=env.config.adp_ignore_zeros,
                    logger=logger,
                )

                t_duals = time.time() - t1_duals

                # Log duals
                la.log_duals(logger_name, duals, msg="LINEAR")

                t1_update = time.time()

                # Use dictionary of duals to update value functions
                env.update_vf(duals, time_step)

                t_update = time.time() - t1_update

            except Exception as e:
                logger.debug(
                    f"Can't extract duals. Exception: '{e}'.", exc_info=True
                )

        t1_realize_decision = time.time()
        reward, serviced, rejected = env.realize_decision(
            time_step,
            best_decisions,
            attribute_trips_dict,
            env.attribute_cars_dict,
        )

        time_dict = dict()

        t_realize_decision = time.time() - t1_realize_decision

        # Add artificial value functions to each lost demand
        if use_artificial_duals:

            t1_artificial_duals = time.time()
            # realize_decision modifies attribute_trips_dict leaving
            # only the trips that were not fulfilled (per od)
            artificial_duals = get_artificial_duals(
                env, time_step, attribute_trips_dict
            )

            time_dict["arficial duals"] = [time.time() - t1_artificial_duals]

            t1_update_vf_artificial = time.time()
            # Use dictionary of duals to update value functions
            env.update_vf(artificial_duals, time_step)

            logger.debug("###### Artificial duals")
            time_dict["update_artificial"] = [
                time.time() - t1_update_vf_artificial
            ]
        
        # The penalties must be discounted from the contribution
        applied_penalties = sum(
            [
                env.config.trip_rejection_penalty[t_r.sq_class]
                for t_r in rejected
            ]
        )
        final_obj = reward-applied_penalties

        logger.debug(
            f"### Objective Function (costs and post costs) - {m.objVal:6.2f} "
            f"X {final_obj:6.2f} ({reward:.2f} -  {applied_penalties:.2f})"
            " - Decision's total reward (costs - penalties)"
        )

        t_total = time.time() - t1_total

        if log_times:
            time_dict.update(
                {
                    "iteration": [iteration],
                    "step": [time_step],
                    "decisions": [t_decisions],
                    "duals": [t_duals],
                    "realize_decision": [t_realize_decision],
                    "update_vf": [t_update],
                    "setup_costs": [t_setup_costs],
                    "setup_constraints": [t_setup_constraints],
                    "optimize": [t_optimize],
                    "total": [t_total],
                }
            )

            times_path = f"{env.config.folder_adp_log}times.csv"
            df = pd.DataFrame(time_dict)
            df.to_csv(
                times_path,
                header=(not os.path.exists(times_path)),
                mode="a",
                index=False,
            )

        # Enable fleet
        env.toggle_fleet(car_type_hide)

        return final_obj, serviced, rejected

    elif (
        m.status != GRB.Status.INF_OR_UNBD
        and m.status != GRB.Status.INFEASIBLE
    ):
        print("Optimization was stopped with status %d" % m.status)

    elif m.status == GRB.Status.INFEASIBLE:

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
    else:
        print(f"Error code: {m.status}.")
        print(
            "Model was proven to be either infeasible or unbounded."
            "To obtain a more definitive conclusion, set the "
            " DualReductions parameter to 0 and reoptimize."
        )

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

        # Save model
        m.write(f"myopic_error_code.lp")

        m.computeIIS()

        if m.IISMinimal:
            print("IIS is minimal\n")
        else:
            print("IIS is not minimal\n")
            print("\nThe following constraint(s) cannot be satisfied:")
        for c in m.getConstrs():
            if c.IISConstr:
                print("%s" % c.constrName)
