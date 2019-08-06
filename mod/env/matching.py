import os
import sys

import numpy as np
from collections import defaultdict
from gurobipy import tuplelist, GRB, Model, quicksum

from mod.env.trip import ClassedTrip
from mod.env.car import Car, HiredCar
import mod.util.log_aux as la
import mod.env.decisions as du

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
        if n_denied >= 0:
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
    # Turn off presolve
    linear.Params.presolve = 0

    linear.optimize()

    if linear.status != GRB.Status.OPTIMAL:
        raise Exception("Error: fixed model isn't optimal.")

    return linear


def extract_duals(m, flow_cars, ignore_zeros=False):
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

        try:
            constr = m.getConstrByName(
                f"CAR_FLOW[{pos},{battery},{contract_duration},{car_type},{car_origin}]"
            )

            # pi = The constraint dual value in the current solution
            shadow_price = constr.pi

            # print(f'The dual value of {constr.constrName} : {shadow_price}')
        except:
            shadow_price = 0

        # Should zero value functions be updated?
        if shadow_price == 0 and ignore_zeros:
            continue

        duals[
            (pos, battery, contract_duration, car_type, car_origin)
        ] = shadow_price

    return duals


def extract_decisions(var_list):

    # list of decision tuples (action, point, level, o, d)
    decisions = list()

    # Loop (decision tuple, var) pairs
    for decision, var in var_list.items():

        if var.x > 0.9:
            decisions.append(decision + (round(var.x),))

    return decisions


# #################################################################### #
# MIP ################################################################ #
# #################################################################### #


def service_trips(
    env,
    trips,
    time_step,
    charge=True,
    agg_level=None,
    myopic=False,
    log_iteration=None,
    sq_guarantee=False,
    universal_service=False,
    penalize_rebalance=True,
):

    logger = la.get_logger(env.config.label)

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
    log_iteration : int, optional
        Iteration number tp log
    sq_guarantee : bool, optional
        If True, users are serviced accoring to their class
    universal_service : bool, optional
        If True, All users must be serviced 
    penalize_rebalance : bool, optional
        If True, rebalancing is further punished (discount value that
        could have been gained by staying still)

    Returns
    -------
    float, list, list
        total contribution, serviced trips, rejected trips
    """

    # Starting assignment model
    m = Model("assignment")

    # Log steps of current episode
    if log_iteration is not None:

        m.setParam("LogToConsole", 0)
        folder_epi_log = (
            f"{env.config.folder_mip_log}episode_{log_iteration:04}/"
        )
        folder_epi_lp = (
            f"{env.config.folder_mip_lp}episode_{log_iteration:04}/"
        )

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

    # ##################################################################
    # SORT CARS ########################################################
    # ##################################################################

    # [TYPE_HIRED|TYPE_FLEET] -> (position, battery) -> list of cars
    attribute_cars_dict = defaultdict(list)

    available_fleet = env.available + env.available_hired
    for car in available_fleet:
        # List of cars with the same attribute (pos, battery level)
        attribute_cars_dict[car.attribute].append(car)

    # ##################################################################
    # SORT TRIPS #######################################################
    # ##################################################################

    class_count_dict = defaultdict(int)
    attribute_trips_dict = defaultdict(list)

    # Create a dictionary associate
    for t in trips:

        # Trip count per class
        class_count_dict[t.sq_class] += 1

        # Group trips with the same ods
        attribute_trips_dict[(t.o.id, t.d.id)].append(t)

    # ##################################################################
    # VARIABLES ########################################################
    # ##################################################################

    # Get all decision tuples, and trip decision tuples per service
    # quality class. If max. battery level is defined, also includes
    # recharge decisions.
    logger.debug(
        f"  - Getting decisions  "
        f"(trips={len(trips)}, cars={len(available_fleet)})"
    )

    decision_cars, decision_class = du.get_decisions(
        env,
        trips,
        # max_battery_level=env.config.battery_levels,
    )
    logger.debug(f"  - Decision count: {len(decision_cars)}")

    # Create variables
    x_var = m.addVars(
        tuplelist(decision_cars), name="x", vtype=GRB.INTEGER, lb=0
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
            (
                env.post_cost(
                    time_step,
                    d,
                    level=agg_level,
                    penalize_rebalance=penalize_rebalance,
                )
                * x_var[d]
            )
            for d in x_var
        )

    # Cost of current decision
    present_contribution = quicksum(env.cost_func(d) * x_var[d] for d in x_var)

    # Maximize present and future outcome
    m.setObjective(
        present_contribution
        + env.config.discount_factor * post_decision_costs,
        GRB.MAXIMIZE,
    )

    # ---------------------------------------------------------------- #
    # CONSTRAINTS ######################################################
    # ---------------------------------------------------------------- #

    # Car flow conservation
    flow_cars_dict = car_flow_constr(m, x_var, attribute_cars_dict)

    # Trip flow conservation
    flow_trips = trip_flow_constrs(
        m, x_var, attribute_trips_dict, universal_service=universal_service
    )

    # Service quality constraints
    if sq_guarantee:
        sq_flow_dict = sq_constrs(m, x_var, decision_class, class_count_dict)

    # Car is obliged to charged if battery reaches minimum level
    # Car flow conservation
    if charge:
        max_battery = env.config.battery_levels
        car_recharge_dict = recharge_constrs(
            m, x_var, attribute_cars_dict, max_battery
        )

    m.optimize()

    if m.status == GRB.Status.UNBOUNDED:
        print("The model cannot be solved because it is unbounded")

    elif m.status == GRB.Status.OPTIMAL:

        # Decision tuple + (n. of times decision was taken)
        best_decisions = extract_decisions(x_var)

        # Logging cost calculus
        la.log_costs(
            env.config.label,
            best_decisions,
            env.cost_func,
            env.post_cost,
            time_step,
            env.config.discount_factor,
            agg_level,
            penalize_rebalance,
        )

        # Number of customers rejected per origin id
        denied_count_dict = get_denied_ids(
            best_decisions, attribute_trips_dict
        )

        # Update shadow prices to be used in the next iterations
        if not myopic:

            try:
                # Relax model (LP) to get duals
                relaxed_model = linearize(m)
                logger.debug(
                    "### Difference original model and relaxed: "
                    f"{m.objVal - relaxed_model.objVal:6.2f}"
                )

                # Extracting shadow prices from car flow constraints
                duals = extract_duals(
                    relaxed_model, flow_cars_dict, ignore_zeros=True
                )

                # Log duals
                la.log_duals(env.config.label, duals)

                # Use dictionary of duals to update value functions
                env.update_vf(duals, time_step)

            except Exception as e:
                logger.debug(
                    f"Can't extract duals. Exception: '{e}'.", exc_info=True
                )

        reward, serviced, rejected = env.realize_decision(
            time_step,
            best_decisions,
            attribute_trips_dict,
            attribute_cars_dict,
        )

        logger.debug(
            "### Objective Function - "
            f"{m.objVal:6.2f} X {reward:6.2f}"
            " - Decision reward"
        )

        return reward, serviced, rejected

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
