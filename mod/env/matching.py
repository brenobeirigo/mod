import os
import sys

import numpy as np
from collections import defaultdict
from gurobipy import tuplelist, GRB, Model, quicksum

from mod.env.trip import ClassedTrip
from mod.env.car import Car, HiredCar
import mod.util.log_util as la
import mod.env.decisions as du
import itertools

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

        if var.x > 0.1:
            decisions.append(decision + (round(var.x),))

    return decisions


def get_total_cost(env, decision, time_step):
    return env.cost_func(
        decision
    ) + env.config.discount_factor * env.post_cost(time_step, decision)


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
                travel_time = env.get_travel_time_od(virtual_car.point, t.o, unit="min")

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
    myopic=False,
    log_iteration=None,
    sq_guarantee=False,
    universal_service=False,
    use_artificial_duals=True,
    linearize_model=True,
):

    logger_name = env.config.log_path(env.adp.n)
    logger = la.get_logger(logger_name)

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
    myopic : bool, optional
        If True, does not learn between iterations, by default False
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
        # Model is deterministic (usefull for testing)
        m.setParam("Seed", 1)

        # https://www.gurobi.com/documentation/8.1/refman/method.html#parameter:Method
        # -1=automatic, 0=primal simplex, 
        # 1=dual simplex, 
        # 2=barrier, 
        # 3=concurrent, 
        # 4=deterministic concurrent, 
        # 5=deterministic concurrent simplex.
        m.setParam("Method", 1)
        
        # https://www.gurobi.com/documentation/8.1/refman/mipfocus.html
        # If you are more interested in finding feasible solutions quickly, you can select MIPFocus=1.
        # If you believe the solver is having no trouble finding good quality solutions, and wish to focus more attention on proving optimality, select MIPFocus=2.
        # If the best objective bound is moving very slowly (or not at all), you may want to try MIPFocus=3 to focus on the bound. 
        m.setParam("MIPFocus", 1)


    # ##################################################################
    # SORT CARS ########################################################
    # ##################################################################

    attribute_cars_dict = defaultdict(list)

    for car in itertools.chain(env.available, env.available_hired):

        # List of cars with the same attribute
        attribute_cars_dict[car.attribute].append(car)

    la.log_attribute_cars_dict(logger_name, attribute_cars_dict)

    # ##################################################################
    # SORT TRIPS #######################################################
    # ##################################################################

    class_count_dict = defaultdict(int)
    attribute_trips_dict = defaultdict(list)

    # Create a dictionary associate
    for trip in trips:

        # Trip count per class
        class_count_dict[trip.sq_class] += 1

        # Group trips with the same ods
        attribute_trips_dict[(trip.o.id, trip.d.id)].append(trip)

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

    decision_cars, decision_class = du.get_decisions(
        env,
        trips,
        # max_battery_level=env.config.battery_levels,
    )

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

    var_type = GRB.INTEGER if linearize_model else GRB.CONTINUOUS

    # Create variables
    x_var = m.addVars(tuplelist(decision_cars), name="x", vtype=var_type, lb=0)

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
            (env.post_cost(time_step, d) * x_var[d]) for d in x_var
        )

    # Cost of current decision
    present_contribution = quicksum(env.cost_func(d) * x_var[d] for d in x_var)

    # Privilege trip decisions
    # trip_decisions = itertools.chain.from_iterable(decision_class.values())
    # extra_trip_weight = quicksum(BIG_M * x_var[d] for d in trip_decisions)
    extra_trip_weight = 0

    # Maximize present and future outcome
    m.setObjective(
        present_contribution
        + env.config.discount_factor * post_decision_costs
        + extra_trip_weight,
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
        if not myopic:

            try:
                # Relax MIP model to get duals
                if linearize_model:
                    m_old = m
                    m = linearize(m)
                    logger.debug(
                        "### Difference original model and relaxed: "
                        f"{m_old.objVal - m.objVal:6.2f}"
                    )

                # Extracting shadow prices from car flow constraints
                duals = extract_duals(m, flow_cars_dict, ignore_zeros=True)

                # Log duals
                la.log_duals(logger_name, duals)

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
        
        # Add artificial value functions to each lost demand
        if use_artificial_duals:

            # realize_decision modifies attribute_trips_dict leaving
            # only the trips that were not fulfilled (per od)
            artificial_duals = get_artificial_duals(
                env, time_step, attribute_trips_dict
            )

            # Use dictionary of duals to update value functions
            env.update_vf(artificial_duals, time_step)

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
