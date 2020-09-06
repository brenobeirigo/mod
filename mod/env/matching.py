import itertools
import os
import time
from collections import defaultdict, namedtuple

import numpy as np
import pandas as pd
from gurobipy import tuplelist, GRB, Model, quicksum

import mod.env.adp.decisions as du
import mod.util.log_util as la
from mod.env.adp.DecisionSet import DecisionSet
from mod.env.adp.DecisionSetReactive import DecisionSetReactive
from mod.env.car import Car
from mod.env.trip import ClassedTrip

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

DecisionOutcome = namedtuple("DecisionOutcome", "cost,post_cost,post_state")


# #################################################################### #
# CONSTRAINTS ######################################################## #
# #################################################################### #


def is_optimal(m):
    if m.status == GRB.Status.OPTIMAL:
        return True
    elif m.status == GRB.Status.UNBOUNDED:
        print("The model cannot be solved because it is unbounded")

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
    else:
        print("Error model.")
        return False


# #################################################################### #
# OPTIMAL REBALANCING ################################################ #
# #################################################################### #

# Optimal rebalancing with perfect information
# Paper: Data-Driven Model Predictive Control of Autonomous
#        Mobility-on-Demand Systems


def ensure_all_demands_serviced_opt(m, cars_ijt, trips_ijt):
    """Ensures that all customer demands are serviced

    Parameters
    ----------
    m : model
        Gurobi model
    cars_ijt : var
        Number of cars moving from i to j at step t
    trips_ijt : dict
        Number of trips from i to j at step t
    """
    flow_cars_dict = m.addConstrs(
        (
            cars_ijt[(du.TRIP_DECISION, i, j, t)]
            == trips_ijt.get((i, j, t), 0)
            for d, i, j, t in cars_ijt.keys()
            if i != j and d == du.TRIP_DECISION
        ),
        name="MATCH_DEMAND",
    )

    return flow_cars_dict


def ensure_all_demands_serviced(
        env,
        m,
        N,
        T,
        cars_ijt,
        trips_ijt,
        step,
        outstandig_ijt=dict(),
        slack_ijt=dict(),
):
    """Ensures that all customer demands are serviced

    Parameters
    ----------
    m : model
        Gurobi model
    cars_ijt : var
        Number of cars moving from i to j at step t
    trips_ijt : dict
        Number of trips from i to j at step t
    """

    for i, j, t in trips_ijt:
        # print(f"# {i:04} - {t:04} = {s_it.get((i, t),0)}")
        m.addConstr(
            sum([cars_ijt.get((du.TRIP_DECISION, p, i, j, t), 0, ) for p in N])
            + slack_ijt.get((i, j, t), 0)
            - outstandig_ijt.get((i, j, t), 0)
            == (0 if t == step else trips_ijt.get((i, j, t), 0)),
            name=f"MATCH_DEMAND[{i},{j},{t}]",
        )


def mpc_optimal_car_flow(env, m, cars_ijt, N, T, s_it):
    """Enforces that for every time interval and each region, the number
    of arriving vehicles equals the number of departing vehicles.

    Parameters
    ----------
    env : Amod
        AMoD environment
    m : model
        Gurobi model
    cars_ijt : var
        Number of cars moving from i to j at step t
    N : list
        List of node ids
    T : int
        Time horizon
    s_it : var
        Number of cars per region i and step t
    """

    m.addConstrs(
        (
            cars_ijt.sum("*", i, "*", t)
            - quicksum(
                cars_ijt.get(
                    (d, j, i, t - max(1, env.travel_time(j, j, i))), 0
                )
                for j in N
                for d in [du.TRIP_DECISION, du.REBALANCE_DECISION]
            )
            == s_it.get((i, t), 0)
            for i in N
            for t in T
        ),
        name=f"CAR_FLOW",
    )


def mpc_car_flow(env, m, cars_pijt, N, T, s_it, step=0):
    """Enforces that for every time interval and each region, the number
    of arriving vehicles equals the number of departing vehicles.

    Parameters
    ----------
    env : Amod
        AMoD environment
    m : model
        Gurobi model
    cars_ijt : var
        Number of cars moving from i to j at step t
    N : list
        List of node ids
    T : int
        Time horizon
    s_it : var
        Number of cars per region i and step t
    """

    m.addConstrs(
        (
            cars_pijt.sum("*", "*", i, "*", t)
            - quicksum(
                cars_pijt.get(
                    (d, j, j, i, t - max(1, env.travel_time(j, j, i))), 0
                )
                for j in N
                for d in [du.TRIP_DECISION, du.REBALANCE_DECISION]
                # for (d, p, j, in_i, t_pji) in cars_pijt
                # if in_i == i
                # and t_pji == t - max(1, env.travel_time(p, j, in_i))
            )
            <= s_it.get((i, t), 0)
            for i in N
            for t in T
        ),
        name=f"CAR_FLOW",
    )


def log_model(m, env, folder_name="", save_files=False, step=None, label=""):
    # Log steps of current episode
    if save_files:
        print(f"Saving MIP{label} model...")

        logger_name = env.config.log_path(env.adp.n)
        logger = la.get_logger(logger_name)

        m.setParam("LogToConsole", 0)
        folder_log = f"{env.config.folder_mip_log}{folder_name}/"
        folder_lp = f"{env.config.folder_mip_lp}{folder_name}/"

        if not os.path.exists(folder_log):
            os.makedirs(folder_log)

        if not os.path.exists(folder_lp):
            os.makedirs(folder_lp)

        step_label = "" if step is None else f"_{step:04}"
        m.Params.LogFile = f"{folder_log}mip{step_label}{label}.log"
        m.Params.ResultFile = f"{folder_lp}mip{step_label}{label}.lp"

        logger.debug(f"Logging MIP execution in '{m.Params.LogFile}'")
        logger.debug(f"Logging MIP model in '{m.Params.ResultFile}'")

    else:
        # Disables all logging (file and console)
        m.setParam("OutputFlag", 0)


def mpc(
        env, current_trips, predicted_trips, step=0, log_mip=True,
):
    """
    Parameters
    ----------
    env : Amod
        Amod environment
    current_trips : list of trips
        List of trips that arrived in the current step.
    predicted_trips : list of list of trips
        List of predicted trips for future steps with size equals the
        horizon.
    step : int, optional
        Current step, by default 0
    horizon : int, optional
        How many steps (current step including) there is , by default 40
    log_mip : bool, optional
        If True, log gurobi .lp and .log, by default True
    use_visited_only : bool, optional
        If True, remove from node set N all nodes that are not a trip
        origin/destination, by default True

    Returns
    -------
    list of decisions, list of trips
        list of decisions for current step, list of filtered trips
        (i.e., trips that are not in the same region)
    """

    # Get all nodes from "centroid_level". All nodes a
    N = env.reachable_point_ids

    # Log events of iteration n

    # Starting time and logs
    # print("log_path:", env.config.log_path(step), " - log_file:", env.config.log_path(step))
    # logger = la.get_logger(
    #     env.config.log_path(step), log_file=env.config.log_path(step),
    # )

    t1_total = time.time()
    logger_name = env.config.log_path(env.adp.n)
    logger = la.get_logger(logger_name)

    # Current trips is in the first step
    predicted_trips.insert(0, current_trips)

    horizon = len(predicted_trips)

    # List
    T = np.arange(step, step + len(predicted_trips))

    logger.debug(
        f"{step:>3} - len(T) = min({step}, {step + len(predicted_trips)}) = {T}"
    )

    logger.debug(
        f"%%%%%%%%%%%% step={step:>4}, "
        f"horizon={horizon:>4}, "
        f"predicted={len(predicted_trips):>4} %%%%%%%%%%%%%"
    )

    logger.debug(
        f"#### Optimal rebalancing strategy #### "
        f"level={env.config.centroid_level}, "
        f"#N={len(N)}, "
        f"T={T} (total steps = {env.config.time_steps}),"
        f"Unreachable({len(env.unreachable_ods)})={env.unreachable_ods},"
    )

    m = Model("mpc_optimal")

    # Create model only with the nodes where trips/cars appear
    distinct_nodes = set()

    # Count of cars arriving in node i at time t
    s_it = dict()
    for i, t_cars in env.level_inbound_dict.items():
        # Nodes where cars arrive
        distinct_nodes.add(i)
        for arrival_step, cars in t_cars.items():
            s_it[(i, arrival_step)] = len(cars)

    logger.debug(
        f"\nCar arrivals per step and positions "
        f"(level={env.config.centroid_level}, step={step})"
    )
    for k, v in sorted(s_it.items(), key=lambda x: (x[0][1], x[0][0])):
        logger.debug(f"{k} = {v}")

    # How many trips per origin, destination, time
    ijt = set()
    trips_ijt = defaultdict(int)
    current_trips_ij = defaultdict(int)
    possible_trips = set()
    # Trips can be picked up within the whole horizon

    logger.debug("\nCreating trip_ijt (N x N x T) data...")
    vars_pijt = set()
    dpt = defaultdict(set)

    for batch_step, trips in enumerate(predicted_trips):

        appereance_step = T[batch_step]
        T_horizon = T[batch_step:]
        logger.debug(
            f" - step={appereance_step:04}, "
            f"n. of trips={len(trips):04}, "
            f"batch_step={batch_step:04}, "
            f"step_horizon={T_horizon}, "
            f"T={T}"
        )

        for trip in trips:
            o, d = trip.o.id, trip.d.id
            distinct_nodes.add(o)
            distinct_nodes.add(d)

            # Trip can happen in all steps after its appearance
            for t in T_horizon:
                ijt.add((o, d, t))
                vars_pijt.add((du.TRIP_DECISION, o, o, d, t))

            # (o, d, t) = n. of trips
            trips_ijt[(o, d, appereance_step)] += 1
            possible_trips.add((o, d, appereance_step))

            # appereance step is the current step
            if appereance_step == step:
                current_trips_ij[(o, d)] += 1

    logger.debug(f"vars_pijt(trips)={len(vars_pijt)}")

    # # Use only nodes where cars or trips are
    if env.config.mpc_use_trip_ods_only:
        N = distinct_nodes
        logger.debug(f"Distinct nodes: {len(N):,}.")

    logger.debug(f"\n# All trips (step={T}):")
    sorted_trips = sorted(
        trips_ijt.items(), key=lambda x: (x[0][2], x[0][0], x[0][1])
    )
    for k, v in sorted_trips:
        logger.debug(f"{k} = {v}")

    logger.debug(
        f"Creating cars_ijt (N x N x N x T) variables (step={step})..."
    )

    # decision (TRIP, REBALANCE), origin, destination, step
    # REBALANCE & (i == j) = STAY
    for batch_step in T:
        for i in N:
            # Stay
            vars_pijt.add((du.REBALANCE_DECISION, i, i, i, batch_step))
            dpt[i].add((du.REBALANCE_DECISION, i, batch_step))

            # Choose possible rebalancing targets
            # All nodes X only neighbors
            N_d = (
                env.neighbors[i]
                if env.config.mpc_rebalance_to_neighbors
                else N
            )

            for j in N_d:
                vars_pijt.add((du.REBALANCE_DECISION, i, i, j, batch_step))

    logger.debug(f"vars_pijt (trips + rebalance + stay)={len(vars_pijt)}")

    # ijt = set(trips_ijt.keys())

    cars_pijt = m.addVars(vars_pijt, name="x", vtype=GRB.CONTINUOUS, lb=0)

    # The slack variables D = {d_ijt}ijt denote the predicted demand of
    # customers wanting to travel from i to j departing at time t that
    # will remain unsatisfied
    slack_ijt = m.addVars(ijt, name="d", vtype=GRB.CONTINUOUS, lb=0)

    # w_ijt is a decision variable denoting the number of outstanding
    # customers at region i ∈ N who wish to travel to region j and be
    # picked up at time t ∈ T.
    outstanding_ijt = m.addVars(ijt, name="w", vtype=GRB.CONTINUOUS, lb=0)

    logger.debug(f" - {len(cars_pijt):,} variables created...")

    logger.debug(f"{len(trips_ijt):,} trip od tuples created.")

    t1 = time.time()
    logger.debug("Constraint 1 - Ensure the entire demand is met.")
    ensure_all_demands_serviced(
        env,
        m,
        N,
        T,
        cars_pijt,
        trips_ijt,
        step,
        outstandig_ijt=outstanding_ijt,
        slack_ijt=slack_ijt,
    )
    logger.debug(f" - time(s):{time.time() - t1:.2f}")

    t1 = time.time()
    logger.debug("Constraint 2 - Guarantee car flow.")
    mpc_car_flow(env, m, cars_pijt, N, T, s_it, step=step)
    logger.debug(f" - time(s):{time.time() - t1:.2f}")

    t1 = time.time()
    logger.debug("Constraint 3 - All outstanding passengers are served.")
    m.addConstrs(
        (
            outstanding_ijt.sum(i, j, "*") == current_trips_ij[(i, j)]
            for (i, j) in current_trips_ij
        ),
        name="OUT",
    )
    logger.debug(f" - time(s):{time.time() - t1:.2f}")

    logger.debug("\nSetting up contribution...")
    t1 = time.time()
    contribution = quicksum(
        env.cost_func(du.convert_decision(d, p, i, j))
        * cars_pijt[(d, p, i, j, t)]
        for d, p, i, j, t in cars_pijt
    )
    logger.debug(f" - time(s):{time.time() - t1:.2f}")

    if env.config.mpc_use_performance_to_go:
        logger.debug("\nSetting up performance to go values...")
        t1 = time.time()
        # Contribution for future steps
        contribution_future = quicksum(
            env.post_cost(t, du.convert_decision(d, p, i, j))[0]
            * cars_pijt[(d, p, i, j, t)]
            for d, p, i, j, t in cars_pijt
            if t == T[-1]
        )
        logger.debug(f" - time(s):{time.time() - t1:.2f}")
    else:
        contribution_future = 0

    # c_outstanding_ijt = cost associated with the waiting time of t
    #                     time steps for an outstanding passenger.
    # c_slack_ijt = cost for not servicing a predicted customer demand
    #               at time t.

    COST_OUTSTANDING = 0.25
    COST_SLACK = 2.5
    of_outstanding = quicksum(
        [
            outstanding_ijt[i, j, t] * ((batch_step - step) * COST_OUTSTANDING)
            for i, j, t in outstanding_ijt
        ]
    )

    of_slack = quicksum(
        [
            slack_ijt[i, j, t] * (batch_step - step) * COST_SLACK
            for i, j, t in slack_ijt
        ]
    )

    logger.debug("Setting objective (min. fleet, max. contribution)...")
    # m.setObjectiveN(fleet_size, 0, 2)
    m.setObjective(
        contribution + contribution_future - of_outstanding - of_slack,
        GRB.MAXIMIZE,
    )
    # m.setObjectiveN(of_outstanding + of_slack, 1, 0)
    # m.setObjectiveN(-contribution, 0, 1)

    # Log mip .log and .ilp
    log_model(m, env, save_files=log_mip, step=step)

    logger.debug("Optimizing...")

    m.optimize()

    if is_optimal(m):

        # Decision tuple + (n. of times decision was taken)
        # ACTION, ORIGIN, DESTINATION, STEP, N.DECISIONS
        best_decisions = sorted(
            extract_decisions(cars_pijt),
            key=lambda x: (x[3], x[0], x[1], x[2]),
        )

        best_decisions_w = sorted(
            extract_decisions(outstanding_ijt),
            key=lambda x: (x[2], x[0], x[1]),
        )

        best_decisions_d = sorted(
            extract_decisions(slack_ijt), key=lambda x: (x[2], x[0], x[1])
        )

        logger.debug(f"\nBest decisions x_ijt-n  (step={step}):")
        for d in best_decisions:
            logger.debug(f" - {d}")

        logger.debug(f"\nBest decisions w_ijt-n  (step={step}):")
        for d in best_decisions_w:
            logger.debug(f" - {d}")

        logger.debug(f"\nBest decisions d_ijt-n  (step={step}):")
        for d in best_decisions_d:
            logger.debug(f" - {d}")

        logger.debug(f"\nTrips & decisions per step  (step={step}):")

        # Decision list per step (played)
        step_decisions_list = []
        for tt, trips in enumerate(predicted_trips):
            appereance_step = tt + step
            logger.debug(
                f"\n## step={appereance_step:04} ####################################"
            )

            # Filter decisions at step
            step_decisions = [
                d for d in best_decisions if d[4] == appereance_step
            ]

            logger.debug(
                f" - Trips (step={appereance_step}, size={len(trips)}):"
            )
            logger.debug(
                sorted(
                    [
                        (i, j, trips_ijt[(i, j, t)])
                        for i, j, t in trips_ijt
                        if t == appereance_step
                    ],
                    key=lambda x: (x[0], x[1]),
                )
            )
            logger.debug(
                f" - Decisions (step={appereance_step}, size={len(step_decisions)}):"
            )

            # Converted decisions and sort by action, and od
            step_decisions = sorted(
                [
                    du.convert_decision(d[0], d[1], d[2], d[3], n=d[5])
                    for d in step_decisions
                ],
                key=lambda x: (x[du.ACTION], x[du.ORIGIN], x[du.DESTINATION]),
            )

            for d in step_decisions:
                logger.debug(f" - {du.shorten_decision(d)}")

            # Decisions for the whole horizon
            step_decisions_list.append(step_decisions)

        # d -> cost, post_cost, post_state
        # post_state -> (t, point, battery, contract, type, car_origin)
        env.decision_info = {
            d[:-1]: (env.cost_func(d),) for d in step_decisions_list[0]
        }

        return step_decisions_list[0]


def optimal_rebalancing(env, it_trips, log_mip=True, use_visited_only=True):
    """Under the assumption of perfect knowledge of customer demand,
    and assuming that the starting positions of the vehicles are free,
    it is possible to find the optimal rebalancing strategy by solving
    the following optimization problem:

    Parameters
    ----------
    env : Amod
        Amod environment
    it_trips : list of lists
        List of list of trips per time step
    log_mip : bool, optional
        If True, log gurobi .lp and .log, by default True
    use_visited_only : bool, optional
        If True, remove from node set N all nodes that are not a trip
        origin/destination, by default True

    Returns
    -------
    list of lists, list of lists
        list of decisions per step, list of filtered trips per step
    """
    # Get all nodes from "centroid_level"
    N = list(env.point_ids_level[env.config.centroid_level])
    T = np.array(range(0, len(it_trips))) + 1

    # Log events of iteration n
    logger = la.get_logger(
        env.config.log_path(env.adp.n),
        log_file=env.config.log_path(env.adp.n),
    )

    logger.debug(
        f"#### Optimal rebalancing strategy #### "
        f"level={env.config.centroid_level}, "
        f"nodes={len(N)}, "
        f"steps={T}"
    )

    m = Model("mpc_optimal")

    logger.debug("\nCreating s_it (N x T) variables...")
    s_it = m.addVars(N, T, name="s", vtype=GRB.CONTINUOUS, lb=0)
    logger.debug(f" - {len(s_it):,} variables created...")

    logger.debug("\nCreating trip_ijt (N x N x T) data...")
    # How many trips per origin, destination, time
    trips_ijt = defaultdict(int)

    # Filtered trips
    it_new_trips = list()

    # Create model only with the nodes where trips appear
    distinct_nodes = set()

    logger.debug(f"\nTotal trips = {sum([len(trips) for trips in it_trips])}")

    for appereance_step, trips in enumerate(it_trips):
        logger.debug(
            f" - step={appereance_step:04}, n. of trips={len(trips):04}"
        )
        new_trips = []
        for trip in trips:

            # Discard trips with origins and destinations
            # within the same region.
            if trip.o.id != trip.d.id and trip.o.id in N and trip.d.id in N:
                trips_ijt[(trip.o.id, trip.d.id, appereance_step)] += 1
                new_trips.append(trip)

                distinct_nodes.add(trip.o.id)
                distinct_nodes.add(trip.d.id)

        # Update new list of trips per step
        it_new_trips.append(new_trips)

    # Use only nodes where cars or trips are
    if use_visited_only:
        N = list(distinct_nodes)
        logger.debug(f"Distinct nodes: {len(N):,}.")
        logger.debug(
            f"Total trips = {sum([len(trips) for trips in it_new_trips])}"
        )

    logger.debug("Creating cars_ijt (N x N x T) variables...")
    # decision, origin, destination, timestep
    cars_ijt = m.addVars(
        [
            (d, i, j, t)
            for i in N
            for j in N
            for t in T
            for d in [du.TRIP_DECISION, du.REBALANCE_DECISION]
            if (i != j and d == du.TRIP_DECISION) or d == du.REBALANCE_DECISION
        ],
        name="x",
        vtype=GRB.CONTINUOUS,
        lb=0,
    )
    logger.debug(f" - {len(cars_ijt):,} variables created...")

    logger.debug(
        f"{len(trips_ijt):,} trip od tuples created "
        f"({len(list(itertools.chain(*it_new_trips))):,} discarded)."
    )

    logger.debug("Constraint 1 - Ensure the entire demand is met.")
    ensure_all_demands_serviced_opt(m, cars_ijt, trips_ijt)

    logger.debug("Constraint 2 - Guarantee car flow.")
    mpc_optimal_car_flow(env, m, cars_ijt, N, T, s_it)

    logger.debug("Constraint 3 - Starting vehicles only in the first step.")
    m.addConstrs(
        (s_it[i, t] == 0 for i in N for t in T if t > 1), name="START",
    )

    logger.debug("\nSetting up contribution...")
    contribution = quicksum(
        env.cost_func(du.convert_decision(d, i, i, j)) * cars_ijt[(d, i, j, t)]
        for d, i, j, t in cars_ijt
    )

    # Total number of vehicles
    fleet_size = quicksum(s_it)

    logger.debug("Setting objective (min. fleet, max. contribution)...")
    m.setObjectiveN(fleet_size, 0, 2)
    m.setObjectiveN(-contribution, 1, 1)

    # Log mip .log and .ilp
    log_model(m, env, save_files=log_mip, step=None)

    logger.debug("Optimizing...")
    m.optimize()

    if is_optimal(m):

        # Decision tuple + (n. of times decision was taken)
        # ACTION, ORIGIN, DESTINATION, STEP, N.DECISIONS
        best_decisions = extract_decisions(cars_ijt)

        # Add car point (repeat origin)
        best_decisions = [(d[0],) + (d[1],) + d[1:] for d in best_decisions]

        # ORIGIN, STEP (0), N.CARS
        itn_cars = extract_decisions(s_it)

        # Total number of cars
        fleet_size = sum(list(zip(*itn_cars))[2])

        logger.debug(f"Total cars = {fleet_size}")
        logger.debug(itn_cars)

        logger.debug("\nBest decisions 1:")
        for d in best_decisions:
            logger.debug(f" - {d}")

        logger.debug("\nTrips & decisions per step")

        # Decision list per step (played)
        step_decisions_list = []

        # Save all decision contributions
        env.decision_info = {}

        for appereance_step, trips in enumerate(it_new_trips):
            logger.debug(
                f"\n## step={appereance_step:04} ####################################"
            )

            # Filter decisions at step
            step_decisions = [
                d for d in best_decisions if d[4] == appereance_step
            ]

            logger.debug(f" - Trips = {len(trips)}")
            logger.debug(
                sorted(
                    [
                        (i, j, trips_ijt[(i, j, t)])
                        for i, j, t in trips_ijt
                        if t == appereance_step
                    ],
                    key=lambda x: (x[0], x[1]),
                )
            )

            logger.debug(" - Decisions:")
            step_decisions = sorted(
                [
                    du.convert_decision(d[0], d[1], d[2], d[3], n=d[5])
                    for d in step_decisions
                ],
                key=lambda x: (x[du.ACTION], x[du.ORIGIN], x[du.DESTINATION]),
            )

            # print("##########", appereance_step)
            # pprint(env.decis)

            for d in step_decisions:
                logger.debug(f" - {du.shorten_decision(d)}")
                env.decision_info[d[:-1]] = (env.cost_func(d),)

            step_decisions_list.append(step_decisions)

        # Change environment to start vehicles at points
        env.cars = []
        for i, _, n_cars in itn_cars:
            while n_cars > 0:
                env.cars.append(Car(env.points[i]))
                n_cars -= 1
        env.available = env.cars

        logger.debug(f"MPC optimal fleet size: {fleet_size}")

        return step_decisions_list, it_new_trips, fleet_size


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


def car_min_rebal_constr(m, x_var, fleet_size, n_targets):
    """Optimal rebalance Alonso-Mora et al. (2017)
    
    Parameters
    ----------
    m : model
        Gurobi model
    x_var : gurobi vars
        STAY and REBALANCING decisions
    fleet_size : int
        Total fleet (PAVs + FAVs)
    n_targets : int
        Number of rebalancing targets
    
    Returns
    -------
    Gurobi constraints
        Optimal rebalancing constraints.
    """
    flow_cars_dict = m.addConstr(
        x_var.sum(
            du.REBALANCE_DECISION, "*", "*", "*", "*", "*", "*", "*", "*"
        )
        == min(fleet_size, n_targets),
        f"CAR_REBAL",
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
        m,
        post_position_time_decisions,
        current_step,
        vehicles_arriving_at,
        max_cars_node=5,
        unrestricted=[],
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

    for post_position, arrival_step_decision_dict in post_position_time_decisions.items():

        arrival_steps_from_busy_cars = set(vehicles_arriving_at[post_position].keys()) - {current_step}
        current_arrival_steps = set(arrival_step_decision_dict.keys())

        sorted_arrival_times_at_post_position = sorted(
            list(arrival_steps_from_busy_cars.union(current_arrival_steps))
        )

        # print(
        #     f"Current step={current_step}",
        #     f"Destination={post_position}",
        #     f"All arrival times={sorted_arrival_times_at_post_position}",
        #     f"Vehicles arriving at destination={vehicles_arriving_at[post_position].keys()}",
        #     f"Constrs. keys={arrival_step_decision_dict.keys()}",
        # )

        constrs = 0
        n_cars_arriving_until_post_time = 0

        for arrival_step in sorted_arrival_times_at_post_position:
            # Depots are unrestricted (unlimited number of vehicles)
            if post_position not in unrestricted:

                n_busy_cars_arriving_at_arrival_step = len(vehicles_arriving_at[post_position].get(arrival_step, []))
                n_cars_arriving_until_post_time += n_busy_cars_arriving_at_arrival_step

                # if n_cars_arriving_until_post_time > 0:
                #     pprint(vehicles_arriving_at[post_position])
                #     print(
                #         f"MAX_CARS_LINK[{post_position},{arrival_step}] = "
                #         f"{max_cars_node} - {n_cars_arriving_until_post_time} = "
                #         f"{max_cars_node - n_cars_arriving_until_post_time}"
                #     )

                if arrival_step not in arrival_step_decision_dict:
                    # print("Constrs = ")
                    continue

                constrs += arrival_step_decision_dict[arrival_step]

                flood_avoidance_constrs[post_position] = m.addConstr(
                    constrs <= max(0, max_cars_node - n_cars_arriving_until_post_time),
                    f"MAX_CARS_LINK[{post_position},{arrival_step}]",
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
        if ignore_zeros and shadow_price <= 0:
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
        m2_var = m2.getVarByName(f'x[{",".join(map(str, list(decision)))}]')
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


def play_decisions(env, trips, time_step, decisions):
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
    best_decisions: int, optional
        Iteration number tp log

    Returns
    -------
    float, list, list
        total contribution, serviced trips, rejected trips
    """

    # List of trips per OD
    attribute_trips_dict = defaultdict(list)

    # Create a dictionary associate
    for trip in trips:
        # Group trips with the same ods
        attribute_trips_dict[(trip.o.id, trip.d.id)].append(trip)

    # print(f"Decisions (step={time_step})")
    # pprint(decisions)

    # print(f"Attribute trips (step={time_step})")
    # pprint(attribute_trips_dict)

    # print(f"Attributes (step={time_step})")
    # pprint(env.attribute_cars_dict)
    final_obj, applied_penalties, serviced, rejected = env.realize_decision(
        time_step, decisions, attribute_trips_dict, env.attribute_cars_dict,
    )

    return final_obj, serviced, rejected


class Matching:

    def __init__(self, env, trips, time_step, iteration=None, log_mip=False, log_times=True, car_type_hide=None,
                 reactive=False):
        self.env = env
        self.trips = trips
        self.time_step = time_step
        self.iteration = iteration
        self.log_mip = log_mip
        self.log_times = log_times
        self.car_type_hide = car_type_hide
        self.reactive = reactive
        self.x_var = None
        self.contribution = None
        self.logger_name = self.env.config.log_path(self.env.adp.n)
        self.logger = la.get_logger(self.logger_name)
        self.m = None
        self.flow_cars_dict = None
        self.times = dict()
        self.best_decisions = None
        self.decision_set = None
        self.rejected_upfront = []

    def service_trips(self):

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
        iteration : int, optional
            Iteration number tp log

        Returns
        -------
        float, list, list
            total contribution, serviced trips, rejected trips
        """

        # Updating current time step in the environment
        self.env.cur_step = self.time_step

        # Starting time and logs
        t1_total = time.time()

        la.log_node_centroid(
            self.logger_name, self.env.cars, self.env.points, self.env.unreachable_ods, self.env.neighbors
        )

        # Disable fleet
        self.env.toggle_fleet(self.car_type_hide)

        # Starting assignment model
        self.m = Model("assignment")

        # Log steps of current episode
        log_model(
            self.m,
            self.env,
            folder_name=self.iteration,
            save_files=self.log_mip,
            step=self.time_step,
            label=("_reb" if self.reactive else ""),
        )

        # Model is deterministic (usefull for testing)
        self.m.setParam("Seed", 1)
        self.m.setParam("DualReductions", 0)

        # ################################################################ #
        # ################################################################ #
        # ################################################################ #

        la.log_attribute_cars_dict(
            self.logger_name,
            self.env.attribute_cars_dict,
            self.env.level_step_inbound_cars,
            unrestricted_ids=self.env.unrestricted_parking_node_ids,
            max_cars=self.env.config.max_cars_link,
        )

        # Number of trips per class
        class_count_dict = defaultdict(int)

        # List of trips per OD
        attribute_trips_sq_dict = defaultdict(list)

        # ################################################################ #
        # REACTIVE REBALANCING ########################################### #
        # ################################################################ #

        # If rebalancing is reactive, rebalancing to unmet users
        if self.env.config.policy_reactive and self.reactive:

            self.decision_set = DecisionSetReactive(self.env, self.trips)
            self.log_reactive_rebalancing()

        else:
            # TODO Rebalance based on car productivity (trips/cars/area)
            t1_decisions = time.time()

            self.decision_set = DecisionSet(self.env, self.trips)
            # Trips that could not be matched to any vehicle
            self.rejected_upfront = self.decision_set.get_rejected_upfront()

            self.times["t_decisions"] = time.time() - t1_decisions

            self.log_decision_set_info()

        self.create_decision_variables()

        # ##################################################################
        # MODEL ############################################################
        # ##################################################################

        # ---------------------------------------------------------------- #
        # COST FUNCTION ####################################################
        # ---------------------------------------------------------------- #

        self.setup_objective_function()

        # ---------------------------------------------------------------- #
        # CONSTRAINTS ######################################################
        # ---------------------------------------------------------------- #
        t1_setup_constraints = time.time()

        # Car flow conservation
        self.flow_cars_dict = car_flow_constr(self.m, self.x_var, self.env.attribute_cars_dict)

        # 2nd round of reactive rebalance
        if self.reactive:
            # Sometimes, cars cannot rebalance due to contract limitations.
            # Then, this check guarantees the constraints is declared only
            # if there are valid rebalancing decision.

            # N. of rebalance = min(N. of cars, N. of targets)
            min_rebalance_dict = car_min_rebal_constr(
                self.m, self.x_var, self.decision_set.n_cars_can_rebalance, len(self.trips)
            )

        else:
            # FAVs return to their origins before contract deadlines
            return_to_station_constrs(self.m, self.x_var, self.decision_set.return_decisions)

            # Trip flow conservation
            flow_trips = trip_flow_constrs(
                self.m,
                self.x_var,
                self.decision_set.attribute_trips_dict,
                universal_service=self.env.config.universal_service,
            )

            # Service quality constraints
            if self.env.config.sq_guarantee:
                sq_flow_dict = sq_constrs(
                    self.m, self.x_var, self.decision_set.classed_decisions, class_count_dict
                )

            # Car is obliged to charged if battery reaches minimum level
            # Car flow conservation
            if self.env.config.enable_recharging:
                car_recharge_dict = recharge_constrs(
                    self.m, self.x_var, self.env.attribute_cars_dict, self.env.config.battery_levels
                )

        self.times["t_setup_constraints"] = time.time() - t1_setup_constraints

        t1_setup_constraints_flood = time.time()
        self.setup_constraints_flood()
        self.times["t_setup_constraints_flood"] = time.time() - t1_setup_constraints_flood

        self.optimize()

        if self.m.status == GRB.Status.UNBOUNDED:
            print("The model cannot be solved because it is unbounded")

        elif self.m.status == GRB.Status.OPTIMAL:

            la.log_solution(self.logger_name, self.x_var)

            # Decision tuple + (n. of times decision was taken)
            self.best_decisions = extract_decisions(self.x_var)

            # Logging cost calculus
            la.log_costs(
                self.logger_name,
                self.best_decisions,
                self.env.cost_func,
                self.env.post_cost,
                self.time_step,
                self.env.config.discount_factor,
                post_opt=True,
                msg="SOLUTION",
            )

            # Number of customers rejected per origin id
            denied_count_dict = get_denied_ids(
                self.best_decisions, self.decision_set.attribute_trips_dict
            )

            self.logger.debug("Denied")
            self.logger.debug(denied_count_dict)

            # Update shadow prices to be used in the next iterations
            if self.env.config.train:
                self.update_vfs()

            t1_realize_decision = time.time()
            (
                final_obj,
                applied_penalties,
                serviced,
                rejected,
            ) = self.env.realize_decision(
                self.time_step,
                self.best_decisions,
                self.decision_set.attribute_trips_dict,
                self.env.attribute_cars_dict,
            )

            rejected = rejected + self.rejected_upfront

            time_dict = dict()

            self.times["t_realize_decision"] = time.time() - t1_realize_decision

            # Add artificial value functions to each lost demand
            if self.env.config.use_artificial_duals:
                self.update_artificial_duals()

            self.logger.debug(
                f"### Objective Function (costs and post costs) - {self.m.objVal:6.2f} "
                f"X {final_obj:6.2f} (penalties={applied_penalties:.2f})"
                " - Decision's total reward (costs - penalties)"
            )

            self.times["t_total"] = time.time() - t1_total

            self.write_execution_times_df(time_dict)

            # Enable fleet
            self.env.toggle_fleet(self.car_type_hide)

            return final_obj, serviced, rejected

        elif (
                self.m.status != GRB.Status.INF_OR_UNBD
                and self.m.status != GRB.Status.INFEASIBLE
        ):
            print("Optimization was stopped with status %d" % self.m.status)

        elif self.m.status == GRB.Status.INFEASIBLE:
            self.do_IIS()

        else:
            print(f"Error code: {self.m.status}.")
            print(
                "Model was proven to be either infeasible or unbounded."
                "To obtain a more definitive conclusion, set the "
                " DualReductions parameter to 0 and reoptimize."
            )

    def setup_objective_function(self):

        t1_setup_costs = time.time()
        self.setup_cost_function()
        self.times["setup_costs"] = time.time() - t1_setup_costs

        t1_setup_penalties = time.time()
        penalty = self.setup_penalty()
        self.times["t_setup_penalties"] = time.time() - t1_setup_penalties

        self.m.setObjective(self.contribution - penalty, GRB.MAXIMIZE)

    def optimize(self):
        t1_optimize = time.time()
        # Try finding integer values for the fractional variables
        optimize_and_fix_fractional_vars(self.m, logger=self.logger)
        # m.optimize()
        self.times["t_optimize"] = time.time() - t1_optimize

    def create_decision_variables(self):
        # Create variables
        self.x_var = self.m.addVars(
            tuplelist(self.decision_set.all_decisions), name="x", vtype=GRB.CONTINUOUS, lb=0
        )

    def log_decision_set_info(self):
        # Logging decision set info
        la.log_decision_info(
            self.logger_name,
            self.time_step,
            self.trips,
            self.decision_set.reachable_trips_i,
            self.decision_set.all_decisions,
            self.env.available,
            self.env.available_hired,
            self.env.available_fleet_size,
        )
        # Logging cost calculus
        la.log_costs(
            self.logger_name,
            self.decision_set.all_decisions,
            self.env.cost_func,
            self.env.post_cost,
            self.time_step,
            self.env.config.discount_factor,
            # msg="",
            # filter_decisions=set([du.TRIP_DECISION]),
            post_opt=False,
        )

    def log_reactive_rebalancing(self):
        self.logger.debug(
            f"  - Reactive rebalance  "
            f"(targets={len(self.decision_set.targets)}, "
            f"decisions={len(self.decision_set.all_decisions)}, "
            f"available cars=[PAV={len(self.env.available)}, "
            f"FAV={len(self.env.available_hired)}, "
            f"total={self.env.available_fleet_size}])"
        )
        # Logging cost calculus
        la.log_costs(
            self.logger_name,
            self.decision_set.all_decisions,
            self.env.cost_func,
            self.env.post_cost,
            self.time_step,
            self.env.config.discount_factor,
            msg="REACTIVE DECISIONS",
            # filter_decisions=set([du.TRIP_DECISION]),
            post_opt=False,
        )

    def do_IIS(self):
        # do IIS
        print("The model is infeasible; computing IIS")
        self.m.computeIIS()
        if self.m.IISMinimal:
            print("IIS is minimal\n")
        else:
            print("IIS is not minimal\n")
            print("\nThe following constraint(s) cannot be satisfied:")
        for c in self.m.getConstrs():
            if c.IISConstr:
                print("%s" % c.constrName)
        # Save model
        self.m.write(f"myopic_error_code.lp")

    def update_artificial_duals(self):
        t1_artificial_duals = time.time()
        # realize_decision modifies attribute_trips_dict leaving
        # only the trips that were not fulfilled (per od)
        artificial_duals = get_artificial_duals(
            self.env, self.time_step, self.decision_set.attribute_trips_dict
        )
        self.times["artificial duals"] = [time.time() - t1_artificial_duals]
        t1_update_vf_artificial = time.time()
        # Use dictionary of duals to update value functions
        self.env.update_vf(artificial_duals, self.time_step)
        self.logger.debug("###### Artificial duals")
        self.times["update_artificial"] = [
            time.time() - t1_update_vf_artificial
        ]

    def update_vfs(self):
        try:

            t1_duals = time.time()

            # Extracting shadow prices from car flow constraints
            duals = extract_duals(
                self.m,
                self.flow_cars_dict,
                ignore_zeros=self.env.config.adp_ignore_zeros,
                logger=self.logger,
            )

            self.times["t_duals"] = time.time() - t1_duals

            # Log duals
            la.log_duals(self.logger_name, duals, msg="LINEAR")

            t1_update = time.time()

            # Use dictionary of duals to update value functions
            self.env.update_vf(duals, self.time_step)

            self.times["t_update"] = time.time() - t1_update

        except Exception as e:
            self.logger.debug(
                f"Can't extract duals. Exception: '{e}'.", exc_info=True
            )

    def setup_constraints_flood(self):

        # Limit the number of cars per node (not in reactive rebalance)
        if self.env.config.max_cars_link is not None and not self.env.config.policy_reactive:

            # decisions_time_pos = defaultdict(list)
            decisions_post_position_time = defaultdict(lambda: defaultdict(int))

            for d in self.x_var:

                # FAVs inbound to their origin are not considered
                if d[du.CAR_TYPE] == Car.TYPE_HIRED:
                    if d[du.CAR_ORIGIN] == d[du.DESTINATION]:
                        continue

                # Cars always can pickup customers
                if self.env.config.unbound_max_cars_trip_decisions:
                    if d[du.ACTION] == du.TRIP_DECISION:
                        continue

                # Cars arriving at destination
                post_time = self.env.decision_info[d].post_state.time
                decisions_post_position_time[d[du.DESTINATION]][post_time] += self.x_var[d]

            # Set up constraint
            max_cars_node_constr = max_cars_node_constrs(
                self.m,
                decisions_post_position_time,
                self.time_step,
                self.env.level_step_inbound_cars[self.env.config.centroid_level],
                max_cars_node=self.env.config.max_cars_link,
                unrestricted=self.env.unrestricted_parking_node_ids,
            )

    def setup_penalty(self):

        attribute_trips_sq_dict = defaultdict(list)
        for trip in self.trips:
            attribute_trips_sq_dict[trip.attribute_backlog].append(trip)

        penalty = 0
        if self.env.config.apply_backlog_rejection_penalty:

            # Penalty (o, d, sq_times)
            penalty_var = self.m.addVars(
                tuplelist(attribute_trips_sq_dict.keys()),
                name="y",
                vtype=GRB.CONTINUOUS,
                lb=0,
            )

            for (o, d, sq_timesback), tp_list in attribute_trips_sq_dict.items():
                self.m.addConstr(
                    penalty_var[o, d, sq_timesback]
                    == len(tp_list)
                    - self.x_var.sum(
                        du.TRIP_DECISION,
                        "*",
                        "*",
                        "*",
                        "*",
                        "*",
                        o,
                        d,
                        sq_timesback,
                    ),
                    f"PEN[{o},{d},{sq_timesback}]",
                )

            penalty = quicksum(
                (
                        self.env.config.backlog_rejection_penalty(sq_timesback)
                        * penalty_var[o, d, sq_timesback]
                )
                for (o, d, sq_timesback,) in penalty_var
            )
        return penalty

    def setup_cost_function(self):

        # If reactive, consider rebalancing costs
        if self.env.config.policy_reactive and self.reactive:

            # d -> cost, post_cost, post_state
            # post_state -> (t, point, battery, contract, type, car_origin)
            self.env.decision_info = {
                d: DecisionOutcome(
                    cost=self.env.cost_func(d, ignore_rebalance_costs=False),
                    post_cost=0,
                    post_state=self.env.preview_decision(self.time_step, d))
                for d in self.x_var
            }
            self.get_contribution()

        # If random, discard rebalance costs and add them later
        elif self.env.config.policy_random or self.env.config.policy_reactive:
            # d -> cost, post_cost, post_state
            # post_state -> (t, point, battery, contract, type, car_origin)
            self.env.decision_info = {
                d: DecisionOutcome(
                    cost=self.env.cost_func(d, ignore_rebalance_costs=True),
                    post_cost=0,
                    post_state=self.env.preview_decision(self.time_step, d))
                for d in self.x_var
            }
            self.get_contribution()

        # If myopic, do not include post decision costs
        elif self.env.config.myopic:
            self.setup_decision_costs()
            self.get_contribution()
        # ADP policy = cost + vfs
        else:

            # d -> cost, post_cost, post_state
            # post_state -> (t, point, battery, contract, type, car_origin)
            self.env.decision_info = {
                d: DecisionOutcome(
                    self.env.cost_func(d),
                    *self.env.post_cost(self.time_step, d)
                ) for d in self.x_var
            }

            self.get_contribution_plus_vfs()

    def get_contribution_plus_vfs(self):
        # Model has learned shadow costs from previous iterations and
        # can use them to determine post decision costs.
        self.contribution = quicksum(
            (
                    self.env.decision_info[d].cost
                    + self.env.config.discount_factor * self.env.decision_info[d].post_cost
            )
            * self.x_var[d]
            for d in self.x_var
        )

    def setup_decision_costs(self):
        # d -> cost, post_cost, post_state
        # post_state -> (t, point, battery, contract, type, car_origin)
        self.env.decision_info = {
            d: DecisionOutcome(
                cost=self.env.cost_func(d, ignore_rebalance_costs=False),
                post_cost=0,
                post_state=self.env.preview_decision(self.time_step, d)
            )
            for d in self.x_var
        }

    def write_execution_times_df(self, time_dict):
        if self.log_times:
            time_dict.update(
                {
                    "iteration": [self.iteration],
                    "step": [self.time_step],
                    "decisions": [self.times["t_decisions"]],
                    "duals": [self.times["t_duals"]],
                    "realize_decision": [self.times["t_realize_decision"]],
                    "update_vf": [self.times["t_update"]],
                    "setup_costs": [self.times["t_setup_costs"]],
                    "setup_penalties": [self.times["t_setup_penalties"]],
                    "setup_constraints": [self.times["t_setup_constraints"]],
                    "setup_constraints_flood": [self.times["t_setup_constraints_flood"]],
                    "optimize": [self.times["t_optimize"]],
                    "total": [self.times["t_total"]],
                }
            )

            times_path = f"{self.env.config.folder_adp_log}times.csv"
            df = pd.DataFrame(time_dict)
            df.to_csv(
                times_path,
                header=(not os.path.exists(times_path)),
                mode="a",
                index=False,
            )

    def get_contribution(self):
        self.contribution = quicksum(
            self.env.decision_info[d].cost * self.x_var[d] for d in self.x_var
        )
