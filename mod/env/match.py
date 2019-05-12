from collections import defaultdict
import mod.env.network as nw
from gurobipy import tuplelist, GRB, Model, quicksum
from pprint import pprint
from mod.env.amod import Amod
import time

# Decisions are tuples following the format
# (ACTION, POSITION, BATTERY, ORIGIN, DESTINATION)

# Labels for decision tuples
ACTION = 0
POSITION = 1
BATTERY = 2
ORIGIN = 3
DESTINATION = 4


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


def extract_decisions(var_list):
    decisions = list()
    for k, var in var_list.items():
        action, point, level, o, d = k

        if var.x > 0.0001:
            decisions.append((action, point, level, o, d, int(var.x)))

            # print(f'v.varName:{v.varName}={v.x}')
    return decisions


def extract_solution(var_list, flow_cars):

    # list of decision tuples (action, point, level, o, d)
    decisions = list()

    # Dual values for car attributes (point, level)
    duals = dict()

    # Loop (decision tuple, var) pairs
    for decision, var in var_list.items():

        if var.x > 0.0001:
            decisions.append(decision + (int(var.x),))

            # FIll out duals
            car_attribute = (decision[POSITION], decision[BATTERY])

            if car_attribute not in duals:
                # pi = The constraint dual value in the current solution
                # (also known as the shadow price).
                c = flow_cars[car_attribute].pi
                if c > 0:
                    duals[car_attribute] = c

    return decisions, duals


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


def get_all_decisions_joint(attribute_neighbors, trip_ods):
    decisions = []
    for c in attribute_neighbors.keys():
        # STAY
        decisions.append((Amod.TRIP_STAY_DECISION,) + c + (c[0],) + (c[0],))
        # RECHARGE
        decisions.append(
            (Amod.RECHARGE_REBALANCE_DECISION,) + c + (c[0],) + (c[0],)
        )

        # PICKUP
        for o, d in trip_ods:
            if o in attribute_neighbors[c[0]]:
                decisions.append((Amod.TRIP_STAY_DECISION,) + c + (o,) + (d,))

        # REBALANCE
        for z in attribute_neighbors[c[0]]:
            if c[0] != z:
                decisions.append(
                    (Amod.RECHARGE_REBALANCE_DECISION,) + c + (c[0],) + (z,)
                )

    return tuplelist(decisions)


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

        best_decisions = extract_decisions(x_var)

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


def adp(env, trips, time_step, charge=True):

    # Starting assignment model
    m = Model("assignment")

    # Disables all logging (file and console)
    m.setParam("OutputFlag", 0)

    # Attributes are queried according to a aggregation level
    agg_level = env.config.incumbent_aggregation_level

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

    # Cost based on post decision (shadow prices from former iterations)
    shadow_cost = quicksum(
        (env.get_value(time_step, d, level=agg_level) * x_var[d])
        for d in x_var
    )

    # Cost of current decision
    cost = quicksum(
        env.cost_func(d[ACTION], d[ORIGIN], d[DESTINATION]) * x_var[d]
        for d in x_var
    )

    m.setObjective(cost + shadow_cost, GRB.MAXIMIZE)

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

        # c = time.time()
        # best_decisions2 = extract_duals(flow_cars, cars_with_attribute.keys())
        # decisions = extract_decisions(x_var)
        # a = time.time()
        best_decisions, duals = extract_solution(x_var, flow_cars)
        # b=time.time()

        # print(f'one: {a-c} X two:{b-a}')
        env.update_values(time_step, duals)

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
            cost_recharging = env.recharge(car, env.config.time_increment)

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
