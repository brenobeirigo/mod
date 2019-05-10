from collections import defaultdict, Counter
import mod.env.network as nw
from gurobipy import tuplelist, GRB, Model, quicksum
from pprint import pprint
from mod.env.amod import Amod

# from mod.env.ml import get_dict_cars_per_attribute


def get_dict_cars_per_attribute(cars, level=0):

    dict_cars_per_attribute = defaultdict(list)

    for c in cars:
        dict_cars_per_attribute[c.attribute(level)].append(c)

    return dict_cars_per_attribute


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


def myopic(env, trips, time_step, charge=True):

    level = 0
    agg_level = 3
    # ##################################################################
    # SORT CARS ########################################################
    # ##################################################################

    # How many cars per attribute
    cars_with_attribute = defaultdict(list)

    # Which positions are surrounding each car position
    dict_car_position_neighbors = dict()

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
        cars_with_attribute[car.attribute(level)].append(car)

        # Was this position already processed?
        car_pos_id = car.point.id_level(level)
        if car_pos_id not in dict_car_position_neighbors:

            # Get zones around current car regions
            nearby_zones = nw.get_neighbor_zones(
                car.point, env.config.pickup_zone_range, env.zones
            )

            # Update set of points cars can reach
            rechable_points.update(nearby_zones)

            dict_car_position_neighbors[car_pos_id] = nearby_zones

    # ##################################################################
    # SORT TRIPS #######################################################
    # ##################################################################

    # Trip ods
    trip_ods = [(t.o.id_level(level), t.d.id_level(level)) for t in trips]

    #  Dictionary of #trips per trip attribute,i.e., (o.id, d.id)
    trips_with_attribute = defaultdict(list)

    # Trips that cannot be picked up
    rejected = list()
    for t in trips:
        o, d = t.o.id_level(level), t.d.id_level(level)

        if o in rechable_points:
            trips_with_attribute[(o, d)].append(t)

        # If no vehicle can reach the trip, it is immediately rejected
        else:
            rejected.append(t)

    # ##################################################################
    # VARIABLES ########################################################
    # ##################################################################

    car_attributes = cars_with_attribute.keys()

    x_stay = [
        (Amod.TRIP_STAY_DECISION,) + c + (c[0],) + (c[0],)
        for c in car_attributes
    ]

    x_pickup = [
        (Amod.TRIP_STAY_DECISION,) + c + (o,) + (d,)
        for o, d in trip_ods
        for c in car_attributes
        if o in dict_car_position_neighbors[c[0]]
    ]

    x_recharge = [
        (Amod.RECHARGE_REBALANCE_DECISION,) + c + (c[0],) + (c[0],)
        for c in car_attributes
    ]

    x_rebalance = [
        (Amod.RECHARGE_REBALANCE_DECISION,) + c + (c[0],) + (z,)
        for c in car_attributes
        for z in dict_car_position_neighbors[c[0]]
        if c[0] != z
    ]

    # Enumerated list of decisions
    all_decisions = tuplelist(
        set(x_pickup + x_stay + x_rebalance + x_recharge)
    )

    # print("REBALANCE (action, point, level, o, d)")
    # pprint(x_rebalance)

    # print("STAY (action, point, level, o, d)")
    # pprint(x_stay)

    # print("PICKUP (action, point, level, o, d)")
    # pprint(x_pickup)

    # print("RECHARGE (action, point, level, o, d)")
    # pprint(x_recharge)

    # print(f"#Pickup: {len(x_pickup)}"
    #       f" - #Stay: {len(x_stay)}"
    #       f" - #Recharge: {len(x_recharge)}"
    #       f" - #Rebalance: {len(x_rebalance)}"
    #       f" - #Total: {len(all_decisions)}")

    # ##################################################################
    # MODEL ############################################################
    # ##################################################################

    m = Model("assignment")

    # if log_path:
    #         m.Params.LogFile = "{}/region_centers_{}.log".format(
    #             log_path, max_delay
    #         )

    #         m.Params.ResultFile = "{}/region_centers_{}.lp".format(
    #             log_path, max_delay
    #         )

    # How many vehice
    x_var = m.addVars(all_decisions, name="x")

    # The objective is to minimize the total pay costs

    value_func = quicksum(
        (env.get_value(time_step, d, level=agg_level) * x_var[d])
        for d in x_var
    )

    cost_func = quicksum(
        env.cost_func(d[0], d[3], d[4]) * x_var[d] for d in x_var
    )

    m.setObjective(cost_func + value_func, GRB.MAXIMIZE)

    # Car flow conservation
    flow_cars = m.addConstrs(
        (
            x_var.sum("*", point, level, "*", "*")
            == len(cars_with_attribute[(point, level)])
            for point, level in car_attributes
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

    # flow_trips = m.addConstrs(
    #     (x_var.sum(Amod.TRIP_STAY_DECISION, point, level, o, d) <= len(
    #         trips_with_attribute[(o, d)])
    #      for o, d in trips_with_attribute.keys()
    #      for point, level in car_attributes
    #      if o in dict_car_position_neighbors[point]),
    #     "TRIP_FLOW",
    # )

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
        # Save model
    # m.write('myopic.lp')
    # Disables all logging (file and console)
    m.setParam("OutputFlag", 0)
    # m.setParam('logFile', "")
    # m.setParam('OutputConsole', 0)

    # Optimize
    m.optimize()

    if m.status == GRB.Status.UNBOUNDED:
        print("The model cannot be solved because it is unbounded")

    if m.status == GRB.Status.OPTIMAL:

        # Get decision tuples associated to positive values
        best_decisions = extract_decisions(x_var)

        # Dictionary car atribute (pos, battery) -> Shadow price
        duals = extract_duals(flow_cars, car_attributes)

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

        # exit(0)
    if (
        m.status != GRB.Status.INF_OR_UNBD
        and m.status != GRB.Status.INFEASIBLE
    ):
        print("Optimization was stopped with status %d" % m.status)
        # exit(0)
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
