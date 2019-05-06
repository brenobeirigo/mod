from collections import defaultdict, Counter
import mod.env.network as nw
from gurobipy import *
from pprint import pprint
from mod.env.amod import Amod
#from mod.env.ml import get_dict_cars_per_attribute


def myopic(env, trips, time_step, charge = True):

    level = 0
    
    # Idle car attributes
    cars_idle = env.cars_idle()
    
    # Trip ods
    trip_ods = [(t.o.id_level(level), t.d.id_level(level)) for t in trips]

    # How many cars per attribute (pos, battery)
    car_count_attribute = Counter([c.attribute(level) for c in cars_idle])
    
    # How many trips per attribute (o,d)
    trip_count_attribute = Counter([t for t in trip_ods])
    
    x_stay = [
        (Amod.TRIP_STAY_DECISION,)+c+(c[0],)+(c[0],)
        for c in list(car_count_attribute.keys())
    ]

    x_pickup = [
        (Amod.TRIP_STAY_DECISION,)+c+(o,)+(d,)
        for o,d in trip_ods
        for c in list(car_count_attribute.keys())
        if o in nw.get_neighbor_zones(
            env.get_point_by_id(c[0]),
            env.config.pickup_zone_range,
            env.zones
        )
    ]

    x_recharge = [
        (Amod.RECHARGE_REBALANCE_DECISION,)+c+(c[0],)+(c[0],)
        for c in list(car_count_attribute.keys())
    ]

    x_rebalance = [
        (Amod.RECHARGE_REBALANCE_DECISION,)+c+(c[0],)+(z,)
        for c in list(car_count_attribute.keys())
        for z in nw.get_neighbor_zones(
            env.get_point_by_id(c[0]),
            env.config.pickup_zone_range,
            env.zones
        )
        if c[0] != z
    ]

    print("REBALANCE (action, point, level, o, d)")
    pprint(x_rebalance)

    print("STAY (action, point, level, o, d)")
    pprint(x_stay)

    print("PICKUP (action, point, level, o, d)")
    pprint(x_stay)

    print("RECHARGE (action, point, level, o, d)")
    pprint(x_recharge)

    print(
        f'#Pickup: {len(x_pickup)}'
        f' - #Stay: {len(x_stay)}'
        f' - #Recharge: {len(x_recharge)}'
        f' - #Rebalance: {len(x_rebalance)}'
    )
    
    # Variables
    x_var = tuplelist(set(x_pickup + x_stay + x_rebalance + x_recharge))

    print(
        f'#Pickup: {len(x_pickup)}'
        f' - #Stay: {len(x_stay)}'
        f' - #Recharge: {len(x_recharge)}'
        f' - #Rebalance: {len(x_rebalance)}'
        f' - #Total: {len(x_var)}'
    )

    # Model
    m = Model("assignment")

    # Assignment variables: x[w,s] == 1 if worker w is assigned to shift s.
    # Since an assignment model always produces integer solutions, we use
    # continuous variables and solve as an LP.
    x_var = m.addVars(x_var, name="x")

    # The objective is to minimize the total pay costs
    m.setObjective(
        quicksum(
            env.cost_func(a,o,d)*x_var[a,pos,battery,o,d]
            for a,pos,battery,o,d in x_var
            ),
            GRB.MAXIMIZE
    )

    # Car flow conservation
    flow_cars = m.addConstrs(
        (
            x_var.sum('*',o,d,'*','*') == car_count_attribute[(o, d)]
            for o,d in car_count_attribute.keys()
        ),
        "CAR_FLOW"
    )

    # Trip flow conservation
    flow_trips = m.addConstrs(
        (
            x_var.sum('*',o,d,'*','*') <= trip_count_attribute[(o, d)]
            for o,d in trip_count_attribute.keys()
        ),
        "TRIP_FLOW"
    )

    # Save model
    #m.write('myopic.lp')

    # Optimize
    m.optimize()
    status = m.status
    if status == GRB.Status.UNBOUNDED:
        print('The model cannot be solved because it is unbounded')
        exit(0)
    if status == GRB.Status.OPTIMAL:
        print('The optimal objective is %g' % m.objVal)
        exit(0)
    if status != GRB.Status.INF_OR_UNBD and status != GRB.Status.INFEASIBLE:
        print('Optimization was stopped with status %d' % status)
        exit(0)

    # do IIS
    print('The model is infeasible; computing IIS')
    m.computeIIS()
    if m.IISMinimal:
        print('IIS is minimal\n')
    else:
        print('IIS is not minimal\n')
        print('\nThe following constraint(s) cannot be satisfied:')
    for c in m.getConstrs():
        if c.IISConstr:
            print('%s' % c.constrName)

def fcfs(env, trips, time_step, charge = True):
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
        #print("Updating status all vehicles")
        car.update(time_step, time_increment=env.config.time_increment)

        # Only available vehicles can be recharged or assigned
        if car.busy:
            #print(f'car {car.id} is busy until {car.arrival_time}')
            continue

        #print(f'car {car.id} is free')

        # Recharge vehicles about to run out of power
        if charge and car.need_recharge(env.config.recharge_threshold):
            
            #print(f' - it needs RECHARGE')
            # Recharge vehicle
            cost_recharging = env.full_recharge(car)
            
            # Subtract cost of recharging
            total_contribution-=cost_recharging
        
        else:
            # Get trips car can pickup

            available_cars.add(car)

            # Get zones around current car regions
            nearby_zones = nw.get_neighbor_zones(
                car.point,
                env.config.pickup_zone_range,
                env.zones
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

                        duration_min, total_distance, best_reward = d,t,r
                        best_trip = trip
                        z_best = z
            
            if best_trip is not None:
                
                # Update car data
                car.update_trip(
                    duration_min,
                    total_distance,
                    best_reward,
                    best_trip
                )

                serviced.add(best_trip)

                total_contribution+=best_reward

                dict_zone_trips[z_best].remove(best_trip)
            

    for rejected_trip_set in dict_zone_trips.values():
        rejected.update(rejected_trip_set)

    return total_contribution, serviced, rejected