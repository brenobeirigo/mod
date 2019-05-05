""" Ride matching methods """

from collections import defaultdict
import mod.env.network as nw

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