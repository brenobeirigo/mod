from pprint import pprint
from gurobipy import tuplelist
from collections import defaultdict
from mod.env.car import HiredCar

# Decision codes

# In a zoned environment with (z1, z2) cells signals:
#  - trip from z1 to z2
#  - stay in zone z1 = z2
#  - rebalance from z1 to z2
#  - recharge in zone z1 = z2

TRIP_DECISION = "TRIP"

STAY_DECISION = "STAY"

RECHARGE_DECISION = "CHAR"

REBALANCE_DECISION = "REBA"

HIRE_DECISION = "HIRE"

decision_list = [
    TRIP_DECISION,
    RECHARGE_DECISION,
    REBALANCE_DECISION,
    STAY_DECISION,
]


def stay_decision(car):
    return (
        (STAY_DECISION,)
        + car.attribute
        + (car.point.id,)
        + (car.point.id,)
        + ("_",)
    )


def recharge_decision(car):
    return (
        (RECHARGE_DECISION,)
        + car.attribute
        + (car.point.id,)
        + (car.point.id,)
        + ("_",)
    )


def rebalance_decision(car, neighbor):
    return (
        (REBALANCE_DECISION,)
        + car.attribute
        + (car.point.id,)
        + (neighbor,)
        + ("_",)
    )


def rebalance_decisions(car, targets, env):
    rebalance_decisions = set()
    for t in targets:
        # Car cannot service trip because it cannot go back
        # to origin in time
        if isinstance(car, HiredCar) and not env.can_move(
            car.point.id, car.point.id, t, car.depot.id, car.contract_duration
        ):
            continue

        rebalance_decisions.add(rebalance_decision(car, t))
    return rebalance_decisions


def trip_decision(car, trip):
    return (
        (TRIP_DECISION,)
        + car.attribute
        + (trip.o.id,)
        + (trip.d.id,)
        + (trip.sq_class,)
    )


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

    x_stay = [(STAY_DECISION,) + c + (c[0],) + (c[0],) for c in car_attributes]

    x_pickup = [
        (TRIP_DECISION,) + c + (o,) + (d,)
        for o, d in trip_ods
        for c in car_attributes
        if o in neighbors[c[0]]
    ]

    x_recharge = [
        (RECHARGE_DECISION,) + c + (c[0],) + (c[0],) for c in car_attributes
    ]

    x_rebalance = [
        (REBALANCE_DECISION,) + c + (c[0],) + (z,)
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
                (STAY_DECISION,)
                + (car.point.id, battery)
                + (car.point.id,)
                + (car.point.id,)
            )

            # Trip #####################################################
            for (o, d), trip_list in trip_ods.items():
                for trip in trip_list:
                    if o in neighbors[pos_level]:
                        decisions.add(
                            (TRIP_DECISION,)
                            + (car.point.id, battery)
                            + (trip.o.id,)
                            + (trip.d.id,)
                        )

            # Recharge #################################################
            if battery < max_battery_level:
                decisions.add(
                    (RECHARGE_DECISION,)
                    + (car.point.id, battery)
                    + (car.point.id,)
                    + (car.point.id,)
                )

            # Rebalancing ##############################################
            for neighbor in rebalance_neighbors[car.point.id]:
                decisions.add(
                    (REBALANCE_DECISION,)
                    + (car.point.id, battery)
                    + (car.point.id,)
                    + (neighbor,)
                )

    return decisions

def get_decision_set_classed3(
    env,
    trips,
    level_id_cars_dict,
    level_id_trips_dict,
    rebalance_targets_dict,
    max_battery_level=None,
):
    """Get list of decision tuples.
    
    Parameters
    ----------
    cars : list
        Owned fleet
    hired_cars : list
        Third-party fleet
    level_id_cars_dict : dict
        For each tuple (level, id), list of cars
    level_id_trips_dict : dict
        For each tuple (level, id), list of trips
    rebalance_targets_dict : dict()
        List of reachable points from each id
    max_battery_level : int, optional
        If declared, add recharge decisions, by default None
    
    Returns
    -------
    list, list
        List of all decision tuples and list of trip decisions for each
        related to each class.
    """

    decisions = set()
    decision_class = defaultdict(list)

    for car in env.available+env.available_hired:
        # Stay ####################################################### #
        decisions.add(stay_decision(car))

        # Rebalancing ################################################ #
        rebalance_targets = rebalance_targets_dict[car.point.id]
        decisions.update(rebalance_decisions(car, rebalance_targets, env))

        # Recharge ################################################### #
        if max_battery_level and car.battery_level < max_battery_level:
            decisions.add(recharge_decision(car))


        if env.config.match_neighbors():
            
            regions = env.get_zone_neighbors(
                car.point,
                level=(env.config.match_level,),
                n_neighbors=(env.config.match_max_neighbors,)
            )

            regions.add(car.point.id_level(env.config.match_level))

            # print(car, regions)

        for trip in trips:
            
            # Skip trips not in the car neighborhood
            if env.config.match_neighbors() and trip.o.id_level(env.config.match_level) not in regions:
                continue

            # Car and trip are not in the same area
            if (
                env.config.match_in_center() and
                car.point.id_level(env.config.match_level) != trip.o.id_level(env.config.match_level)
            ):
                continue

            # Car cannot service trip because it cannot go back
            # to origin in time
            if isinstance(car, HiredCar) and not env.can_move(
                car.point.id,
                trip.o.id,
                trip.d.id,
                car.depot.id,
                car.contract_duration,
            ):
                continue

            # Time to reach trip origin
            travel_time = env.get_travel_time_od(car.point, trip.o)

            # Can the car reach the trip origin?
            if travel_time <= trip.max_delay:
                
                 # Setup decisions
                d = trip_decision(car, trip)
                decisions.add(d)
                
                # Car can fulfill the shortest delay
                if travel_time <= trip.min_delay:
                    
                    # ---------------------------------------- #
                    # DECISIONS ASSOCIATED TO EACH SQ CLASS ## #
                    # ---------------------------------------- #

                    # There might be repeated decisions
                    # associated to the same class since
                    # several trips can depart from the same
                    # place.

                    decision_class[trip.sq_class].append(d)

    return decisions, decision_class

def get_decision_set_classed(
    env,
    level_id_cars_dict,
    level_id_trips_dict,
    rebalance_targets_dict,
    max_battery_level=None,
):
    """Get list of decision tuples.
    
    Parameters
    ----------
    cars : list
        Owned fleet
    hired_cars : list
        Third-party fleet
    level_id_cars_dict : dict
        For each tuple (level, id), list of cars
    level_id_trips_dict : dict
        For each tuple (level, id), list of trips
    rebalance_targets_dict : dict()
        List of reachable points from each id
    max_battery_level : int, optional
        If declared, add recharge decisions, by default None
    
    Returns
    -------
    list, list
        List of all decision tuples and list of trip decisions for each
        related to each class.
    """

    decisions = set()
    decision_class = defaultdict(list)

    for car in env.available+env.available_hired:
        # Stay ####################################################### #
        decisions.add(stay_decision(car))

        # Rebalancing ################################################ #
        rebalance_targets = rebalance_targets_dict[car.point.id]
        decisions.update(rebalance_decisions(car, rebalance_targets, env))

        # Recharge ################################################### #
        if max_battery_level and car.battery_level < max_battery_level:
            decisions.add(recharge_decision(car))

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

            # Check if there are cars to match to trips. Hired
            # vehicles always match trips.
            if (level, level_id) in level_id_cars_dict:

                car_list = level_id_cars_dict[(level, level_id)]

                for car in car_list:
                    # Car cannot service trip because it cannot go back
                    # to origin in time
                    if isinstance(car, HiredCar) and not env.can_move(
                        car.point.id,
                        trip.o.id,
                        trip.d.id,
                        car.depot.id,
                        car.contract_duration,
                    ):
                        continue

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
                        if level <= trip.sq1_level:
                            decision_class[trip.sq_class].append(d)

                        decisions.add(d)
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


def get_decision_set_classed2(
    env,
    trips,
    rebalance_targets_dict,
    max_battery_level=None,
    rebalance_reach=True,
):
    """Get list of decision tuples.
    
    Parameters
    ----------
    cars : list
        Owned fleet
    hired_cars : list
        Third-party fleet
    level_id_cars_dict : dict
        For each tuple (level, id), list of cars
    level_id_trips_dict : dict
        For each tuple (level, id), list of trips
    rebalance_targets_dict : dict()
        List of reachable points from each id
    max_battery_level : int, optional
        If declared, add recharge decisions, by default None
    
    Returns
    -------
    list, list
        List of all decision tuples and list of trip decisions for each
        related to each class.
    """

    decisions = defaultdict(set)
    decision_class = defaultdict(set)
    class_count_dict = defaultdict(int)
    od_trips_dict = defaultdict(int)

    # Cars per attribute
    attribute_cars_dict = defaultdict(list)

    # Which positions are surrounding each car position
    attribute_rebalance = set()

    for car in env.available + env.available_hired:

        # List of cars with the same attribute (pos, battery level)
        attribute_cars_dict[car.attribute].append(car)

        # ############################################################ #
        # DECISIONS ################################################## #
        # ############################################################ #

        # Stay ####################################################### #
        decisions[car.type].add(stay_decision(car))

        # Rebalancing ################################################ #

        # Cars in the same positions can rebalance to the same places
        if car.point.id not in attribute_rebalance:

            # Compute origin point
            attribute_rebalance.add(car.point.id)

            # If not None, access neighbors at reach degrees
            if env.config.rebalance_reach:
                rebalance_targets = env.get_neighbors(
                    car.point, reach=rebalance_reach
                )

            # Get region center neighbors
            else:
                rebalance_targets = env.get_zone_neighbors(
                    car.point,
                    # Tuple of region center levels cars can rebalance to
                    level=env.config.rebalance_level,
                    # Number of targets can reach at each level
                    n_neighbors=env.config.n_neighbors,
                )

            # All points a car can rebalance to, from its corrent point
            decisions[car.type].update(
                rebalance_decisions(car, rebalance_targets, env)
            )

        # Recharge ################################################### #
        if max_battery_level and car.battery_level < max_battery_level:
            decisions[car.type].add(recharge_decision(car))

        for trip in trips:

            # Trip count per class
            class_count_dict[t.sq_class] += 1

            for level in range(trip.sq2_level + 1):

                # If car and trip match in any level
                if car.point.id_level(level) == trip.o.id_level(level):

                    # Check if hired car can service trip within
                    # contract duration
                    if isinstance(car, HiredCar) and not env.can_move(
                        car.point.id,
                        trip.o.id,
                        trip.d.id,
                        car.depot.id,
                        car.contract_duration,
                    ):
                        continue

                    # Setup decisions
                    d = trip_decision(car, trip)

                    decisions[car.type].add(d)

                    # Decision belongs to service quality contraints
                    if level <= trip.sq1_level:
                        decision_class[trip.sq_class].add(d)

                    # Stop as soon as trip and car are matched
                    break

    return decisions, decision_class, attribute_cars_dict, class_count_dict
