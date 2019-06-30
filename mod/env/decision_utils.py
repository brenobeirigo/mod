from pprint import pprint
from gurobipy import tuplelist
from collections import defaultdict

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

decision_list = [
    TRIP_DECISION,
    RECHARGE_DECISION,
    REBALANCE_DECISION,
    STAY_DECISION,
]


def stay_decision(car):
    return (
        (STAY_DECISION,)
        + (car.point.id, car.battery_level)
        + (car.point.id,)
        + (car.point.id,)
        + (car.type,)
        + (car.contract_duration,)
    )


def end_contract_decision(car):
    return (
        (END_CONTRACT_DECISION,)
        + (car.point.id, car.battery_level)
        + (car.point.id,)
        + (car.start_end_point.id,)
        + (car.type,)
        + (car.contract_duration,)
    )


def recharge_decision(car):
    return (
        (RECHARGE_DECISION,)
        + (car.point.id, car.battery_level)
        + (car.point.id,)
        + (car.point.id,)
        + (car.type,)
        + (car.contract_duration,)
    )


def rebalance_decision(car, neighbor, hire=False):
    return (
        ((HIRE_DECISION if hire else REBALANCE_DECISION),)
        + (car.point.id, car.battery_level)
        + (car.point.id,)
        + (neighbor,)
        + (car.type,)
        + (car.contract_duration,)
    )


def rebalance_decisions(car, targets, hire=False):
    rebalance_decisions = set()
    for t in targets:
        rebalance_decisions.add(rebalance_decision(car, t, hire=hire))
    return rebalance_decisions


def hire_decisions(car, targets, hire=False):
    hire_decisions = set()
    for t in targets:
        hire_decisions.add(rebalance_decision(car, t, hire=hire))
    return hire_decisions


def trip_decision(car, trip, hire=True):
    return (
        (TRIP_DECISION,)
        + (car.point.id, car.battery_level)
        + (trip.o.id,)
        + (trip.d.id,)
        + (car.type,)
        + (car.contract_duration,)
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


def get_decision_set_classed(
    cars,
    hired_cars,
    level_id_cars_dict,
    level_id_trips_dict,
    rebalance_targets_dict,
    max_battery_level=None,
):

    decisions = defaultdict(set)
    decision_class = defaultdict(list)

    for car in cars:
        # Stay ####################################################### #
        decisions[car.type].add(stay_decision(car))

        # Rebalancing ################################################ #
        rebalance_targets = rebalance_targets_dict[car.type][car.point.id]
        decisions[car.type].update(rebalance_decisions(car, rebalance_targets))

        # Recharge ################################################### #
        if max_battery_level and car.battery_level < max_battery_level:
            decisions[car.type].add(recharge_decision(car))

    for car in hired_cars:

        # if car.started_contract:

        # Rebalancing ############################################ #
        rebalance_targets = rebalance_targets_dict[car.type][car.point.id]
        decisions[car.type].update(rebalance_decisions(car, rebalance_targets))

        # Stay ################################################### #
        decisions[car.type].add(stay_decision(car))

        # Recharge ############################################### #
        if max_battery_level and car.battery_level < max_battery_level:
            decisions[car.type].add(recharge_decision(car))

        # else:

        #     # Hire new vehicles
        #     decisions[car.type].update(
        #         hire_decisions(car, rebalance_targets, hire=True)
        #     )

        #     # End contract
        #     decisions[car.type].add(end_contract_decision(car))

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

            for car_type in level_id_cars_dict:

                # Check if there are cars to match to trips. Hired
                # vehicles always match trips.
                if (level, level_id) in level_id_cars_dict[car_type]:

                    car_list = level_id_cars_dict[car_type][(level, level_id)]

                    for car in car_list:

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

                            decisions[car.type].add(d)
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
