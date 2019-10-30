import itertools
from collections import defaultdict
from mod.env.car import HiredCar, VirtualCar

# Decision codes

TRIP_DECISION = 0

STAY_DECISION = 1

RECHARGE_DECISION = 2

REBALANCE_DECISION = 3

HIRE_DECISION = 4

label_dict = {
    TRIP_DECISION: "TRIP",
    STAY_DECISION: "STAY",
    RECHARGE_DECISION: "CHAR",
    REBALANCE_DECISION: "REBA",
    HIRE_DECISION: "HIRE",
}


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

DISCARD = "-"


def stay_decision(car):
    return (
        (STAY_DECISION,)
        + car.attribute
        + (car.point.id,)
        + (car.point.id,)
        + (DISCARD,)
    )


def recharge_decision(car):
    return (
        (RECHARGE_DECISION,)
        + car.attribute
        + (car.point.id,)
        + (car.point.id,)
        + (DISCARD,)
    )


def rebalance_decision(car, neighbor):
    return (
        (REBALANCE_DECISION,)
        + car.attribute
        + (car.point.id,)
        + (neighbor,)
        + (DISCARD,)
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

        d_rebalance = rebalance_decision(car, t)

        # Cars know what decisions they generated
        rebalance_decisions.add(d_rebalance)

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


def get_virtual_decisions(trips):
    """Create virtual trip and stay decisions.
    When the system must fullfil all orders, virtual cars can be
    used to pretend customers were serviced.
    
    Parameters
    ----------
    trips : list
        Trip list from where virtual car origins are drawn.
    
    Returns
    -------
    list
        Virtual decisions
    """

    decisions = set()

    for trip in trips:

        virtual_car = VirtualCar(trip.o)
        # Stay
        decisions.add(stay_decision(virtual_car))

        # Pickup
        decisions.add(trip_decision(virtual_car, trip))

    return decisions


def get_decisions(env, trips, min_battery_level=None, myopic=False):
    """Get list of decision tuples.

    Parameters
    ----------
    env : Amod object
        Amod environment
    trips : list
        Trips placed in the current time step
    min_battery_level : int, optional
        Create recharging decisions with car battery level is lower, by
        default None

    Returns
    -------
    (list, dict(list))
        List of all decisions
        List of trip decisions per class
    """

    decisions = set()
    decision_class = defaultdict(list)
    # Which positions are surrounding each car position
    attribute_rebalance = dict()

    # ##################################################################
    # SORT CARS ########################################################
    # ##################################################################

    for car in itertools.chain(env.available, env.available_hired):
        # Stay ####################################################### #
        d_stay = stay_decision(car)
        decisions.add(d_stay)

        # Vehicle knows nothing about future states, hence no rebalancing
        if not myopic:
            # Rebalancing ################################################ #
            try:
                attribute_rebalance[car.point.id] = attribute_rebalance[
                    car.point.id
                ]  - set(car.tabu)
            except:
                attribute_rebalance[car.point.id] = env.get_zone_neighbors(
                    car.point.id
                )  - set(car.tabu)

            # attribute_rebalance[car.point.id] = (
            #     attribute_rebalance.get(car.point.id, set())
            #     | env.get_zone_neighbors(car.point.id)) - set(car.tabu)
            # try:
            #     # Get rebalance targets if previously defined (car with
            #     # the same attribute)
            #     rebalance_targets = attribute_rebalance[car.point.id] - set(car.tabu)
            #     attribute_rebalance[car.point.id] = rebalance_targets - set(car.tabu)

            # except:
            #     # Get region center neighbors
            #     rebalance_targets = env.get_zone_neighbors(car.point.id)
            #     attribute_rebalance[car.point.id] = rebalance_targets - set(car.tabu)
            #     # All points a car can rebalance to from its corrent point
            #     # attribute_rebalance[car.point.id] = rebalance_targets

        # Recharge ################################################### #
        if min_battery_level and car.battery_level < min_battery_level:
            decisions.add(recharge_decision(car))

        for trip in trips:

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
            travel_time = env.get_travel_time_od(car.point, trip.o, unit="min")

            # Can the car reach the trip origin?
            if travel_time <= min(
                env.config.matching_delay, trip.max_delay_from_placement
            ):

                # Setup decisions
                d = trip_decision(car, trip)
                decisions.add(d)

                # Car can fulfill the shortest delay
                if travel_time <= min(
                    env.config.matching_delay, trip.min_delay_from_placement
                ):

                    # ---------------------------------------- #
                    # DECISIONS ASSOCIATED TO EACH SQ CLASS ## #
                    # ---------------------------------------- #

                    # There might be repeated decisions
                    # associated to the same class since
                    # several trips can depart from the same
                    # place.

                    decision_class[trip.sq_class].append(d)

    if not myopic:

        # Cars only rebalance to places where
        for car in itertools.chain(env.available, env.available_hired):

            # Vehicles in the same location cannot rebalance to any
            # location in their tabu lists
            d_rebalance = rebalance_decisions(
                car, attribute_rebalance[car.point.id], env
            )

            if not d_rebalance:
                # Remove from tabu if not empty.
                # Avoid cars are corned indefinitely
                if car.tabu:
                    car.tabu.popleft()

            # Vehicles can stay idle for a maximum number of steps.
            # If they surpass this number, they can rebalance to farther
            # areas.
            if env.config.max_idle_step_count:

                if car.idle_step_count >= env.config.max_idle_step_count:
                    farther = env.get_zone_neighbors(car.point.id, explore=True)

                    # print(f"farther: {farther} - d_rebalance: {d_rebalance}")
                    d_rebalance = rebalance_decisions(
                        car, farther, env
                    )
                    # d_rebalance = d_rebalance | farther

            
            
            # print(f"farther: {d_rebalance}")

            decisions.update(d_rebalance)

            # print(f"{car} - {len(d_rebalance)} - {d_rebalance}")

            # print(car, d_rebalance)

            # car.decisions.update(d_rebalance)

            # if not car.decisions:
            #     d_stay = stay_decision(car)
            #     decisions.add(d_stay)
            #     try:
            #         removed = car.tabu.popleft()
            #     except:
            #         removed = "-empty-"
            #     # print(f"{car} has no decisions. {d_stay} added. removed from tabu = {removed}")
            #     # Start removing from tabu
            #     # car.tabu.popleft()
            # elif len(d_rebalance) == 1:
            #     try:
            #         removed = car.tabu.popleft()
            #     except:
            #         removed = "-empty-"

            # d_stay = stay_decision(car)
            # decisions.add(d_stay)
            # print(f"{car} has only 1 rebalance decisions={d_rebalance}. Removed tabu: {removed}")

            # car.decisions = set()
    return decisions, decision_class
