import itertools
from collections import defaultdict
from mod.env.car import HiredCar, VirtualCar
import random

# Decision codes

TRIP_DECISION = 0

STAY_DECISION = 1

RECHARGE_DECISION = 2

REBALANCE_DECISION = 3

# Drive back to origin
RETURN_DECISION = 4

label_dict = {
    TRIP_DECISION: "TRIP",
    STAY_DECISION: "STAY",
    RECHARGE_DECISION: "CHAR",
    REBALANCE_DECISION: "REBA",
    RETURN_DECISION: "BACK",
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
    """Stay in current position"""
    return (
        (STAY_DECISION,)
        + car.attribute
        + (car.point.id,)
        + (car.point.id,)
        + (DISCARD,)
    )


def recharge_decision(car):
    """Stay in current position recharging"""
    return (
        (RECHARGE_DECISION,)
        + car.attribute
        + (car.point.id,)
        + (car.point.id,)
        + (DISCARD,)
    )


def return_decision(car):
    """Back to car origin"""
    return (
        (RETURN_DECISION,)
        + car.attribute
        + (car.point.id,)
        + (car.origin.id,)
        + (DISCARD,)
    )


def rebalance_decision(car, neighbor):
    """Rebalance car to neighbor"""
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
        # Car cannot rebalance since it cannot go back to origin in time
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


def get_decisions(env, trips, min_battery_level=None):
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
    decisions_return = set()
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

        # Vehicle knows nothing about future states, hence it rebalances
        if not env.config.myopic:
            # Rebalancing ################################################ #

            neighbors = env.attribute_rebalance[car.point.id]

            d_rebalance = rebalance_decisions(car, neighbors, env)

            # Return trips are always available
            if isinstance(car, HiredCar):

                # If car has moved from depot
                if car.point.id != car.depot.id:

                    # d_return_trip = rebalance_decision(car, car.depot.id)
                    # d_rebalance.add(d_return_trip)

                    travel_time = env.get_travel_time_od(
                        car.point, car.depot, unit="min"
                    )

                    # Car must return if when contract is about to end
                    # (1 step offset given)
                    return_trip_duration = (
                        travel_time + env.config.time_increment
                    )

                    if car.contract_duration <= return_trip_duration:
                        d_return = return_decision(car)
                        d_rebalance.add(d_return)
                        decisions_return.add(d_return)

            if not d_rebalance:
                # Remove from tabu if not empty.
                # Avoid cars are corned indefinitely
                if car.tabu:
                    car.tabu.popleft()

            # Vehicles can stay idle for a maximum number of steps.
            # If they surpass this number, they can rebalance to farther
            # areas.
            if env.config.max_idle_step_count:

                # Car can rebalance to farther locations besides the
                # closest
                if car.idle_step_count >= env.config.max_idle_step_count:
                    farther = env.get_zone_neighbors(
                        car.point.id, explore=True
                    )

                    # print(f"farther: {farther} - d_rebalance: {d_rebalance}")
                    d_rebalance.update(rebalance_decisions(car, farther, env))
                    # d_rebalance = d_rebalance | farther

            # print(f"farther: {d_rebalance}")

            decisions.update(d_rebalance)

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

    return decisions, decisions_return, decision_class
