import itertools
from collections import defaultdict
from mod.env.car import HiredCar, VirtualCar
import random
import numpy as np

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


def stay_decision_reb(car):
    """Stay in middle position"""
    return (
        (STAY_DECISION,)
        + car.attribute
        + (car.middle_point.id,)
        + (car.middle_point.id,)
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
        # Car cannot service trip because it cannot go back
        # to origin in time
        if isinstance(car, HiredCar) and not env.can_move(
            car.point.id, car.point.id, t, car.depot.id, car.contract_duration
        ):
            continue

        rebalance_decisions.add(rebalance_decision(car, t))
    return rebalance_decisions


def rebalance_decisions_thompson(car, targets, env):
    rebalance_decisions = set()
    prob_d = list()
    for t in targets:
        # Car cannot rebalance since it cannot go back to origin in time
        if isinstance(car, HiredCar) and not env.can_move(
            car.point.id, car.point.id, t, car.depot.id, car.contract_duration
        ):
            continue

        d_summary = (env.cur_step, car.point.id, t)
        a = env.beta_ab[(d_summary)]["a"]
        b = env.beta_ab[(d_summary)]["b"]
        prob = env.beta_sampler.next_sample(a, b)
        prob_d.append((prob, t, d_summary))

        # Cars know what decisions they generated
        # rebalance_decisions.add(d_rebalance)
    # TODO make it more efficient
    selected = sorted(prob_d, reverse=True, key=lambda k: (k[0],))[
        0 : env.config.max_targets
    ]

    for _, t, d_summary in selected:
        env.beta_ab[(d_summary)]["b"] += 1
        d_rebalance = rebalance_decision(car, t)
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

    # ##################################################################
    # SORT CARS ########################################################
    # ##################################################################
    from_location = defaultdict(int)
    for car in itertools.chain(env.available, env.available_hired):

        # If idle_annealing is active (i.e., it is a number),
        # cars can decide to stay only if they haven't been parked
        # for idle_annealing steps.
        # For example, if idle_annealing = 1, and a car idle_step_count
        # is also 1, it can't park anymore.
        # The idle_annealing grows with the iterations such that in the
        # end of the experiment, cars are allowed to stay parked for
        # longer periods.
        # Notice that a car idle_step_count is zeroed after servicing
        # customer or rebalancing.
        if env.config.idle_annealing is not None:

            # Can stay only when idle annealing is large.
            if car.idle_step_count < env.config.idle_annealing:
                # Stay ############################################### #
                d_stay = stay_decision(car)
                decisions.add(d_stay)
        else:
            # Stay ################################################### #
            d_stay = stay_decision(car)
            decisions.add(d_stay)

        # TODO Logic for FAVs
        if from_location[car.point.id] > 0:
            continue

        # This position will not be considered again (same for other PAVS)
        # TODO it differs for FAVs
        from_location[car.point.id] += 1

        # Rebalancing decisions (also add return decisions)
        d_rebalance = set()
        # Return ##################################################### #
        # Return trips are always available
        if isinstance(car, HiredCar):

            # If car has moved from depot
            if car.point.id != car.depot.id:

                # Car must return when contract is about to end
                return_trip_duration = env.get_travel_time_od(
                    car.point, car.depot, unit="min"
                )

                neighbors = env.attribute_rebalance[car.point.id]

                if car.contract_duration <= return_trip_duration:
                    d_return = return_decision(car)
                    d_rebalance.add(d_return)
                    decisions_return.add(d_return)
                    decisions.add(d_return)

        # Rebalancing ################################################ #
        # myopic = NO
        # reactive = NO (Rebalance decisions only in 2nd round)
        # random, train, and test = YES
        if env.config.consider_rebalance:

            neighbors = env.attribute_rebalance[car.point.id]
            if env.config.activate_thompson:
                d_rebalance = rebalance_decisions_thompson(car, neighbors, env)
            else:
                if isinstance(car, HiredCar):
                    # Car can always rebalance to its home station.
                    # Makes sense when parking costs are cheaper at 
                    # home station.
                    neighbors.add(car.depot.id)

                d_rebalance = rebalance_decisions(car, neighbors, env)

            if not d_rebalance:
                # Remove from tabu if not empty.
                # Avoid cars are corned indefinitely
                if car.tabu:
                    car.tabu.popleft()

            # TODO this is here because of a lack or rebalancing options
            # thompson selected is small 0.2
            if len(d_rebalance) == 1:
                d_stay = stay_decision(car)
                decisions.add(d_stay)

            # Vehicles can stay idle for a maximum number of steps.
            # If they surpass this number, they can rebalance to farther
            # areas.
            if env.config.max_idle_step_count:

                # Car can rebalance to farther locations besides the
                # closest after staying still for idle_step_count steps
                if car.idle_step_count >= env.config.max_idle_step_count:
                    farther = env.get_zone_neighbors(
                        car.point.id, explore=True
                    )

                    d_rebalance.update(rebalance_decisions(car, farther, env))

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
            pk_time = env.get_travel_time_od(car.point, trip.o, unit="min")

            # Discount time increment because it covers the worst case
            # scenario (user waiting since the beginning of the round)
            max_pk_time = trip.max_delay - env.config.time_increment

            # Can the car reach the trip origin?
            if pk_time <= max_pk_time + trip.tolerance:

                # Setup decisions
                d = trip_decision(car, trip)
                decisions.add(d)

                # TODO think about entire class sq penalties
                # # Car can fulfill the shortest delay
                # if travel_time <= min(
                #     env.config.matching_delay, trip.min_delay_from_placement
                # ):

                #     # ---------------------------------------- #
                #     # DECISIONS ASSOCIATED TO EACH SQ CLASS ## #
                #     # ---------------------------------------- #

                #     # There might be repeated decisions
                #     # associated to the same class since
                #     # several trips can depart from the same
                #     # place.

                #     decision_class[trip.sq_class].append(d)

    # Rebalancing vehicles can only stop to pick up new trips
    for car in itertools.chain(env.rebalancing, env.rebalancing_hired):
        # Stay ################################################### #
        d_stay = stay_decision_reb(car)
        decisions.add(d_stay)

        # Try matching trips departing from the closest middle point
        for trip in trips:

            # Car cannot service trip because it cannot go back
            # to origin in time
            if isinstance(car, HiredCar) and not env.can_move(
                car.middle_point.id,
                trip.o.id,
                trip.d.id,
                car.depot.id,
                car.contract_duration,
                delay_offset=car.elapsed,  # Time to reach middle
            ):
                continue

            # Discount time increment because it covers the worst case
            # scenario (user waiting since the beginning of the round)
            max_pk_time = trip.max_delay - env.config.time_increment

            # Time to reach trip origin
            pk_time = env.get_travel_time_od(
                car.middle_point, trip.o, unit="min"
            )

            # Can the car reach the trip origin?
            if pk_time + car.elapsed <= max_pk_time + trip.tolerance:
                # Setup decisions
                d = trip_decision(car, trip)
                decisions.add(d)

    return decisions, decisions_return, decision_class


def get_rebalancing_decisions(env, targets):
    """Stay and rebalancing decisions for the reactive rebalancing
    policy.

    Parameters
    ----------
    env : AMoD
        AMoD environment
    targets : list
        Rebalancing targets

    Returns
    -------
    set, int
        Set of all decisions (rebalancing + stay)
        Number of cars that can rebalance
    """
    decisions = set()

    # How many cars can rebalance? Hired cars can rebalance only if
    # contract limit is not surpassed.
    n_can_rebalance = 0

    # ##################################################################
    # SORT CARS ########################################################
    # ##################################################################

    for car in itertools.chain(env.available, env.available_hired):

        # Stay ####################################################### #
        d_stay = stay_decision(car)
        decisions.add(d_stay)

        if env.config.activate_thompson:
            d_rebalance = rebalance_decisions_thompson(car, targets, env)
        else:
            d_rebalance = rebalance_decisions(car, targets, env)

        if not d_rebalance:
            # Remove from tabu if not empty.
            # Avoid cars are corned indefinitely
            if car.tabu:
                car.tabu.popleft()
        else:
            # Rebalance decision was created
            n_can_rebalance += 1

        # TODO this is here because of a lack or rebalancing options
        # thompson selected is small 0.2
        if len(d_rebalance) == 1:
            d_stay = stay_decision(car)
            decisions.add(d_stay)

        # Vehicles can stay idle for a maximum number of steps.
        # If they surpass this number, they can rebalance to farther
        # areas.
        if env.config.max_idle_step_count:

            # Car can rebalance to farther locations besides the
            # closest after staying still for idle_step_count steps
            if car.idle_step_count >= env.config.max_idle_step_count:
                farther = env.get_zone_neighbors(car.point.id, explore=True)

                # print(f"farther: {farther} - d_rebalance: {d_rebalance}")
                d_rebalance.update(rebalance_decisions(car, farther, env))
                # d_rebalance = d_rebalance | farther

        # print(f"farther: {d_rebalance}")

        decisions.update(d_rebalance)

    return decisions, n_can_rebalance
