import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
import time

# Adding project folder to import modules
root = os.getcwd().replace("\\", "/")
sys.path.append(root)

from mod.env.amod import Amod
from mod.env.config import Config, ConfigStandard, NY_TRIPS_EXCERPT_DAY
from mod.env.match import fcfs, myopic
from mod.env.trip import (
    get_random_trips,
    get_trip_count_step,
    get_trips_random_ods,
)
from mod.env.visual import StepLog


def test_matching_methods(
    amod, matching_func, step_trip_list=None, show_stats=False
):

    step_log = StepLog(amod)
    total_duration = 0
    for time_step in range(amod.config.time_steps):

        if step_trip_list:
            trips = step_trip_list[time_step]
        else:
            # Get random set of trips
            trips = get_random_trips(
                amod.points,
                time_step,
                amod.config.min_trips,
                amod.config.max_trips,
            )

        # Match cars and trips and get:
        # - revenue round
        # - list serviced requests
        # - list rejected requests
        t1 = time.time()
        revenue, serviced, rejected = matching_func(amod, trips, time_step)
        duration = time.time() - t1
        # print('\n###', duration)
        total_duration += duration
        print(
            f"### Time step: {time_step+1:>3}"
            f" ### Profit: {revenue:>10.2f}"
            f" ### Service level: {len(serviced)/max(1,len(rejected)):>7.2%}"
            f" ### Trips: {len(trips):>3}"
            " ###"
        )

        step_log.add_record(revenue, serviced, rejected)

        # Show car stats
        if show_stats:
            amod.print_current_stats()
    print("Total duration:", total_duration)
    return step_log


if __name__ == "__main__":

    c = ConfigStandard()
    c.update({"FLEET_SIZE": 10, "ROWS": 100, "COLS": 100})
    amod = Amod(c)

    step_trip_count_15 = get_trip_count_step(
        NY_TRIPS_EXCERPT_DAY, step=15, multiply_for=1
    )

    # Creating all trips based on NY demand
    step_trip_list = get_trips_random_ods(
        amod.points,
        step_trip_count_15,
        offset_start=amod.config.offset_repositioning,
        offset_end=amod.config.offset_termination,
    )

    matching_functions = {"EXACT MYOPIC": myopic, "FCFS": fcfs}

    for label, func in matching_functions.items():

        # Get the execution history of the function
        step_log = test_matching_methods(
            deepcopy(amod),
            func,
            step_trip_list=deepcopy(step_trip_list),
            show_stats=False,
        )

        step_log.plot_fleet_status()
        step_log.plot_service_status()
        step_log.overall_log(label=label)
        print("travel_time", amod.get_travel_time.cache_info())
        print("get_distance", amod.get_distance.cache_info())
        print("cost_func", amod.cost_func.cache_info())
