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
from mod.env.match import fcfs, myopic, adp
import mod.env.network as nw
from mod.env.trip import (
    get_random_trips,
    get_trip_count_step,
    get_trips_random_ods,
)
from mod.env.visual import StepLog, EpisodeLog


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
        # print(
        #     f"### Time step: {time_step+1:>3}"
        #     f" ### Profit: {revenue:>10.2f}"
        #     f" ### Service level: {len(serviced)/len(rejected):>7.2%}"
        #     f" ### Trips: {len(trips):>3}"
        #     " ###"
        # )

        step_log.add_record(revenue, serviced, rejected)

        # Show car stats
        if show_stats:
            amod.print_current_stats()
    print("Total duration:", total_duration)
    return step_log


if __name__ == "__main__":

    c = ConfigStandard()
    c.update(
        {
            ConfigStandard.FLEET_SIZE: 250,
            ConfigStandard.ROWS: 32,
            ConfigStandard.COLS: 32,
            ConfigStandard.BATTERY_LEVELS: 20,
            ConfigStandard.PICKUP_ZONE_RANGE: 2,
            ConfigStandard.AGGREGATION_LEVELS: 3,
            ConfigStandard.INCUMBENT_AGGREGATION_LEVEL: 2,
            ConfigStandard.ORIGIN_CENTERS: 4,
            ConfigStandard.ORIGIN_CENTER_ZONE_SIZE: 3,
        }
    )

    # c.update({"FLEET_SIZE": 10, "ROWS": 100, "COLS": 100})
    amod = Amod(c)
    episodeLog = EpisodeLog(config=c)
    amod = Amod(c)

    try:
        # Load last episode
        values, counts = episodeLog.load()
        amod.values = values
        amod.count = counts

    except Exception as e:
        print(f"No previous episodes were saved {e}.")

    # -----------------------------------------------------------------#
    # Trips ############################################################
    # -----------------------------------------------------------------#

    try:
        origin_ids = episodeLog.load_origins()
        origins = [amod.points[p] for p in origin_ids]
        print(f"\n{len(origins)} origins loaded.")

    except:

        # Create random centers from where trips come from
        origins = nw.get_demand_origin_centers(
            amod.points,
            amod.zones,
            amod.config.origin_centers,
            amod.config.origin_center_zone_size,
        )

        print(f"\nSaving {len(origins)} origins.")
        episodeLog.save_origins([o.id for o in origins])

    step_trip_count_15 = get_trip_count_step(
        NY_TRIPS_EXCERPT_DAY, step=15, multiply_for=0.126
    )

    # Creating all trips based on NY demand
    step_trip_list = get_trips_random_ods(
        amod.points,
        step_trip_count_15,
        offset_start=amod.config.offset_repositioning,
        offset_end=amod.config.offset_termination,
    )

    matching_functions = {"MYOPIC": myopic, "ADP": adp, "FCFS": fcfs}

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
