import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random

# Adding project folder to import modules
root = os.getcwd().replace("\\", "/")
sys.path.append(root)

from mod.env.amod import Amod
from mod.env.visual import StepLog, EpisodeLog
from mod.env.config import ConfigStandard, NY_TRIPS_EXCERPT_DAY
from mod.env.match import adp, myopic, fcfs
from mod.env.trip import (
    get_random_trips,
    get_trip_count_step,
    get_trips_random_ods,
)
import mod.env.network as nw
from pprint import pprint


if __name__ == "__main__":

    # -----------------------------------------------------------------#
    # Amod environment #################################################
    # -----------------------------------------------------------------#

    config = ConfigStandard()
    config.update(
        {
            ConfigStandard.FLEET_SIZE: 30,
            ConfigStandard.ROWS: 40,
            ConfigStandard.COLS: 40,
            ConfigStandard.BATTERY_LEVELS: 10,
            ConfigStandard.PICKUP_ZONE_RANGE: 2,
            ConfigStandard.AGGREGATION_LEVELS: 4,
            ConfigStandard.INCUMBENT_AGGREGATION_LEVEL: 2,
            ConfigStandard.ORIGIN_CENTERS: 4,
            ConfigStandard.ORIGIN_CENTER_ZONE_SIZE: 3,
        }
    )

    # -----------------------------------------------------------------#
    # Episodes #########################################################
    # -----------------------------------------------------------------#
    episodes = 100
    episodeLog = EpisodeLog(config=config)
    amod = Amod(config)

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

    # Get demand pattern from NY city
    step_trip_count_15 = get_trip_count_step(
        NY_TRIPS_EXCERPT_DAY, step=15, multiply_for=0.167
    )

    print(
        f"### DEMAND ###"
        f" - min: {min(step_trip_count_15)}"
        f" - max: {max(step_trip_count_15)}"
    )

    # Loop all episodes, pick up trips, and learn where they are
    for n in range(episodeLog.n, episodes):

        # Create all episode trips
        step_trip_list = get_trips_random_ods(
            amod.points,
            step_trip_count_15,
            offset_start=amod.config.offset_repositioning,
            offset_end=amod.config.offset_termination,
            origins=origins,
        )

        # Start saving data of each step in the environment
        step_log = StepLog(amod)

        # Resetting environment
        amod.reset()

        # Iterate through all steps and match requests to cars
        for step, trips in enumerate(step_trip_list):

            revenue, serviced, rejected = adp(
                amod, trips, step, aggregation="weighted"
            )

            # ---------------------------------------------------------#
            # Update log with iteration ################################
            # ---------------------------------------------------------#

            step_log.add_record(revenue, serviced, rejected)

            # Show time step statistics
            # step_log.show_info()

            # What each vehicle is doing?
            # amod.print_fleet_stats()

        # -------------------------------------------------------------#
        # Compute episode info #########################################
        # -------------------------------------------------------------#
        print(
            f"####### "
            f"[Episode {n:>5}] "
            f"- {episodeLog.last_episode_stats()} "
            f"#######"
        )
        episodeLog.compute_episode(
            step_log, weights=amod.get_weights(), progress=True
        )

    episodeLog.compute_learning()
