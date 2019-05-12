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
from mod.env.config import (
    ConfigStandard,
    NY_TRIPS_EXCERPT_DAY,
    FOLDER_FLEET_PLOT,
    FOLDER_SERVICE_PLOT,
    FOLDER_EPISODE_TRACK,
    FOLDER_OUTPUT,
)
from mod.env.match import myopic
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
            ConfigStandard.FLEET_SIZE: 10,
            ConfigStandard.ROWS: 15,
            ConfigStandard.COLS: 15,
            ConfigStandard.BATTERY_LEVELS: 20,
            ConfigStandard.PICKUP_ZONE_RANGE: 2,
            ConfigStandard.AGGREGATION_LEVELS: 3,
        }
    )

    # -----------------------------------------------------------------#
    # Episodes #########################################################
    # -----------------------------------------------------------------#
    episodes = 60
    episodeLog = EpisodeLog(config=config)
    amod = Amod(config)

    try:
        # Load last episode
        values, counts = episodeLog.load()
        amod.values = values
        amod.count = counts
    except:
        print("No previous episodes were saved.")

    # -----------------------------------------------------------------#
    # Trips ############################################################
    # -----------------------------------------------------------------#

    # Create random centers from where trips come from
    n_centers = 4
    neighborhood_levels = 3
    origins = nw.get_demand_origin_centers(
        amod.points, amod.zones, n_centers, neighborhood_levels
    )

    # Get demand pattern from NY city
    step_trip_count_15 = get_trip_count_step(
        NY_TRIPS_EXCERPT_DAY, step=15, multiply_for=0.01
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

        print(
            f"####### [Episode {n:>5}]"
            f" - {episodeLog.last_episode_stats()} ########"
        )

        # Iterate through all steps and match requests to cars
        for step, trips in enumerate(step_trip_list):

            revenue, serviced, rejected = myopic(amod, trips, step)

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
        episodeLog.compute_episode(step_log, progress=True)

    episodeLog.compute_learning()
