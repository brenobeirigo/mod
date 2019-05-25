import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random

# Adding project folder to import modules
root = os.getcwd().replace("\\", "/")
sys.path.append(root)

from mod.env.amod import Amod, AmodNetwork
from mod.env.visual import StepLog, EpisodeLog
from mod.env.config import ConfigNetwork, NY_TRIPS_EXCERPT_DAY
from mod.env.match import adp_network, myopic, fcfs
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

    config = ConfigNetwork()
    config.update(
        {
            ConfigNetwork.FLEET_SIZE: 20,
            ConfigNetwork.ROWS: 32,
            ConfigNetwork.COLS: 32,
            ConfigNetwork.BATTERY_LEVELS: 20,
            ConfigNetwork.PICKUP_ZONE_RANGE: 2,
            ConfigNetwork.AGGREGATION_LEVELS: 4,
            ConfigNetwork.INCUMBENT_AGGREGATION_LEVEL: 2,
            ConfigNetwork.ORIGIN_CENTERS: 5,
            ConfigNetwork.ORIGIN_CENTER_ZONE_SIZE: 5,
        }
    )

    # -----------------------------------------------------------------#
    # Episodes #########################################################
    # -----------------------------------------------------------------#
    episodes = 250
    episodeLog = EpisodeLog(config=config)
    amod = AmodNetwork(config)

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
        # TODO choose level to query origins
        origins = nw.query_demand_origin_centers(
            amod.points,
            amod.config.origin_centers,
            amod.config.get_step_level(2),
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

        # Start saving data of each step in the enadp_networkvironment
        step_log = StepLog(amod)

        # Resetting environment
        amod.reset()

        # Iterate through all steps and match requests to cars
        for step, trips in enumerate(step_trip_list):

            revenue, serviced, rejected = adp_network(
                amod,
                trips,
                step,
                # agg_level=amod.config.incumbent_aggregation_level,
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
