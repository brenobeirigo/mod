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
            # Fleet
            ConfigNetwork.FLEET_SIZE: 1,
            ConfigNetwork.BATTERY_LEVELS: 20,
            # Time - Increment (min)
            ConfigNetwork.TIME_INCREMENT: 1,
            ConfigNetwork.OFFSET_REPOSIONING: 15,
            ConfigNetwork.OFFSET_TERMINATION: 30,
            # NETWORK ##################################################
            # Region centers are created in steps of how much time?
            ConfigNetwork.STEP_SECONDS: 60,
            # Demand spawn from how many centers?
            ConfigNetwork.ORIGIN_CENTERS: 1,
            # Cars rebalance to up to #region centers
            ConfigNetwork.N_CLOSEST_NEIGHBORS: 4,
            # Cars can access locations within region centers
            # established in which neighborhood level?
            ConfigNetwork.NEIGHBORHOOD_LEVEL: 4,
            ConfigNetwork.AGGREGATION_LEVELS: 6,
            ConfigNetwork.SPEED: 30,
        }
    )

    # ################################################################ #
    # Slice demand ################################################### #
    # ################################################################ #

    # What is the level covered by origin area?
    # E.g., levels 1, 2, 3 = 60, 120, 180
    # if level_origins = 3
    level_origins = 1

    # Data correspond to 1 day NY demand
    total_hours = 24
    earliest_hour = 0
    resize_factor = 1
    max_steps = int(total_hours * 60 / config.time_increment)
    earliest_step_min = int(earliest_hour * 60 / config.time_increment)

    # ---------------------------------------------------------------- #
    # Episodes ####################################################### #
    # ---------------------------------------------------------------- #
    episodes = 270
    episodeLog = EpisodeLog(config=config)
    amod = AmodNetwork(config)

    try:
        # Load last episode
        progress = episodeLog.load_progress()
        amod.load_progress(progress)

    except Exception as e:
        print(f"No previous episodes were saved {e}.")

    # ---------------------------------------------------------------- #
    # Trips ########################################################## #
    # ---------------------------------------------------------------- #

    try:
        origin_ids = episodeLog.load_ods()
        origins = [amod.points[p] for p in origin_ids]
        print(f"\n{len(origins)} origins loaded.")

    except:

        # Create random centers from where trips come from
        # TODO choose level to query origins
        origins = nw.query_centers(
            amod.points,
            amod.config.origin_centers,
            amod.config.get_step_level(level_origins),
        )

        print(f"\nSaving {len(origins)} origins.")
        episodeLog.save_origins([o.id for o in origins])

    # Get demand pattern from NY city
    step_trip_count = get_trip_count_step(
        NY_TRIPS_EXCERPT_DAY,
        step=config.time_increment,
        multiply_for=resize_factor,
        earliest_step=earliest_step_min,
        max_steps=max_steps,
    )

    print(
        f"### DEMAND ###"
        f" - min: {min(step_trip_count)}"
        f" - max: {max(step_trip_count)}"
    )

    destinations = nw.query_centers(
        amod.points,
        amod.config.origin_centers,
        amod.config.get_step_level(level_origins),
    )

    # ---------------------------------------------------------------- #
    # Experiment ##################################################### #
    # ---------------------------------------------------------------- #

    # Loop all episodes, pick up trips, and learn where they are
    for n in range(episodeLog.n, episodes):

        # Create all episode trips
        step_trip_list = get_trips_random_ods(
            amod.points,
            step_trip_count,
            offset_start=amod.config.offset_repositioning,
            offset_end=amod.config.offset_termination,
            origins=origins,
            destinations=destinations,
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
                step + 1,
                neighborhood_level=config.neighborhood_level,
                n_neighbors=config.n_neighbors,
                # agg_level=amod.config.incumbent_aggregation_level,
            )

            # ---------------------------------------------------------#
            # Update log with iteration ################################
            # ---------------------------------------------------------#

            step_log.add_record(revenue, serviced, rejected)

            step_log.plot_fleet_operation()

            # print(
            #     "##########################################################################################"
            # )
            # # Show time step statistics
            # step_log.show_info()

            # # What each vehicle is doing?
            # amod.print_fleet_stats()

        # amod.print_car_traces()

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
