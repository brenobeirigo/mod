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
from mod.env.match import adp3, myopic, fcfs
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
            ConfigNetwork.FLEET_SIZE: 120,
            ConfigNetwork.BATTERY_LEVELS: 20,
            ConfigNetwork.PICKUP_ZONE_RANGE: 2,
            ConfigNetwork.AGGREGATION_LEVELS: 4,
            ConfigNetwork.INCUMBENT_AGGREGATION_LEVEL: 2,
            ConfigNetwork.ORIGIN_CENTERS: 4,
            ConfigNetwork.ORIGIN_CENTER_ZONE_SIZE: 3,
            ConfigNetwork.TIME_INCREMENT: 1,
            ConfigNetwork.STEP_SECONDS: 30,
            ConfigNetwork.N_CLOSEST_NEIGHBORS: 4,
            ConfigNetwork.NEIGHBORHOOD_LEVEL: 1,
            # ConfigNetwork.OFFSET_REPOSIONING: 15,
            # ConfigNetwork.OFFSET_TERMINATION: 30,
        }
    )

    hours = 24
    earliest_step = 0

    # -----------------------------------------------------------------#
    # Episodes #########################################################
    # -----------------------------------------------------------------#
    episodes = 250
    episodeLog = EpisodeLog(config=config)
    amod = AmodNetwork(config)

    try:
        # Load last episode
        values, counts, transient_bias, variance_g, step_size_func, lambda_stepsize, aggregation_bias = (
            episodeLog.load()
        )
        amod.values = values
        amod.count = counts
        amod.transient_bias = transient_bias
        amod.variance_g = variance_g
        amod.step_size_func = step_size_func
        amod.lambda_stepsize = lambda_stepsize
        amod.aggregation_bias = aggregation_bias

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
        NY_TRIPS_EXCERPT_DAY,
        step=config.time_increment,
        multiply_for=0.167,
        earliest_step=earliest_step,
        max_steps=int(hours * 60 / config.time_increment),
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

            revenue, serviced, rejected = adp3(
                amod,
                trips,
                step + 1,
                neighborhood_level=config.neighborhood_level,
                n_neighbors=config.n_neighbors
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
