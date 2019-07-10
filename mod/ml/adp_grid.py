import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random

# Adding project folder to import modules
root = os.getcwd().replace("\\", "/")
sys.path.append(root)

from mod.env.amod.AmodGrid import AmodGrid
from mod.env.visual import StepLog, EpisodeLog
from mod.env.config import ConfigStandard, NY_TRIPS_EXCERPT_DAY
from mod.env.match import adp_grid, myopic, fcfs
from mod.env.trip import (
    get_random_trips,
    get_trip_count_step,
    get_trips_random_ods,
)
import mod.env.network as nw
from pprint import pprint
import mod.env.match as match

# Reproducibility of the experiments
random.seed(1)

if __name__ == "__main__":

    # -----------------------------------------------------------------#
    # Amod environment #################################################
    # -----------------------------------------------------------------#

    config = ConfigStandard()
    config.update(
        {
            ConfigStandard.FLEET_SIZE: 200,
            ConfigStandard.ROWS: 50,
            ConfigStandard.COLS: 50,
            ConfigStandard.BATTERY_LEVELS: 20,
            ConfigStandard.PICKUP_ZONE_RANGE: 4,
            ConfigStandard.AGGREGATION_LEVELS: 4,
            ConfigStandard.INCUMBENT_AGGREGATION_LEVEL: 2,
            ConfigStandard.ORIGIN_CENTERS: 4,
            ConfigStandard.TIME_INCREMENT: 1,
            ConfigStandard.ORIGIN_CENTER_ZONE_SIZE: 3,
            ConfigStandard.DEMAND_TOTAL_HOURS: 6,
        }
    )

    # -----------------------------------------------------------------#
    # Episodes #########################################################
    # -----------------------------------------------------------------#
    episodes = 2000
    episodeLog = EpisodeLog(config=config)
    amod = AmodGrid(config)

    try:
        # Load last episode
        values, counts = episodeLog.load_progress()
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
        # episodeLog.save_origins([o.id for o in origins])

    # Get demand pattern from NY city
    step_trip_count_15 = get_trip_count_step(
        NY_TRIPS_EXCERPT_DAY,
        step=config.time_increment,
        multiply_for=config.demand_resize_factor,
        earliest_step=config.demand_earliest_step_min,
        max_steps=config.demand_max_steps,
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
            classed=False,
        )

        # Start saving data of each step in the environment
        step_log = StepLog(amod)

        # Resetting environment
        amod.reset()

        # Iterate through all steps and match requests to cars
        for step, trips in enumerate(step_trip_list):

            # Compute fleet status
            step_log.compute_fleet_status()

            # Show time step statistics
            # step_log.show_info()

            # What each vehicle is doing?
            # amod.print_fleet_stats()

            revenue, serviced, rejected = adp_grid(
                amod,
                trips,
                step + 1,
                value_function_update=match.AVERAGED_UPDATE,
                myopic=True,
                # agg_level=amod.config.incumbent_aggregation_level,
            )

            # ---------------------------------------------------------#
            # Update log with iteration ################################
            # ---------------------------------------------------------#

            step_log.add_record(revenue, serviced, rejected)

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
            step_log, weights=amod.get_weights(step), progress=True
        )

    episodeLog.compute_learning()
