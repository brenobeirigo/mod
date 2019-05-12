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
)
from mod.env.match import myopic
from mod.env.trip import (
    get_random_trips,
    get_trip_count_step,
    get_trips_random_ods,
)
import mod.env.network as nw
from pprint import pprint


def get_random_centers(env, n_zones, neighbors):
    # Choose random location
    random_centers = random.choices(
        env.points,
        # weights=weights,
        k=n_zones,
    )

    print("Random centers:", random_centers)

    nodes = []
    for c in random_centers:
        nodes.extend(nw.get_neighbor_zones(c, neighbors, env.zones))

    return [env.points[n] for n in nodes]


if __name__ == "__main__":

    config = ConfigStandard()
    config.update(
        {
            ConfigStandard.FLEET_SIZE: 250,
            ConfigStandard.ROWS: 32,
            ConfigStandard.COLS: 32,
            ConfigStandard.BATTERY_LEVELS: 20,
            ConfigStandard.PICKUP_ZONE_RANGE: 2,
            ConfigStandard.AGGREGATION_LEVELS: 3,
        }
    )

    episodeLog = EpisodeLog()

    amod = Amod(config)

    print("## Random centers")

    # Random centers in map
    n_zones = 4
    neighborhood_levels = 5
    origins = get_random_centers(amod, n_zones, neighborhood_levels)

    # Dynamic programming algorithm
    episodes = 4000

    step_trip_count_15 = get_trip_count_step(
        NY_TRIPS_EXCERPT_DAY, step=15, multiply_for=0.167
    )

    print(
        f"### DEMAND ###"
        f" - min: {min(step_trip_count_15)}"
        f" - max: {max(step_trip_count_15)}"
    )

    for n in range(episodes):

        total_reward = 0
        serviced_count = 0
        total_count = 0

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

        # adp.print_table()
        # adp.stats()

        print(
            f"####### [Episode {n:>5}]"
            f" - {episodeLog.last_episode_stats()} ########"
        )

        # Iterate through all steps and match requests to cars
        for step, trips in enumerate(step_trip_list):

            revenue, serviced, rejected = myopic(amod, trips, step)

            total_reward += revenue
            serviced_count += len(serviced)
            total_count += len(trips)

            # Update log with iteration
            # step_log.add_record(revenue, serviced, rejected)

            # step_log.show_info()

            # step_log.plot_fleet_status(
            #     file_path=FOLDER_FLEET_PLOT + config.label + f"{n:04}",
            #     file_format="png",
            #     dpi=150,
            # )

            # step_log.plot_service_status(
            #     file_path=FOLDER_SERVICE_PLOT + config.label + f"{n:04}",
            #     file_format="png",
            #     dpi=150,
            # )

            # amod.print_fleet_stats()

        # step_log.overall_log()
        episodeLog.add_record(total_reward, serviced_count / total_count)

        # # Saving last episode
        # path = FOLDER_EPISODE_TRACK + config.label + ".npy"
        # np.save(
        #     path,
        #     {
        #         t: {
        #             g: {a: value for a, value in a_value.items()}
        #             for g, a_value in g_a.items()
        #         }
        #         for t, g_a in amod.values.items()
        #     },
        # )

    episodeLog.plot_reward(scale="linear")
    episodeLog.plot_service_rate()
