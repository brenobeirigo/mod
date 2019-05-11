import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
import pickle

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
from mod.env.match import fcfs, myopic
from mod.env.trip import (
    get_random_trips,
    get_trip_count_step,
    get_trips_random_ods,
)
from mod.env.ml import Adp, get_state
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
            ConfigStandard.AGGREGATION_LEVELS: 4,
        }
    )

    episodeLog = EpisodeLog()

    amod = Amod(config)
    adp = Adp(amod)

    print("## Random centers")

    n_zones = 4
    neighborhood_levels = 5
    origins = get_random_centers(amod, n_zones, neighborhood_levels)
    pprint(origins)

    print(amod.zones)

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

    adp.print_dimension(max(step_trip_count_15), config.time_steps)

    df = pd.DataFrame()

    episode_reward_list = list()
    episode_service_rate = list()

    for n in range(episodes):

        total_reward = 0

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

            # print("## BEFORE")
            # amod.print_current_stats()

            revenue, serviced, rejected = myopic(amod, trips, step)

            total_reward += revenue

            # Update log with iteration
            step_log.add_record(revenue, serviced, rejected)
            try:
                sr = len(serviced) / len(trips)
            except:
                sr = 0

            # print(
            #     f"### Time step: {step+1:>3}"
            #     f" ### Profit: {revenue:>10.2f}"
            #     f" ### Service level: {sr:>6.2%}"
            #     f" ### Trips: {len(trips):>3}"
            #     " ###"
            # )

            # amod.print_current_stats()
        #     # pprint(
        #     #     {
        #     #         state:{
        #     #             action:f'{value:>5.2f}'
        #     #             for action, value in actions.items() if value > 0
        #     #         } for state, actions in amod.Q.items()
        #     #     }
        #     # )

        # step_log.overall_log()
        episodeLog.add_record(step_log.total_reward, step_log.service_rate)

        step_log.plot_fleet_status(
            file_path=FOLDER_FLEET_PLOT + config.label + f"{n:04}",
            file_format="png",
            dpi=150,
        )

        step_log.plot_service_status(
            file_path=FOLDER_SERVICE_PLOT + config.label + f"{n:04}",
            file_format="png",
            dpi=150,
        )

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
