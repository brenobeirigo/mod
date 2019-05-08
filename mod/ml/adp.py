import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Adding project folder to import modules
root = os.getcwd().replace("\\", "/")
sys.path.append(root)

from mod.env.amod import Amod
from mod.env.visual import StepLog, EpisodeLog
from mod.env.config import ConfigStandard, NY_TRIPS_EXCERPT_DAY
from mod.env.match import fcfs, myopic
from mod.env.trip import (
    get_random_trips,
    get_trip_count_step,
    get_trips_random_ods,
)
from mod.env.ml import Adp, get_state
from pprint import pprint

if __name__ == "__main__":

    config = ConfigStandard()
    # config.update(
    #     {
    #         ConfigStandard.AGGREGATION_LEVELS: 4,
    #         ConfigStandard.FLEET_SIZE: 1,
    #         ConfigStandard.ROWS: 32,
    #         ConfigStandard.BATTERY_LEVELS: 20,
    #         ConfigStandard.COLS: 32,
    #         ConfigStandard.PICKUP_ZONE_RANGE: 2,
    #     }
    # )

    episodeLog = EpisodeLog()

    amod = Amod(config)
    adp = Adp(amod)

    # Dynamic programming algorithm
    episodes = 2000

    step_trip_count_15 = get_trip_count_step(
        NY_TRIPS_EXCERPT_DAY, step=15, multiply_for=0.05
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
        )

        # Start saving data of each step in the environment
        step_log = StepLog(amod)

        # Resetting environment
        amod.reset()

        # adp.print_table()
        # adp.stats()

        print(f"####### [Episode {n:>5}] ########")

        # Iterate through all steps and match requests to cars
        for step, trip_list in enumerate(step_trip_list):

            # Where are the cars, and what are
            #  they doing at the current step?
            amod.update_fleet_status(step)

            reward, list_serviced, list_rejected = myopic(
                amod, trip_list, step
            )

            total_reward += reward

            # Update log with iteration
            step_log.add_record(reward, list_serviced, list_rejected)

            # amod.print_current_stats()

        # pprint(
        #     {
        #         state:{
        #             action:f'{value:>5.2f}'
        #             for action, value in actions.items() if value > 0
        #         } for state, actions in amod.Q.items()
        #     }
        # )

        # step_log.plot_timestep_status()
        # step_log.plot_trip_coverage_battery_level()
        step_log.overall_log()

        episodeLog.add_record(step_log.total_reward, step_log.service_rate)

    episodeLog.plot_reward(scale="linear")
    episodeLog.plot_service_rate()
