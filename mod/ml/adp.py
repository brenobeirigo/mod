import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Adding project folder to import modules
root = os.getcwd().replace("\\", "/")
sys.path.append(root)

from mod.env.amod import Amod, ActionSpace
from mod.env.visual import StepLog
from mod.env.config import ConfigStandard, NY_TRIPS_EXCERPT_DAY
from mod.env.match import fcfs
from mod.env.trip import get_random_trips, get_trip_count_step
from mod.env.ml import Adp, get_state


if __name__ == "__main__":

    config = ConfigStandard()
    amod = Amod(config)
    adp = Adp(amod)

    # Dynamic programming algorithm
    episodes = 2000

    
    step_trip_count = get_trip_count_step(
        NY_TRIPS_EXCERPT_DAY,
        step=15,
        multiply_for=0.2
    )

    amod.print_dimension(max(step_trip_count), config.time_steps)

    for n in range(episodes):

        total_reward = 0

        # Create all episode trips
        step_trip_list = [[]]*config.offset_repositioning
        step_trip_list.extend([
        get_random_trips(amod.points, t, n_trips,  n_trips)
        for t, n_trips in enumerate(step_trip_count)])
        step_trip_list.extend([[]]*config.offset_termination)

        # Start saving data of each step in the environment
        step_log = StepLog(amod)
        
        # Resetting environment
        amod.reset()
        
        print(
            f'####### [Episode {n:>5}] ########'
            f' [Q size: {len(amod.Q.keys()):>6}] ########'
        )

        # Iterate through all steps and match requests to cars
        for step, trip_list in enumerate(step_trip_list):

            for car in amod.cars:

                # Check if vehicles finished their tasks
                car.update(step, time_increment=config.time_increment)

            reward, list_serviced, list_rejected = adp.iterate(step, trip_list)

            total_reward+=reward

            # Update log with iteration
            step_log.add_record(
                reward,
                list_serviced,
                list_rejected
            )

            #amod.print_current_stats()
    
        # pprint(
        #     {
        #         state:{
        #             action:f'{value:>5.2f}'
        #             for action, value in actions.items() if value > 0
        #         } for state, actions in amod.Q.items()
        #     }
        # )

        step_log.plot_timestep_status()
        step_log.plot_trip_coverage_battery_level()
        step_log.overall_log()

        print(n, total_reward)
