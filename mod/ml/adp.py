import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Adding project folder to import modules
root = os.getcwd().replace("\\", "/")
sys.path.append(root)

from mod.env.amod import Amod, StepLog, ActionSpace
from mod.env.config import ConfigStandard
from mod.env.match import fcfs
from mod.env.trip import get_random_trips, get_trip_count_step

def iterate(amod, trips, step, epsilon = 0.1, gamma = 0.8, alpha = 0.1):
    """Service all trips in a given step, update the environment
    and store the iteration result in the 'step_log' object.
    
    Arguments:
        amod {Amod} -- Environment
        trips {list} -- List of trips
        step {int} -- Current step
    
    Returns:
        [type] -- [description]
    """

    # print(
    #     f'---- Time step {step:>4} '
    #     f'---- # Trips = {len(trips):>4} '
    #     f'---- # States visited:{len(amod.Q.keys())}'
    # )
    
    # Current state, return None if all cars are busy or there are no trips
    state, dict_car_attribute, dict_trip_attribute = amod.get_state_sparse2(
        step,
        amod.cars,
        trips
    )
    
    action = amod.action_space.get_action_tuple(amod.cars)

    reward, list_serviced, list_rejected = fcfs(
        amod,
        trips,
        step
    )
    
    # Get the best action so far
    # Get actions taken from current state
    action_dict = amod.Q.get(state, None)
    
    # Update the new value
    amod.Q[state][action] = (
        (1 - alpha) * reward
        + alpha * (
            amod.Q[state].get(action, 0)
            if state in amod.Q else 0
        )
    )

    return reward, list_serviced, list_rejected


if __name__ == "__main__":

    config = ConfigStandard()
    amod = Amod(config)
    amod.start_action_space_sparse()

    # Dynamic programming algorithm
    episodes = 2000

    # Trip data to group in steps
    ny_trips = root + '/data/input/32874_samples_01_feb_2011_NY.csv'
    step_trip_count = get_trip_count_step(ny_trips, step=15, multiply_for=0.2)

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
        
        # print(
        #     f'####### [Episode {n:>5}] ########'
        #     f' [Q size: {len(amod.Q.keys()):>6}] ########'
        # )

        # Iterate through all steps and match requests to cars
        for step, trip_list in enumerate(step_trip_list):

            for car in amod.cars:

                # Check if vehicles finished their tasks
                car.update(step, time_increment=config.time_increment)


            reward, list_serviced, list_rejected = iterate(amod,trip_list, step)

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

        # step_log.plot_timestep_status()
        # step_log.plot_trip_coverage_battery_level()
        # step_log.overall_log()

        print(n, total_reward)
