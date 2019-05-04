import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Adding project folder to import modules
root = os.getcwd().replace("\\", "/")
sys.path.append(root)

from mod.env.amod import Amod, StepLog
from mod.env.config import Config, ConfigStandard
from mod.env.match import fcfs
from mod.env.trip import get_random_trips


def test_fcfs_random_trips(amod, show_stats=False):
    
    step_log = StepLog(amod)

    for time_step in range(amod.config.time_steps):
        
        # Get random set of trips
        trips = get_random_trips(
            amod.points,
            time_step,
            amod.config.min_trips,
            amod.config.max_trips
        )

        print(f'### Time step {time_step+1:>3} ######################################')

        # Match cars and trips and get:
        # - final contribution
        # - list serviced requests
        # - list rejected requests
        contribution, list_serviced, list_rejected = fcfs(amod, trips, time_step)
        step_log.add_record(contribution,list_serviced,list_rejected)
        
        # Show car stats
        if show_stats:
            amod.print_current_stats()

    step_log.plot_timestep_status()
    step_log.plot_trip_coverage_battery_level()
    step_log.overall_log()

if __name__ == "__main__":

    config = ConfigStandard()
    amod = Amod(config)
    #amod.print_environment()

    test_fcfs_random_trips(amod)
    