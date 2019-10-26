import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import random
import pandas as pd
import numpy as np
from collections import defaultdict
from datetime import timedelta, datetime
import math

# Adding project folder to import modules
root = os.getcwd().replace("\\", "/")
sys.path.append(root)

import mod.env.config as conf

conf.TRIP_FILES[0]


def get_step_trip_list(
    path, step=15, earliest_step=0, max_steps=None, resize_factor=1
):
    """Read trip csv file and return list of trips for each time step.

    Parameters
    ----------
    path : str
        Trip list file
    step : int, optional
        Time step (min) to aggregate trips, by default 15
    earliest_step : int, optional
        Trip list starts from earliest step, by default 0
    max_steps : int, optional
        Total number of steps from earliest step, by default None
    resize_factor: float
        Percentage of trips sampled in each time step. Used to create
        smaller test cases.

    Returns
    -------
    list if trip info list
        List of trip tuples (time, count, o, d) occuring in each time
        step. 
    """

    df = pd.read_csv(path, index_col="pickup_datetime", parse_dates=True)

    # List of list of trip info (time, passenger count, o_id, d_id)
    step_trip_list = []

    # Time increment
    step_timedelta = timedelta(minutes=step)

    # Earliest time window
    from_datetime = df.index[0]

    # Earliest time
    from_datetime = from_datetime + earliest_step * step_timedelta
    limit_datetime = df.index[-1]

    if max_steps:
        limit_datetime = from_datetime + max_steps * step_timedelta

    while True:
        # Right time window
        to_datetime = from_datetime + step_timedelta
        df_slice = df[from_datetime:to_datetime]

        # Trips associated to timestep
        trip_list = []

        for i in range(0, len(df_slice) - 1):
            # What time trip has arrived into the system
            placement_time = df_slice.index[i]

            # How many passengers
            passenger_count = df_slice.iloc[i]["passenger_count"]

            # Origin id
            pk_id = df_slice.iloc[i]["pk_id"]

            # Destination id
            dp_id = df_slice.iloc[i]["dp_id"]

            # Trip info tuple is added to step
            trip_list.append(
                (placement_time, int(passenger_count), int(pk_id), int(dp_id))
            )

        # Update time windows
        from_datetime = to_datetime

        # Sample trips in step
        if resize_factor < 1:
            sample_size = math.ceil(resize_factor * len(trip_list))
            trip_list = random.sample(trip_list, k=sample_size)

        step_trip_list.append(trip_list)

        # Finished processing trips
        if from_datetime >= limit_datetime:
            break

    np.save(step_trip_list, path.replace("csv", "npy"))

    return step_trip_list


print(conf.TRIP_FILES[0])
trips = get_step_trip_list(conf.TRIP_FILES[0])
print(trips)
