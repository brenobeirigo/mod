import random
import pandas as pd
import numpy as np
from collections import defaultdict

class Trip:
    trip_count = 0
    def __init__(self, o, d, time):
        self.o = o
        self.d = d
        self.time = time
        self.id = Trip.trip_count
        self.attribute = (self.o.id,
                          self.d.id)
        Trip.trip_count +=1
        self.picked_by = None
        
    def __str__(self):
        return f'T{self.id:02}({self.o},{self.d})'
    
    def __repr__(self):

        return (
            f'Trip{{'
            'id={self.id:03},'
            'o={self.o.id:03},'
            'd={self.d.id:03},'
            'time={self.time:03}}}'
        )

####################################################################
###### Trip helpers ################################################
####################################################################

def get_trip_count_step(path, step=15, multiply_for=1):
    
    df_trips = pd.read_csv(
        path,
        index_col='pickup_datetime',
        parse_dates=True
    )

    # Select first column
    df_trips = df_trips.iloc[:,0]
    df_trips = df_trips.resample(f'{step}T').count()

    trip_count_step = (np.array(df_trips)*multiply_for).astype(int)

    return trip_count_step


def get_random_trips(locations_list, time_step, min_trips, max_trips):
    """ Return a random number of trips
    """
    trips = list()

    # Choose random location
    from_locations = random.choices(
        locations_list,
        k = (
            min_trips if min_trips == max_trips else
            random.randint(min_trips, max_trips)
        )
    )

    for o in from_locations:
        # Choose random destination
        d = random.choice(locations_list)

        if o != d:
            trips.append(Trip(o, d, time_step))

    return trips
