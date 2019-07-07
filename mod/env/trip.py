import random
import pandas as pd
import numpy as np
from collections import defaultdict
import mod.env.network as nw
from datetime import timedelta, datetime
import math


class Trip:
    trip_count = 0

    def __init__(self, o, d, time):
        self.o = o
        self.d = d
        self.time = time
        self.id = Trip.trip_count
        Trip.trip_count += 1
        self.picked_by = None
        self.dropoff_time = None

    def attribute(self, level):
        return (self.o.id_level(level), self.d.id_level(level))

    def __str__(self):
        return f"T{self.id:02}({self.o},{self.d})"

    def __repr__(self):

        return (
            f"Trip{{"
            f"id={self.id:03},"
            f"o={self.o.level_ids},"
            f"d={self.d.level_ids},"
            f"time={self.time:03}}}"
        )

    def can_be_picked_by(self, car, level=0):
        return self.o.id_level(level) == car.point.id_level(level)


class ClassedTrip(Trip):
    SQ_CLASS_1 = "A"
    SQ_CLASS_2 = "B"

    sq_classes = dict(A=1.0, B=0.9)
    sq_level_class = dict(A=[3, 3], B=[3, 4])
    class_proportion = dict(A=0.1, B=0.9)

    @classmethod
    def get_levels(cls):
        class_levels = set()
        for levels in cls.sq_level_class.values():
            class_levels.update(levels)
        return class_levels

    def __init__(self, o, d, time, sq_class):
        super().__init__(o, d, time)
        self.sq_class = sq_class

        # Level demanded in best case scenario
        self.sq1_level = ClassedTrip.sq_level_class[sq_class][0]

        # Level demanded in worst case scenario
        self.sq2_level = ClassedTrip.sq_level_class[sq_class][1]

        # Region center id of best case pickup scenario
        self.id_sq1_level = self.o.id_level(self.sq1_level)

        # Region center id of worst case pickup scenario
        self.id_sq2_level = self.o.id_level(self.sq2_level)

    @property
    def attribute(self, level=0):
        return (self.o.id_level(level), self.d.id_level(level), self.sq_class)

    def __str__(self):
        return f"{self.sq_class}{self.id:02}({self.o},{self.d})"

    def __repr__(self):

        return (
            f"Trip{{"
            f"id={self.id:03},"
            f"o={self.o.level_ids},"
            f"d={self.d.level_ids},"
            f"sq={self.sq_class},"
            f"time={self.time:03}}}"
        )


# #################################################################### #
# Trip helpers ####################################################### #
# #################################################################### #


def get_trip_count_step(
    path, step=15, multiply_for=1, earliest_step=0, max_steps=None
):

    df_trips = pd.read_csv(path, index_col="pickup_datetime", parse_dates=True)

    # Select first column
    df_trips = df_trips.iloc[:, 0]
    df_trips = df_trips.resample(f"{step}T").count()

    trip_count_step = (np.array(df_trips) * multiply_for).astype(int)

    if max_steps:
        trip_count_step = trip_count_step[
            earliest_step : earliest_step + max_steps
        ]

    return trip_count_step


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

    return step_trip_list


def get_random_trips(
    locations_list,
    time_step,
    min_trips,
    max_trips,
    origins=None,
    destinations=None,
    classed=False,
):
    """ Return a random number of trips
    """
    trips = list()
    # weights = np.zeros(len(locations_list))
    # weights[0] = 1

    # Choose random location
    from_locations = random.choices(
        (origins if origins else locations_list),
        # weights=weights,
        k=(
            min_trips
            if min_trips == max_trips
            else random.randint(min_trips, max_trips)
        ),
    )

    # Destination set
    to_locations = destinations if destinations else locations_list

    for o in from_locations:

        # Choose random destination
        d = random.choice(to_locations)

        if o != d:

            if classed:
                trips.append(
                    ClassedTrip(
                        o,
                        d,
                        time_step,
                        (
                            ClassedTrip.SQ_CLASS_1
                            if random.random() < 0.3
                            else ClassedTrip.SQ_CLASS_2
                        ),
                    )
                )

            else:
                trips.append(Trip(o, d, time_step))

    return trips


def get_trip_list_step(
    points, n_steps, min_trips, max_trips, offset_start=0, offset_end=0
):

    # Populate first steps with empty lists
    step_trip_list = [[]] * offset_start

    if min_trips and max_trips:
        step_trip_list.extend(
            [
                get_random_trips(points, t, min_trips, max_trips)
                for t in n_steps
            ]
        )

    # Populate last steps with empty lists
    step_trip_list.extend([[]] * offset_end)

    return step_trip_list


def get_trips_random_ods(
    points,
    step_trip_count,
    offset_start=0,
    offset_end=0,
    origins=None,
    destinations=None,
    classed=False,
):

    # Populate first steps with empty lists
    step_trip_list = [[]] * offset_start

    step_trip_list.extend(
        [
            get_random_trips(
                points,
                t,
                n_trips,
                n_trips,
                origins=origins,
                destinations=destinations,
                classed=classed,
            )
            for t, n_trips in enumerate(step_trip_count)
        ]
    )

    # Populate last steps with empty lists
    step_trip_list.extend([[]] * offset_end)

    return step_trip_list


def get_trips(
    points,
    step_trips,
    offset_start=0,
    offset_end=0,
    classed=False,
    resize_factor=1,
):

    # Populate first steps with empty lists
    step_trip_list = [[]] * offset_start

    for step, trips in enumerate(step_trips):
        step_trip_list.append(
            [
                ClassedTrip(
                    points[o],
                    points[d],
                    step,
                    (
                        ClassedTrip.SQ_CLASS_1
                        if random.random() <= ClassedTrip.class_proportion["A"]
                        else ClassedTrip.SQ_CLASS_2
                    ),
                )
                for time, count, o, d in trips
                if random.random() < resize_factor
            ]
        )

    # Populate last steps with empty lists
    step_trip_list.extend([[]] * offset_end)

    return step_trip_list
