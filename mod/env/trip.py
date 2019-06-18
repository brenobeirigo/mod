import random
import pandas as pd
import numpy as np
from collections import defaultdict
import mod.env.network as nw


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

    sq_classes = dict(A=0.8, B=0.6)
    sq_level_class = dict(A=[1, 2], B=[2, 3])

    def __init__(self, o, d, time, sq_class):
        super().__init__(o, d, time)
        self.sq_class = sq_class
        self.sq1_level = ClassedTrip.sq_level_class[sq_class][0]
        self.sq2_level = ClassedTrip.sq_level_class[sq_class][1]
        self.id_sq1_level = self.o.id_level(self.sq1_level)
        self.id_sq2_level = self.o.id_level(self.sq2_level)

    def attribute(self, level):
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


def get_random_trips(
    locations_list,
    time_step,
    min_trips,
    max_trips,
    origins=None,
    destinations=None,
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
            # trips.append(Trip(o, d, time_step))
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
            )
            for t, n_trips in enumerate(step_trip_count)
        ]
    )

    # Populate last steps with empty lists
    step_trip_list.extend([[]] * offset_end)

    return step_trip_list
