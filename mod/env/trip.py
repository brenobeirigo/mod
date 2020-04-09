import random
import pandas as pd
import numpy as np
from collections import defaultdict
from datetime import timedelta, datetime
import math
import time
import mod.env.network as nw

# Reproducibility of the experiments
random.seed(1)


class Trip:
    trip_count = 0

    def __init__(self, o, d, time):
        self.o = o
        self.d = d
        self.time = time  # step
        self.pk_step = None  # step
        self.id = Trip.trip_count
        Trip.trip_count += 1
        self.picked_by = None
        self.dropoff_time = None
        self.pk_delay = None
        # Accrue backlogging delay
        self.backlog_delay = 0

    def attribute(self, level):
        return (self.o.id_level(level), self.d.id_level(level))

    def __str__(self):
        return f"T{self.id:03}[{self.o:>4},{self.d:>4}]"

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
    def __str__(self):
        return (
            f"[{self.time:04}({self.placement})]"
            f"{self.sq_class}{self.id:03}[{self.o:>4},{self.d:>4}]"
            f" - remaining: {self.max_delay_from_placement:>6.2f} min"
        )

    def __init__(
        self,
        o,
        d,
        time,
        sq_class,
        elapsed_sec=0,
        placement=None,
        max_delay=10,
        tolerance=5,
        distance_km=None,
    ):
        super().__init__(o, d, time)
        self.sq_class = sq_class

        # How much time has passed from the beginning of the step
        # to the announcement time
        self.elapsed_sec = elapsed_sec

        # Datetime trip was placed in the system
        self.placement = placement

        # Min/Max class delays
        self.max_delay = max_delay
        self.tolerance = tolerance
        self.distance_km = distance_km

        # Min/Max delays discounting announcement
        self.max_delay_from_placement = (
            self.max_delay - (60 - self.elapsed_sec) / 60
        )

    def update_delay(self, period_min):
        self.max_delay_from_placement -= period_min
        self.backlog_delay += period_min
        return self.max_delay_from_placement

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
            f"bklog={self.backlog_delay},"
            f"time={self.time:03}}}"
        )

    def info(self):

        return (
            f"Trip{{"
            f"id={self.id:03},"
            f"o={self.o.level_ids},"
            f"d={self.d.level_ids},"
            f"sq={self.sq_class},"
            f"time={self.time:03},"
            f"pk_delay={self.pk_delay},"
            f"max_delay={self.max_delay:6.2f},"
            f"from_placement={self.max_delay_from_placement:6.2f},"
            f"tolerance={self.tolerance:6.2f},"
            f"elapsed={self.elapsed_sec:6.2f}}}"
        )


# #################################################################### #
# Trip helpers ####################################################### #
# #################################################################### #

# Trip tuples per step (key is file path)
trip_od_list = dict()

# List of trips and trip counts per step (key is file path)
trip_demand_dict = dict()


def load_trips_ods_from(tripdata_path, config):

    global trip_od_list

    # Load previously read ods
    if tripdata_path in trip_od_list:
        step_trip_od_list = trip_od_list[tripdata_path]

    # Read trip ods for the first time
    else:
        print("Creating trip tuple list per step...")

        # Sample trip data if resize factor < 1
        step_trip_od_list = get_step_trip_list(
            tripdata_path,
            step=config.time_increment,
            earliest_step=config.demand_earliest_step_min,
            max_steps=config.demand_max_steps,
        )

        # Store created list
        trip_od_list[tripdata_path] = step_trip_od_list

    return step_trip_od_list


def get_df(step_trip_list, show_service_data=False, earliest_datetime=None):
    """Get dataframe from sampled trip list.

    Parameters
    ----------
    step_trip_list : list of lists
        List of trip lists occuring in the same step.
    show_service_data: bool
        Show trip pickup and dropoff results.
    earliest_datetime: datetime
        Trip start time - rebalance offset

    Returns
    -------
    DataFrame
        Dataframe with trip data info.
    """
    d = defaultdict(list)
    for trips in step_trip_list:
        for t in trips:
            d["placement_datetime"].append(t.placement)
            d["pk_id"].append(t.o.id)
            d["dp_id"].append(t.d.id)
            d["sq_class"].append(t.sq_class)
            d["max_delay"].append(t.max_delay)
            d["tolerance"].append(t.tolerance)
            lon_o, lat_o = nw.tenv.lonlat(t.o.id)
            lon_d, lat_d = nw.tenv.lonlat(t.d.id)
            d["passenger_count"].append(1)
            d["pickup_latitude"].append(lat_o)
            d["pickup_longitude"].append(lon_o)
            d["dropoff_latitude"].append(lat_d)
            d["dropoff_longitude"].append(lon_d)

            if show_service_data:
                if t.pk_delay is not None:
                    pickup_datetime = t.placement + timedelta(
                        minutes=t.pk_delay
                    )
                    pickup_datetime_str = datetime.strftime(
                        pickup_datetime, "%Y-%m-%d %H:%M:%S"
                    )

                if t.dropoff_time is not None:
                    dropoff_datetime = earliest_datetime + timedelta(
                        minutes=t.dropoff_time
                    )

                    dropoff_datetime_str = datetime.strftime(
                        dropoff_datetime, "%Y-%m-%d %H:%M:%S"
                    )

                d["pickup_step"].append(
                    t.pk_step if t.pk_step is not None else "-"
                )
                d["pickup_delay"].append(
                    t.pk_delay if t.pk_delay is not None else "-"
                )
                d["pickup_datetime"].append(
                    pickup_datetime_str if t.pk_delay is not None else "-"
                )
                d["dropoff_time"].append(
                    t.dropoff_time if t.dropoff_time is not None else "-"
                )
                d["dropoff_datetime"].append(
                    dropoff_datetime_str if t.dropoff_time is not None else "-"
                )
                d["picked_by"].append(t.picked_by)

    df = pd.DataFrame.from_dict(dict(d))
    return df


def get_ny_demand(
    config, tripdata_path, points, seed=None, prob_dict=None, centroid_level=0
):

    step_trip_od_list = load_trips_ods_from(tripdata_path, config)

    if tripdata_path not in trip_demand_dict or seed is not None:

        # Use loaded trip od list to create RESIZED trip list per step
        step_trip_list = get_trips(
            points,
            step_trip_od_list,
            config.trip_class_proportion,
            config.trip_max_pickup_delay,
            config.trip_tolerance_delay,
            offset_start=config.offset_repositioning_steps,
            offset_end=config.offset_termination_steps,
            classed=config.demand_is_classed,
            resize_factor=config.demand_resize_factor,
            seed=seed,
            prob_dict=prob_dict,
            centroid_level=centroid_level,
        )

        # Count number of trips per step
        step_trip_count = [len(trips) for trips in step_trip_list]
        list_count_trips = (step_trip_list, step_trip_count)

        if not config.demand_sampling:
            # Save trips to be used on next call
            trip_demand_dict[tripdata_path] = list_count_trips

    else:
        list_count_trips = trip_demand_dict[tripdata_path]

    return list_count_trips


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
    The list of trips is saved in a .npy file in the same directory
    of the .csv for fast processing.

    Parameters
    ----------
    path : str
        Trip list .csv file
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

    # Processed trip data (list of trips) is saved in a .npy file
    # for faster reading
    path_npy = (
        f"{path.split('.')[0]}_"
        f"increment={step:03}min_"
        f"earlieststep={earliest_step:04}_"
        f"maxsteps={(f'{max_steps:04}' if max_steps else '--')}_"
        f"resize={resize_factor:.2f}.npy"
    )

    try:
        print(f"Trying to load processed trip data from '{path_npy}'")
        t1 = time.time()
        step_trip_list = np.load(path_npy, allow_pickle=True)
        print(f"Trip list loaded (took {time.time() - t1:10.6f} seconds)")

    except Exception as e:
        print(f"Loading .npy failed. Exception:'{e}'. Processing trip data...")
        t1 = time.time()
        df = pd.read_csv(path, index_col="pickup_datetime", parse_dates=True)

        # List of list of trip info (time, passenger count, o_id, d_id)
        step_trip_list = []

        # Time increment
        step_timedelta = timedelta(minutes=step)

        # Earliest time (first date)
        from_datetime = datetime(
            year=df.index[0].year, month=df.index[0].month, day=df.index[0].day
        )

        # Earliest time
        from_datetime = from_datetime + earliest_step * step_timedelta
        limit_datetime = df.index[-1]

        if max_steps:
            limit_datetime = from_datetime + max_steps * step_timedelta

        while True:
            # Right time window
            to_datetime = from_datetime + step_timedelta
            mask = (df.index >= from_datetime) & (df.index < to_datetime)
            df_slice = df[mask]

            # Trips associated to timestep
            trip_list = []

            # placement_first = df_slice.index[0]
            for i in range(0, len(df_slice)):
                # What time trip has arrived into the system
                placement_time = df_slice.index[i]

                # Time delta
                elapsed_sec = (placement_time - from_datetime).seconds

                # How many passengers
                passenger_count = df_slice.iloc[i]["passenger_count"]

                # Origin id
                pk_id = df_slice.iloc[i]["pk_id"]

                # Destination id
                dp_id = df_slice.iloc[i]["dp_id"]

                # Distance in kilometers
                trip_dist_km = df_slice.iloc[i]["trip_distance"]

                # Trip info tuple is added to step
                trip_list.append(
                    (
                        placement_time,
                        int(elapsed_sec),
                        int(passenger_count),
                        int(pk_id),
                        int(dp_id),
                        float(trip_dist_km),
                    )
                )

            # Update time windows
            from_datetime = to_datetime

            # Sample trips in step
            if resize_factor < 1:
                sample_size = math.ceil(resize_factor * len(trip_list))
                trip_list = random.sample(trip_list, k=sample_size)
                trip_list.sort(key=lambda t: t[0])

            step_trip_list.append(trip_list)

            # Finished processing trips
            if from_datetime >= limit_datetime:
                break

        print(f"Processed finished {time.time()-t1:10.6f} seconds. Saving...")
        t2 = time.time()
        np.save(path_npy, step_trip_list)
        print(f"Saved in {time.time()-t2:10.6f} seconds.")

    return step_trip_list


def get_random_trips(
    locations_list,
    time_step,
    min_trips,
    max_trips,
    class_proportion,
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
                        random.choices(
                            population=list(class_proportion.keys()),
                            weights=list(class_proportion.values()),
                            k=1,
                        ),
                    )
                )

            else:
                trips.append(Trip(o, d, time_step))

    return trips


def get_trip_list_step(
    points,
    n_steps,
    min_trips,
    max_trips,
    class_proportion,
    offset_start=0,
    offset_end=0,
):

    # Populate first steps with empty lists
    step_trip_list = [[]] * offset_start

    if min_trips and max_trips:
        step_trip_list.extend(
            [
                get_random_trips(
                    points, t, min_trips, max_trips, class_proportion
                )
                for t in n_steps
            ]
        )

    # Populate last steps with empty lists
    step_trip_list.extend([[]] * offset_end)

    return step_trip_list


def get_trips_random_ods(
    points,
    step_trip_count,
    class_proportion,
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
                class_proportion,
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


def get_min(h, m, s, min_bin_size=30):
    s = (h * 3600 + m * 60 + s) // (min_bin_size * 60)
    return s


def get_class(pk_id, pickup_datetime, prob_info, min_bin_size=30):
    """Return 1 if request belongs to first class """

    if "time_bin" in prob_info:
        min_bin_size = prob_info["time_bin"]

    prob_dict = prob_info["data"]
    min_bin = get_min(
        pickup_datetime.hour,
        pickup_datetime.minute,
        pickup_datetime.second,
        min_bin_size=min_bin_size,
    )
    # print("####", pickup_datetime)
    # print(
    #     "time={}:{}:{}".format(
    #         pickup_datetime.hour,
    #         pickup_datetime.minute,
    #         pickup_datetime.second,
    #     )
    # )
    # print("MIN", min_bin_size, min_bin)
    try:
        prob = prob_dict[pk_id][min_bin]
        if random.random() <= prob:
            return 1
        else:
            return 0
    except:
        return 0


def get_trips(
    points,
    step_trips,
    class_proportion,
    max_delay,
    tolerance,
    offset_start=0,
    offset_end=0,
    classed=False,
    resize_factor=1,
    seed=None,
    prob_dict=None,
    centroid_level=0,
):

    # Populate first steps with empty lists
    step_trip_list = [[]] * offset_start

    # Guarantee everytime the same trips are sampled.
    # Used in conjunction with the testing data.
    if seed is not None:
        random.seed(seed)
    step_trips_resized = list()

    # Sampling trips
    for step, trips in enumerate(step_trips):
        resized = list()
        for t in trips:
            # Only add a new trip "resize_factor" percent of the time
            if random.random() < resize_factor:
                resized.append(t)
        step_trips_resized.append(resized)

    for step, trips in enumerate(step_trips_resized):
        trip_list = list()
        for time, elapsed_sec, count, o, d, distance_km in trips:
            # Use probability distribution from file
            if prob_dict is not None:
                random_class = get_class(o, time, prob_dict)
                random_class = "A" if random_class else "B"
            else:
                # Choose a class according to a probability
                random_class = random.choices(
                    population=list(class_proportion.keys()),
                    weights=list(class_proportion.values()),
                    k=1,
                )[0]

            # Points become region centroids (point id only if
            # centroid_level=0)
            trip_list.append(
                ClassedTrip(
                    points[points[o].id_level(centroid_level)],
                    points[points[d].id_level(centroid_level)],
                    offset_start + step,
                    random_class,
                    elapsed_sec=elapsed_sec,
                    placement=time,
                    max_delay=max_delay[random_class],
                    tolerance=tolerance[random_class],
                    distance_km=distance_km,
                )
            )
        step_trip_list.append(trip_list)

    # Populate last steps with empty lists
    step_trip_list.extend([[]] * offset_end)

    return step_trip_list
