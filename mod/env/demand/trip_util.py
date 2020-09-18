import math
import random
import time
from collections import defaultdict
from datetime import timedelta, datetime

import numpy as np
import pandas as pd

import mod.env.network as nw
from mod.env.demand.ClassedTrip import ClassedTrip
from mod.env.demand.Trip import Trip

# Reproducibility of the experiments
random.seed(1)

# Trip tuples
TIME = 0
ELAPSED_SEC = 1
COUNT = 2
ORIGIN = 3
DESTINATION = 4
DISTANCE_KM = 5

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


def get_df_from_sampled_trips(step_trip_list, show_service_data=False, earliest_datetime=None):
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
    for step, trips in enumerate(step_trip_list):
        for t in trips:
            d["placement_datetime"].append(t.placement)
            d["step"].append(step + 1)
            d["pk_id"].append(t.o.id)
            d["dp_id"].append(t.d.id)
            d["sq_class"].append(t.sq_class)
            d["max_delay"].append(t.max_delay)
            d["elapsed_sec"].append(t.elapsed_sec)
            d["max_delay_from_placement"].append(t.max_delay_from_placement)
            d["delay_close_step"].append(t.delay_close_step)
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
                d["times_backlogged"].append(t.times_backlogged)
                d["pickup_step"].append(
                    t.pk_step if t.pk_step is not None else "-"
                )
                d["dropoff_step"].append(
                    t.dp_step if t.dp_step is not None else "-"
                )
                d["pickup_delay"].append(
                    t.pk_delay if t.pk_delay is not None else "-"
                )

                d["pickup_duration"].append(
                    t.pk_duration if t.pk_duration is not None else "-"
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
    df.sort_values(by=["placement_datetime", "sq_class"], inplace=True)
    return df


def get_ny_demand(
        config,
        tripdata_path,
        points,
        seed=None,
        prob_dict=None,
        centroid_level=0,
        unreachable_ods=None,
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
            time_increment=config.time_increment,
            unreachable_ods=unreachable_ods,
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
                          earliest_step: earliest_step + max_steps
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

                try:
                    # Distance in kilometers (Rotterdam)
                    trip_dist_km = df_slice.iloc[i]["trip_dist_km"]
                except:
                    # Distance in kilometers (TS)
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

        print(f"Processed finished {time.time() - t1:10.6f} seconds. Saving...")
        t2 = time.time()
        np.save(path_npy, step_trip_list)
        print(f"Saved in {time.time() - t2:10.6f} seconds.")

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
        time_increment=1,
        unreachable_ods=None,
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

        filtered = list()
        for t in trips:
            o = points[t[ORIGIN]].id_level(centroid_level)
            d = points[t[DESTINATION]].id_level(centroid_level)

            # Unreachable ods are excluded
            if unreachable_ods and (
                    o in unreachable_ods or d in unreachable_ods
            ):
                continue

            # Trips with centroid(o) == centroid(d) are excluded
            if o == d:
                continue

            filtered.append(t)

        # Only add a new trip "resize_factor" percent of the time
        resized = random.choices(filtered, k=int(resize_factor * len(trips)))
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

            o_centroid = points[points[o].id_level(centroid_level)]
            d_centroid = points[points[d].id_level(centroid_level)]
            # Points become region centroids (point id only if
            # centroid_level=0)
            trip_list.append(
                ClassedTrip(
                    o_centroid,
                    d_centroid,
                    offset_start + step,
                    random_class,
                    elapsed_sec=elapsed_sec,
                    placement=time,
                    max_delay=max_delay[random_class],
                    tolerance=tolerance[random_class],
                    distance_km=distance_km,
                    time_increment=time_increment,
                )
            )
        step_trip_list.append(trip_list)

    # Populate last steps with empty lists
    step_trip_list.extend([[]] * offset_end)

    return step_trip_list
