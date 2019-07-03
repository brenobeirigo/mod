import os
import sys
import numpy as np
from pprint import pprint
from threading import Thread
from copy import deepcopy
import time
from random import random, choice

# Adding project folder to import modules
root = os.getcwd().replace("\\", "/")
sys.path.append(root)

from mod.env.amod.AmodNetworkHired import AmodNetworkHired
from mod.env.amod.AmodNetwork import AmodNetwork
from mod.env.visual import StepLog, EpisodeLog
import mod.env.visual as vi
from mod.env.config import (
    ConfigNetwork,
    FOLDER_OUTPUT,
    NY_TRIPS_EXCERPT_DAY,
    SCENARIO_UNBALANCED,
    SCENARIO_NYC,
    TRIP_FILES,
)
from mod.env.match import adp_network, adp_network_hired2
from mod.env.car import Car, HiredCar
from mod.env.trip import (
    get_trip_count_step,
    get_trips_random_ods,
    get_step_trip_list,
    get_trips,
)
import mod.env.trip as tp
import mod.env.network as nw

from mod.env.simulator import PlotTrack
from mod.env import match


def get_sim_config():

    config = ConfigNetwork()
    # Pull graph info
    region, label, node_count, center_count, edge_count = nw.query_info()

    info = (
        "##############################################################"
        f"\n### Region: {region} G(V={node_count}, E={edge_count})"
        f"\n### Center count: {center_count}"
    )

    print(info)
    # ---------------------------------------------------------------- #
    # Amod environment ############################################### #
    # ---------------------------------------------------------------- #

    config.update(
        {
            ConfigNetwork.TEST_LABEL: "UNIVERSAL",
            # Fleet
            ConfigNetwork.FLEET_SIZE: 1500,
            ConfigNetwork.BATTERY_LEVELS: 1,
            # Time - Increment (min)
            ConfigNetwork.TIME_INCREMENT: 1,
            ConfigNetwork.OFFSET_REPOSIONING: 30,
            ConfigNetwork.OFFSET_TERMINATION: 30,
            # -------------------------------------------------------- #
            # NETWORK ################################################ #
            # -------------------------------------------------------- #
            ConfigNetwork.NAME: label,
            ConfigNetwork.REGION: region,
            ConfigNetwork.NODE_COUNT: node_count,
            ConfigNetwork.EDGE_COUNT: edge_count,
            ConfigNetwork.CENTER_COUNT: center_count,
            # Region centers are created in steps of how much time?
            ConfigNetwork.STEP_SECONDS: 30,
            # Cars rebalance to up to #region centers
            ConfigNetwork.N_CLOSEST_NEIGHBORS: (8,),
            # Cars can access locations within region centers
            # established in which neighborhood level?
            ConfigNetwork.NEIGHBORHOOD_LEVEL: 4,
            # Cars can rebalance to neighbor centers of level:
            # Why not max rebalance level?
            ConfigNetwork.REBALANCE_LEVEL: (1,),
            # ConfigNetwork.REBALANCE_REACH: 2,
            ConfigNetwork.REBALANCE_MULTILEVEL: False,
            # ConfigNetwork.LEVEL_DIST_LIST: [0, 30, 60, 90, 120, 180, 270],
            ConfigNetwork.LEVEL_DIST_LIST: [0, 60, 90, 180, 300],
            # How many levels separated by step secresize_factorc
            # LEVEL_DIST_LIST must be filled
            ConfigNetwork.AGGREGATION_LEVELS: 5,
            ConfigNetwork.SPEED: 30,
            # -------------------------------------------------------- #
            # DEMAND ################################################# #
            # -------------------------------------------------------- #
            ConfigNetwork.DEMAND_TOTAL_HOURS: 24,
            ConfigNetwork.DEMAND_EARLIEST_HOUR: 0,
            ConfigNetwork.DEMAND_RESIZE_FACTOR: 1,
            # Demand spawn from how many centers?
            ConfigNetwork.ORIGIN_CENTERS: 3,
            # Demand arrives in how many centers?
            ConfigNetwork.DESTINATION_CENTERS: 3,
            # OD level extension
            ConfigNetwork.DEMAND_CENTER_LEVEL: 4,
            # Demand scenario
            ConfigNetwork.DEMAND_SCENARIO: SCENARIO_UNBALANCED,
            ConfigNetwork.TRIP_BASE_FARE: {
                tp.ClassedTrip.SQ_CLASS_1: 4.8,
                tp.ClassedTrip.SQ_CLASS_2: 2.4,
            },
            # -------------------------------------------------------- #
            # LEARNING ############################################### #
            # -------------------------------------------------------- #
            ConfigNetwork.DISCOUNT_FACTOR: 1,
            ConfigNetwork.HARMONIC_STEPSIZE: 1,
            # -------------------------------------------------------- #
            # HIRING ################################################# #
            # -------------------------------------------------------- #
            ConfigNetwork.CONTRACT_DURATION_LEVEL: 15,
            ConfigNetwork.CONGESTION_PRICE: 10,
        }
    )
    return config


run_plot = PlotTrack(get_sim_config())

trip_demand_dict = dict()


def hire_cars_trip_regions(amod, trips, contract_duration_h, step):
    # Hired fleet is appearing in trip origins

    hired_cars = [
        HiredCar(
            amod.points[t.id_sq1_level],
            contract_duration_h,
            current_step=step,
            current_arrival=step * amod.config.time_increment,
            duration_level=amod.config.contract_duration_level,
        )
        for t in trips
    ]

    return hired_cars


def hire_cars_centers(amod, contract_duration_h, step):
    # Hired fleet is appearing in trip origins

    hired_cars = [
        HiredCar(
            amod.points[c],
            contract_duration_h,
            current_step=step,
            current_arrival=step * amod.config.time_increment,
            duration_level=amod.config.contract_duration_level,
            # battery_level_miles_max=amod.battery_size_distances,
        )
        for c in amod.points_level[2]
        if random() < 1
    ]

    return hired_cars


def get_ny_demand(amod, tripdata_path):

    if tripdata_path in trip_demand_dict:
        return trip_demand_dict[tripdata_path]

    print("Creating trip tuple list per step...")
    # Create list of trips with real world data
    step_trip_od_list = get_step_trip_list(
        tripdata_path,
        step=amod.config.time_increment,
        earliest_step=amod.config.demand_earliest_step_min,
        max_steps=amod.config.demand_max_steps,
        # resize_factor=amod.config.demand_resize_factor,
    )

    print("Creating trip list per step...")
    step_trip_list = get_trips(
        amod.points,
        step_trip_od_list,
        offset_start=amod.config.offset_repositioning,
        offset_end=amod.config.offset_termination,
        classed=True,
        resize_factor=amod.config.demand_resize_factor,
    )

    # Count number of trips per step
    step_trip_count = [len(trips) for trips in step_trip_od_list]

    trip_demand_dict[tripdata_path] = (step_trip_list, step_trip_count)

    return trip_demand_dict[tripdata_path]


def sim(plot_track, config):

    step_delay = PlotTrack.STEP_DELAY
    enable_plot = PlotTrack.ENABLE_PLOT
    SKIP_STEPS = 1
    ENABLE_HIRING = True
    contract_duration_h = 8

    # ---------------------------------------------------------------- #
    # Episodes ####################################################### #
    # ---------------------------------------------------------------- #
    episodes = 10000
    amod = AmodNetworkHired(config)
    episode_log = EpisodeLog(config=config, adp=amod.adp)
    plot_track.set_env(amod)

    # ---------------------------------------------------------------- #
    # Plot centers and guidelines #################################### #
    # ---------------------------------------------------------------- #
    if enable_plot:
        plot_track.plot_centers(
            amod.points,
            nw.Point.levels,
            nw.Point.levels[config.demand_center_level],
            nw.Point.levels[config.neighborhood_level],
            show_sp_lines=PlotTrack.SHOW_SP_LINES,
            show_lines=PlotTrack.SHOW_LINES,
        )

    print(f"Loading demand scenario '{config.demand_scenario}'...")

    try:
        # Load last episode
        episode_log.load_progress()

    except Exception as e:
        print(f"No previous episodes were saved (Exception: '{e}').")

    # ---------------------------------------------------------------- #
    # Process demand ################################################# #
    # ---------------------------------------------------------------- #

    if config.demand_scenario == SCENARIO_UNBALANCED:

        origins, destinations = episode_log.get_od_lists(amod)

        # Get demand pattern from NY city
        step_trip_count = get_trip_count_step(
            TRIP_FILES[0],
            step=config.time_increment,
            multiply_for=config.demand_resize_factor,
            earliest_step=config.demand_earliest_step_min,
            max_steps=config.demand_max_steps,
        )

    # ---------------------------------------------------------------- #
    # Experiment ##################################################### #
    # ---------------------------------------------------------------- #

    # Loop all episodes, pick up trips, and learn where they are
    for n in range(episode_log.n, episodes):

        if config.demand_scenario == SCENARIO_UNBALANCED:

            # Sample ods for iteration n
            step_trip_list = get_trips_random_ods(
                amod.points,
                step_trip_count,
                offset_start=amod.config.offset_repositioning,
                offset_end=amod.config.offset_termination,
                origins=origins,
                destinations=destinations,
                classed=True,
            )

        elif config.demand_scenario == SCENARIO_NYC:

            trips_file_path = choice(TRIP_FILES)

            print(f"Processing demand file '{trips_file_path}'...")

            step_trip_list, step_trip_count = get_ny_demand(
                amod, trips_file_path
            )

        print(
            f"### DEMAND ###"
            f" - min: {min(step_trip_count)}"
            f" - max: {max(step_trip_count)}"
        )

        plot_track.opt_episode = n

        # Start saving data of each step in the adp_network
        step_log = StepLog(amod)

        # Resetting environment
        amod.reset()

        # ------------------------------------------------------------ #
        # Plot fleet current status ################################## #
        # ------------------------------------------------------------ #
        if enable_plot:

            # Computing initial timestep
            plot_track.compute_movements(0)

        # Iterate through all steps and match requests to cars
        for step, trips in enumerate(deepcopy(step_trip_list)):

            if enable_plot:
                # Update optimization time step
                plot_track.opt_step = step

                # Assign trips at step
                plot_track.trips_dict[step] = vi.compute_trips(trips)

            # Loop cars and update their current status as well as the
            # the list of available vehicles.

            # Show time step statistics
            if step % SKIP_STEPS == 0:
                step_log.show_info()
            # print(f"### STEP {step:>4} ###############################")

            # Compute fleet status
            step_log.compute_fleet_status()

            # What each vehicle is doing?
            # if len(trips) == 0:
            #     amod.print_fleet_stats(filter_status=[Car.ASSIGN])
            # amod.print_fleet_stats(filter_status=[])

            # ######################################################## #
            # TIME INCREMENT HAS PASSED ############################## #
            # ######################################################## #

            # ***Change available and available_hired
            amod.update_fleet_status(step)

            # Stats summary
            # print(" - Pre-decision statuses:")
            # amod.print_fleet_stats_summary()
            if ENABLE_HIRING:

                hired_cars = []

                hired_cars = hire_cars_trip_regions(
                    amod, trips, contract_duration_h, step
                )
                # hired_cars = hire_cars_centers(amod, contract_duration_h, step)

                # Add hired fleet to model
                amod.hired_cars.extend(hired_cars)
                amod.available_hired.extend(hired_cars)

            # Adding hired caras
            # print(f"Hiring {len(hired_cars)} cars.")

            # Optimize
            revenue, serviced, rejected = adp_network_hired2(
                amod,
                trips,
                step + 1,
                sq_guarantee=True,
                charge=False,
                myopic=False,
                value_function_update=match.WEIGHTED_UPDATE,
                episode=n,
                universal_service=True,
                # agg_level=0,
                # log_path=config.folder_mip,
                # sq_guarantee=False,
            )

            # Virtual hired cars are discarded
            if ENABLE_HIRING:

                discarded = amod.discard_excess_hired()
                # print(f"{discarded} cars discarded.")
            # -------------------------------------------------------- #
            # Update log with iteration ############################## #
            # -------------------------------------------------------- #
            step_log.add_record(revenue, serviced, rejected)

            # Stats summary
            # print(" - Post-decision statuses")
            # amod.print_fleet_stats_summary()

            # -------------------------------------------------------- #
            # Plotting fleet activity ################################ #
            # -------------------------------------------------------- #
            # print("Computing movements...")
            if enable_plot:
                plot_track.compute_movements(step + 1)
                # print("Finished computing...")

                time.sleep(step_delay)

        episode_log.compute_episode(step_log, weights=amod.adp.get_weights())

        # -------------------------------------------------------------#
        # Compute episode info #########################################
        # -------------------------------------------------------------#
        print(
            f"####### "
            f"[Episode {n:>5}] "
            f"- {episode_log.last_episode_stats()} "
            f"#######"
        )

    episode_log.compute_learning()


run_plot.start_animation(sim)
