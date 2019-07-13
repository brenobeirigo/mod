import os
import sys
import numpy as np
from pprint import pprint
from threading import Thread
from copy import deepcopy
import time
import random

# Adding project folder to import modules
root = os.getcwd().replace("\\", "/")
sys.path.append(root)

from mod.env.amod.AmodNetworkHired import AmodNetworkHired
from mod.env.amod.AmodNetwork import AmodNetwork
from mod.env.visual import StepLog, EpisodeLog
import mod.env.adp.adp as adp

import mod.env.visual as vi
from mod.env.config import (
    ConfigNetwork,
    FOLDER_OUTPUT,
    NY_TRIPS_EXCERPT_DAY,
    SCENARIO_UNBALANCED,
    SCENARIO_NYC,
    TRIP_FILES,
)

import mod.env.config as conf

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

# Reproducibility of the experiments
random.seed(1)


def get_sim_config():

    config = ConfigNetwork()
    # Pull graph info
    region, label, node_count, center_count, edge_count = nw.query_info()

    info = (
        "##############################################################"
        f"\n### Region: {region} G(V={node_count}, E={edge_count})"
        f"\n### Center count: {center_count}"
    )

    level_id_count_dict = {
        int(level): (i + 1, count)
        for i, (level, count) in enumerate(center_count.items())
    }

    level_id_count_dict[0] = (0, node_count)

    print(info)
    # ---------------------------------------------------------------- #
    # Amod environment ############################################### #
    # ---------------------------------------------------------------- #

    config.update(
        {
            ConfigNetwork.TEST_LABEL: "20KM_200_0.1_PUNISH",
            # Fleet
            ConfigNetwork.FLEET_SIZE: 300,
            ConfigNetwork.FLEET_START: conf.FLEET_START_LAST,
            ConfigNetwork.BATTERY_LEVELS: 1,
            # Time - Increment (min)
            ConfigNetwork.TIME_INCREMENT: 1,
            ConfigNetwork.OFFSET_REPOSIONING: 15,
            ConfigNetwork.OFFSET_TERMINATION: 15,
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
            ConfigNetwork.LEVEL_DIST_LIST: [
                0,
                60,
                90,
                120,
                180,
                270,
                750,
                1140,
            ],
            # Trips and cars have to match in these levels
            # 9 = 990 and 10=1140
            ConfigNetwork.MATCHING_LEVELS: (6, 7),
            # How many levels separated by step secresize_factorc
            # LEVEL_DIST_LIST must be filled (1=disaggregate)
            ConfigNetwork.AGGREGATION_LEVELS: 6,
            ConfigNetwork.SPEED: 20,
            # -------------------------------------------------------- #
            # DEMAND ################################################# #
            # -------------------------------------------------------- #
            ConfigNetwork.DEMAND_TOTAL_HOURS: 4,
            ConfigNetwork.DEMAND_EARLIEST_HOUR: 5,
            ConfigNetwork.DEMAND_RESIZE_FACTOR: 0.1,
            ConfigNetwork.DEMAND_SAMPLING: True,
            # Demand spawn from how many centers?
            ConfigNetwork.ORIGIN_CENTERS: 3,
            # Demand arrives in how many centers?
            ConfigNetwork.DESTINATION_CENTERS: 3,
            # OD level extension
            ConfigNetwork.DEMAND_CENTER_LEVEL: 4,
            # Demand scenario
            ConfigNetwork.DEMAND_SCENARIO: SCENARIO_NYC,
            ConfigNetwork.TRIP_BASE_FARE: {
                tp.ClassedTrip.SQ_CLASS_1: 4,
                tp.ClassedTrip.SQ_CLASS_2: 2,
            },
            # -------------------------------------------------------- #
            # LEARNING ############################################### #
            # -------------------------------------------------------- #
            ConfigNetwork.DISCOUNT_FACTOR: 0.02,
            ConfigNetwork.HARMONIC_STEPSIZE: 1,
            ConfigNetwork.STEPSIZE_CONSTANT: 0.1,
            ConfigNetwork.STEPSIZE_RULE: adp.STEPSIZE_MCCLAIN,
            # ConfigNetwork.STEPSIZE_RULE: adp.STEPSIZE_CONSTANT,
            # -------------------------------------------------------- #
            # HIRING ################################################# #
            # -------------------------------------------------------- #
            ConfigNetwork.CONTRACT_DURATION_LEVEL: 15,
            ConfigNetwork.CONGESTION_PRICE: 10,
            # -------------------------------------------------------- #
            ConfigNetwork.MATCH_METHOD: ConfigNetwork.MATCH_DISTANCE,
            ConfigNetwork.MATCH_LEVEL: 2,
        }
    )
    return config


start_config = get_sim_config()
run_plot = PlotTrack(start_config)


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
        if random.random() < 1
    ]

    return hired_cars


def alg_adp(
    plot_track,
    config,
    episodes=200,
    enable_charging=False,
    is_myopic=False,
    # LOG ############################################################ #
    skip_steps=0,
    # PLOT ########################################################### #
    step_delay=PlotTrack.STEP_DELAY,
    enable_plot=False,
    # HIRING ######################################################### #
    enable_hiring=False,
    contract_duration_h=2,
    sq_guarantee=False,
    universal_service=False,
    # TRIPS ########################################################## #
    classed_trips=True,
):
    # ---------------------------------------------------------------- #
    # Episodes ####################################################### #
    # ---------------------------------------------------------------- #

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

        if config.demand_scenario == conf.SCENARIO_UNBALANCED:

            # Sample ods for iteration n
            step_trip_list = get_trips_random_ods(
                amod.points,
                step_trip_count,
                offset_start=amod.config.offset_repositioning,
                offset_end=amod.config.offset_termination,
                origins=origins,
                destinations=destinations,
                classed=classed_trips,
            )

        elif config.demand_scenario == conf.SCENARIO_NYC:

            trips_file_path = random.choice(TRIP_FILES)

            # print(f"Processing demand file '{trips_file_path}'...")

            step_trip_list, step_trip_count = tp.get_ny_demand(
                config, trips_file_path, amod.points
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

        # print("Position cars:")
        # pprint([c.point for c in amod.cars])

        # ------------------------------------------------------------ #
        # Plot fleet current status ################################## #
        # ------------------------------------------------------------ #
        if enable_plot:

            # Computing initial timestep
            plot_track.compute_movements(0)

        start_time = time.time()

        # Iterate through all steps and match requests to cars
        for step, trips in enumerate(deepcopy(step_trip_list)):

            if enable_plot:
                # Update optimization time step
                plot_track.opt_step = step

                # Assign trips at step
                plot_track.trips_dict[step] = vi.compute_trips(trips)

            # Loop cars and update their current status as well as the
            # the list of available vehicles.

            # print(f"### STEP {step:>4} ###############################")

            # Compute fleet status after making decision in step - 1
            # What each car is doing when trips are arriving?
            step_log.compute_fleet_status(step)

            if skip_steps > 0 and step % skip_steps == 0:
                step_log.show_info()
                # What each vehicle is doing?
                # if len(trips) == 0:
                #     amod.print_fleet_stats(filter_status=[Car.ASSIGN])
                amod.print_fleet_stats(filter_status=[])

            # ######################################################## #
            # TIME INCREMENT HAS PASSED ############################## #
            # ######################################################## #

            # ***Change available and available_hired
            amod.update_fleet_status(step + 1)

            # Stats summary
            # print(" - Pre-decision statuses:")
            # amod.print_fleet_stats_summary()
            if enable_hiring:

                hired_cars = []
                if trips:

                    hired_cars = hire_cars_trip_regions(
                        amod, trips, contract_duration_h, step
                    )
                    # print("Hired:", len(hired_cars))
                else:

                    hired_cars = hire_cars_centers(
                        amod, contract_duration_h, step
                    )

                # hired_cars = hire_cars_centers(amod, contract_duration_h, step)

                # Add hired fleet to model
                amod.hired_cars.extend(hired_cars)
                amod.available_hired.extend(hired_cars)

            # Adding hired caras
            # print(f"Hiring {len(hired_cars)} cars.")

            # Optimize
            revenue, serviced, rejected = adp_network_hired2(
                # Amod environment with configuration file
                amod,
                # Trips to be matched
                trips,
                # Service step (+1 trip placement step)
                step + 1,
                # Guarantee lowest pickup delay for a share of users
                sq_guarantee=sq_guarantee,
                # All users are picked up
                universal_service=universal_service,
                # Allow recharging
                charge=enable_charging,
                # If True, does not use learned information
                myopic=is_myopic,
                # Save mip .lp and .log of iteration n
                # log_iteration=n,
                # agg_level=1,
                # Use hierarchical aggregation to update values
                value_function_update=match.WEIGHTED_UPDATE,
            )

            # Virtual hired cars are discarded
            if enable_hiring:

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

        amod.update_fleet_status(step + 1)

        episode_log.compute_episode(
            step_log, time.time() - start_time, weights=amod.adp.get_weights()
        )

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

    return amod.adp.reward


run_plot.start_animation(alg_adp)

