import os
import sys
from copy import deepcopy
import time
import random

# Adding project folder to import modules
root = os.getcwd().replace("\\", "/")
sys.path.append(root)

from mod.env.matching import service_trips
import mod.util.log_aux as la

from mod.env.amod.AmodNetworkHired import AmodNetworkHired
from mod.env.visual import StepLog, EpisodeLog
import mod.env.adp.adp as adp

import mod.env.visual as vi
from mod.env.config import (
    ConfigNetwork,
    SCENARIO_UNBALANCED,
    SCENARIO_NYC,
    TRIP_FILES,
)

import mod.env.config as conf


from mod.env.car import Car, HiredCar
from mod.env.trip import get_trip_count_step, get_trips_random_ods
import mod.env.trip as tp
import mod.env.network as nw

from mod.env.simulator import PlotTrack

# Reproducibility of the experiments
random.seed(1)


def get_sim_config(update_dict):

    config = ConfigNetwork()

    # Pull graph info
    region, label, node_count, center_count, edge_count, region_type = (
        nw.query_info()
    )

    info = (
        "##############################################################"
        f"\n### Region: {region} G(V={node_count}, E={edge_count})"
        f"\n### Center count: {center_count}"
        f"\n### Region type: {region_type}"
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
            ConfigNetwork.TEST_LABEL: "SIM",
            # Fleet
            ConfigNetwork.FLEET_SIZE: 50,
            ConfigNetwork.FLEET_START: conf.FLEET_START_LAST,
            ConfigNetwork.BATTERY_LEVELS: 1,
            # Time - Increment (min)
            ConfigNetwork.TIME_INCREMENT: 1,
            ConfigNetwork.OFFSET_REPOSIONING: 15,
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
            ConfigNetwork.STEP_SECONDS: 15,
            # Cars can access locations within region centers
            # established in which neighborhood level?
            ConfigNetwork.NEIGHBORHOOD_LEVEL: 0,
            # Cars can rebalance to neighbor centers of level:
            # Why not max rebalance level?
            ConfigNetwork.REBALANCE_LEVEL: (0, 1, 2, 3, 4, 5),
            # Cars rebalance to up to #region centers
            ConfigNetwork.N_CLOSEST_NEIGHBORS: (4, 4, 4, 2, 2, 1),
            # ConfigNetwork.REBALANCE_REACH: 2,
            ConfigNetwork.REBALANCE_MULTILEVEL: False,
            # Aggregation (temporal, spatial, contract, car type)
            ConfigNetwork.AGGREGATION_LEVELS: [
                adp.AggLevel(
                    temporal=adp.DISAGGREGATE,
                    spatial=adp.DISAGGREGATE,
                    contract=adp.DISAGGREGATE,
                    car_type=adp.DISAGGREGATE,
                    car_origin=adp.DISAGGREGATE,
                ),
                adp.AggLevel(
                    temporal=1,
                    spatial=adp.DISAGGREGATE,
                    contract=adp.DISCARD,
                    car_type=adp.DISAGGREGATE,
                    car_origin=adp.DISAGGREGATE,
                ),
                adp.AggLevel(
                    temporal=1,
                    spatial=1,
                    contract=adp.DISCARD,
                    car_type=adp.DISAGGREGATE,
                    car_origin=1,
                ),
                adp.AggLevel(
                    temporal=1,
                    spatial=2,
                    contract=adp.DISCARD,
                    car_type=adp.DISAGGREGATE,
                    car_origin=2,
                ),
                adp.AggLevel(
                    temporal=1,
                    spatial=3,
                    contract=adp.DISCARD,
                    car_type=adp.DISAGGREGATE,
                    car_origin=3,
                ),
                adp.AggLevel(
                    temporal=1,
                    spatial=4,
                    contract=adp.DISCARD,
                    car_type=adp.DISAGGREGATE,
                    car_origin=4,
                ),
                adp.AggLevel(
                    temporal=1,
                    spatial=5,
                    contract=adp.DISCARD,
                    car_type=adp.DISAGGREGATE,
                    car_origin=5,
                ),
                adp.AggLevel(
                    temporal=1,
                    spatial=6,
                    contract=adp.DISCARD,
                    car_type=adp.DISAGGREGATE,
                    car_origin=6,
                ),
            ],
            ConfigNetwork.LEVEL_TIME_LIST: [1, 2, 3],
            ConfigNetwork.LEVEL_CAR_ORIGIN: {
                Car.TYPE_FLEET: {adp.DISCARD: adp.DISCARD},
                Car.TYPE_HIRED: {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6},
                Car.TYPE_TO_HIRE: {adp.DISCARD: adp.DISCARD},
            },
            ConfigNetwork.LEVEL_CAR_TYPE: {
                Car.TYPE_FLEET: {adp.DISCARD: Car.TYPE_FLEET},
                Car.TYPE_HIRED: {adp.DISCARD: Car.TYPE_FLEET},
                Car.TYPE_TO_HIRE: {adp.DISCARD: Car.TYPE_FLEET},
            },
            ConfigNetwork.LEVEL_CONTRACT_DURATION: {
                Car.TYPE_FLEET: {adp.DISCARD: Car.INFINITE_CONTRACT_DURATION},
                Car.TYPE_HIRED: {
                    adp.DISAGGREGATE: adp.DISAGGREGATE,
                    adp.DISCARD: Car.INFINITE_CONTRACT_DURATION,
                },
                Car.TYPE_TO_HIRE: {
                    adp.DISCARD: Car.INFINITE_CONTRACT_DURATION
                },
            },
            ConfigNetwork.LEVEL_DIST_LIST: [
                0,
                30,
                60,
                120,
                150,
                240,
                600,
            ],  # , 300],
            # From which region center levels cars are hired
            ConfigNetwork.LEVEL_RC: 5,
            # Trips and cars have to match in these levels
            # 9 = 990 and 10=1140
            ConfigNetwork.MATCHING_LEVELS: (0, 0),
            # How many levels separated by step secresize_factorc
            # LEVEL_DIST_LIST must be filled (1=disaggregate)
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
            ConfigNetwork.DEMAND_CENTER_LEVEL: 0,
            # Demand scenario
            ConfigNetwork.DEMAND_SCENARIO: SCENARIO_NYC,
            ConfigNetwork.TRIP_BASE_FARE: {
                tp.ClassedTrip.SQ_CLASS_1: 400,
                tp.ClassedTrip.SQ_CLASS_2: 400,
            },
            # -------------------------------------------------------- #
            # LEARNING ############################################### #
            # -------------------------------------------------------- #
            ConfigNetwork.DISCOUNT_FACTOR: 0.05,
            ConfigNetwork.HARMONIC_STEPSIZE: 1,
            ConfigNetwork.STEPSIZE_CONSTANT: 0.1,
            ConfigNetwork.STEPSIZE_RULE: adp.STEPSIZE_MCCLAIN,
            # ConfigNetwork.STEPSIZE_RULE: adp.STEPSIZE_CONSTANT,
            # -------------------------------------------------------- #
            # HIRING ################################################# #
            # -------------------------------------------------------- #
            ConfigNetwork.CONTRACT_DURATION_LEVEL: 15,
            ConfigNetwork.CONGESTION_PRICE: 1000,
            # -------------------------------------------------------- #
            ConfigNetwork.MATCH_METHOD: ConfigNetwork.MATCH_DISTANCE,
            ConfigNetwork.MATCH_LEVEL: 2,
        }
    )

    config.update(update_dict)

    return config


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


def hire_cars_centers(amod, contract_duration_h, step, rc_level=2):
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
        for c in amod.points_level[rc_level]
        # if random.random() < 1
    ]

    return hired_cars


def alg_adp(
    plot_track,
    config,
    # PLOT ########################################################### #
    step_delay=PlotTrack.STEP_DELAY,
    episodes=30,
    enable_charging=False,
    is_myopic=False,
    # LOG ############################################################ #
    skip_steps=1,
    # HIRING ######################################################### #
    enable_hiring=False,
    contract_duration_h=2,
    sq_guarantee=False,
    universal_service=False,
    # TRIPS ########################################################## #
    classed_trips=True,
    # Progress track
    save_progress=True,
    # Create service rate and fleet status plots for each iteration
    plots=False,
    # Save .csv files for each iteration with fleet and demand statuses
    # throughtout all time steps
    save_df=False,
    # Save total reward, total service rate, and weights after iteration
    save_overall_stats=True,
    # Update value functions (dictionary in progress.npy file)
    # after each iteration
    save_learning=True,
):
    # ---------------------------------------------------------------- #
    # Episodes ####################################################### #
    # ---------------------------------------------------------------- #

    amod = AmodNetworkHired(config)
    episode_log = EpisodeLog(config=config, adp=amod.adp)
    if plot_track:
        plot_track.set_env(amod)

    # Logging events
    logger = la.get_logger(
        config.label,
        level_file=la.DEBUG,
        level_console=la.INFO,
        log_file=config.log_path,
    )

    # ---------------------------------------------------------------- #
    # Plot centers and guidelines #################################### #
    # ---------------------------------------------------------------- #
    if plot_track:
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

            # Load a random .csv file with trips from NYC
            trips_file_path = random.choice(TRIP_FILES)

            # print(f"Processing demand file '{trips_file_path}'...")

            step_trip_list, step_trip_count = tp.get_ny_demand(
                config, trips_file_path, amod.points
            )

        logger.debug(
            "##################################"
            f" Iteration {n:04} "
            f"- Demand (min={min(step_trip_count)}"
            f", max={max(step_trip_count)})"
            "##################################"
        )

        if plot_track:
            plot_track.opt_episode = n

        # Start saving data of each step in the adp_network
        step_log = StepLog(amod)

        # Resetting environment
        amod.reset()

        # ------------------------------------------------------------ #
        # Plot fleet current status ################################## #
        # ------------------------------------------------------------ #
        if plot_track:

            # Computing initial timestep
            plot_track.compute_movements(0)

        start_time = time.time()

        # Iterate through all steps and match requests to cars
        for step, trips in enumerate(deepcopy(step_trip_list)):

            logger.debug(
                "###########################################"
                f" (step={step}, trips={len(trips)}) "
                "###########################################"
            )

            if plot_track:
                # Update optimization time step
                plot_track.opt_step = step

                # Create trip dictionary of coordinates
                plot_track.trips_dict[step] = vi.compute_trips(trips)

            if save_progress:
                logger.debug("  - Computing fleet status...")
                # Compute fleet status after making decision in step - 1
                # What each car is doing when trips are arriving?
                step_log.compute_fleet_status(step)

            # ######################################################## #
            # TIME INCREMENT HAS PASSED ############################## #
            # ######################################################## #

            # Loop cars and update their current status as well as the
            # the list of available vehicles (change available and
            # available_hired)
            amod.update_fleet_status(step + 1)

            # What each vehicle is doing after update?
            la.log_fleet_activity(
                config.label,
                step + 1,
                skip_steps,
                step_log,
                filter_status=[],
                msg="post update",
            )

            if enable_hiring:

                hired_cars = []

                if trips:

                    hired_cars = hire_cars_trip_regions(
                        amod, trips, contract_duration_h, step
                    )

                    logger.debug(
                        f"**** Hiring {len(hired_cars)} in the trip regions."
                    )

                else:

                    hired_cars = hire_cars_centers(
                        amod,
                        contract_duration_h,
                        step,
                        rc_level=config.level_rc,
                    )

                    logger.debug(
                        f"**** Hiring {len(hired_cars)} in region centers."
                    )

                # Add hired fleet to model
                amod.hired_cars.extend(hired_cars)
                amod.available_hired.extend(hired_cars)

            # Optimize
            revenue, serviced, rejected = service_trips(
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
                # # Save mip .lp and .log of iteration n
                # log_iteration=n,
                # # Use hierarchical aggregation to update values
                # agg_level=1,
                # Penalize rebalancing by subtracting the potential
                # profit accrued by staying still during the rebalance
                # process.
                penalize_rebalance=True,
            )

            # What each vehicle is doing?
            la.log_fleet_activity(
                config.label,
                step,
                skip_steps,
                step_log,
                filter_status=[],
                msg="after decision",
            )

            # Virtual hired cars are discarded
            if enable_hiring:

                logger.debug(
                    f"Total hired: {len(amod.hired_cars):4} "
                    f"(available={len(amod.available_hired)})"
                )

                amod.discard_excess_hired()

                logger.debug(
                    f"Total hired: {len(amod.hired_cars):4} "
                    f"(available={len(amod.available_hired)})"
                    " AFTER DISCARDING"
                )

            # -------------------------------------------------------- #
            # Update log with iteration ############################## #
            # -------------------------------------------------------- #
            step_log.add_record(revenue, serviced, rejected)

            # -------------------------------------------------------- #
            # Plotting fleet activity ################################ #
            # -------------------------------------------------------- #

            if plot_track:

                logger.info("Computing movements...")

                plot_track.compute_movements(step + 1)

                logger.info("Finished computing...")

                time.sleep(step_delay)

        amod.update_fleet_status(step + 1)

        # -------------------------------------------------------------#
        # Compute episode info #########################################
        # -------------------------------------------------------------#

        logger.debug("  - Computing iteration...")
        episode_log.compute_episode(
            step_log,
            time.time() - start_time,
            weights=amod.adp.get_weights(),
            save_df=save_df,
            plots=plots,
            save_learning=save_learning,
            save_overall_stats=save_overall_stats,
        )

        logger.info(
            f"####### "
            f"[Episode {n:>5}] "
            f"- {episode_log.last_episode_stats()} "
            f"#######"
        )

    # Plot overall performance (reward, service rate, and weights)
    episode_log.compute_learning()

    return amod.adp.reward


if __name__ == "__main__":

    args = sys.argv[1:]
    print(args)
    try:
        test_label = args[0]
    except:
        test_label = "500DISA"

    try:
        hire = "-hire" in args
        print("Hiring!!!")
    except:
        print("Not hiring")
        hire = False

    print("###### STARTING EXPERIMENTS")
    start_config = get_sim_config(
        {
            # ConfigNetwork.LEVEL_DIST_LIST: [0, 60, 120, 300],
            # ConfigNetwork.LEVEL_DIST_LIST: [0, 45, 60, 90, 150],
            ConfigNetwork.TEST_LABEL: test_label
        }
    )

    alg_adp(
        None,
        start_config,
        episodes=1000,
        contract_duration_h=2,
        sq_guarantee=hire,
        enable_hiring=hire,
        universal_service=hire,
    )
