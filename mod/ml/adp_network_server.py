import os
import sys
from copy import deepcopy
import time
import random
import numpy as np
import itertools

# Adding project folder to import modules
import mod.env.Point

root = os.getcwd().replace("\\", "/")
sys.path.append(root)

from mod.env.matching import (
    service_trips,
    optimal_rebalancing,
    play_decisions,
    mpc,
)
import mod.util.log_util as la

from mod.env.amod.AmodNetworkHired import (
    AmodNetworkHired,
    # exe_times,
    # decision_post,
)
from mod.env.visual import StepLog, EpisodeLog
import mod.env.adp.adp as adp

import mod.env.visual as vi
from mod.env.config import ConfigNetwork

import mod.env.config as conf

from mod.env.fleet.Car import Car
from mod.env.demand.trip_util import get_trip_count_step, get_trips_random_ods
import mod.env.demand.trip_util as tp
import mod.env.network as nw

from mod.env.simulator import PlotTrack


# Reproducibility of the experiments
random.seed(1)


def get_sim_config(update_dict):

    config = ConfigNetwork()

    # Pull graph info
    (
        region,
        label,
        node_count,
        center_count,
        edge_count,
        region_type,
    ) = nw.query_info()

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
            ConfigNetwork.TEST_LABEL: "HUGE",
            # Fleet
            ConfigNetwork.FLEET_SIZE: 5,
            ConfigNetwork.FLEET_START: conf.FLEET_START_LAST,
            ConfigNetwork.BATTERY_LEVELS: 1,
            # Time - Increment (min)
            ConfigNetwork.TIME_INCREMENT: 1,
            ConfigNetwork.OFFSET_REPOSITIONING_MIN: 15,
            ConfigNetwork.OFFSET_TERMINATION_MIN: 60,
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
            # Penalize rebalancing by subtracting the potential
            # profit accrued by staying still during the rebalance
            # process.
            ConfigNetwork.PENALIZE_REBALANCE: False,
            # Cars rebalance to up to #region centers at each level
            # CAUTION! Change max_neighbors in tenv application if > 4
            ConfigNetwork.N_CLOSEST_NEIGHBORS: ((0, 8), (1, 8)),
            # Only used when max idle step count is not None
            ConfigNetwork.N_CLOSEST_NEIGHBORS_EXPLORE: ((2, 16), (3, 16)),
            # ConfigNetwork.REBALANCE_REACH: 2,
            ConfigNetwork.REBALANCE_MULTILEVEL: False,
            ConfigNetwork.LEVEL_DIST_LIST: [0, 150, 300, 600],
            # Aggregation (temporal, spatial, contract, car type)
            # # FAVs and PAVS BEST HIERARCHICAL AGGREGATION
            # ConfigNetwork.AGGREGATION_LEVELS: [
            #     adp.AggLevel(
            #         temporal=1,
            #         spatial=0,
            #         battery=adp.DISAGGREGATE,
            #         contract=2,
            #         car_type=adp.DISAGGREGATE,
            #         car_origin=adp.DISAGGREGATE,
            #     ),
            #     adp.AggLevel(
            #         temporal=1,
            #         spatial=0,
            #         battery=adp.DISAGGREGATE,
            #         contract=3,
            #         car_type=adp.DISAGGREGATE,
            #         car_origin=3,
            #     ),
            #     adp.AggLevel(
            #         temporal=1,
            #         spatial=0,
            #         battery=adp.DISAGGREGATE,
            #         contract=adp.DISCARD,
            #         car_type=adp.DISAGGREGATE,
            #         car_origin=adp.DISCARD,
            #     ),
            #     adp.AggLevel(
            #         temporal=3,
            #         spatial=2,
            #         battery=adp.DISAGGREGATE,
            #         contract=adp.DISCARD,
            #         car_type=adp.DISAGGREGATE,
            #         car_origin=adp.DISCARD,
            #     ),
            #     adp.AggLevel(
            #         temporal=3,
            #         spatial=3,
            #         battery=adp.DISAGGREGATE,
            #         contract=adp.DISCARD,
            #         car_type=adp.DISAGGREGATE,
            #         car_origin=adp.DISCARD,
            #     ),
            # ],
            # #### ONLY PAVs - BEST PERFORMANCE HiERARCHICAL AGGREGATION
            # ConfigNetwork.AGGREGATION_LEVELS: [
            #     adp.AggLevel(
            #         temporal=1,
            #         spatial=adp.DISAGGREGATE,
            #         battery=adp.DISAGGREGATE,
            #         contract=adp.DISCARD,
            #         car_type=adp.DISAGGREGATE,
            #         car_origin=adp.DISCARD,
            #     ),
            #     adp.AggLevel(
            #         temporal=3,
            #         spatial=2,
            #         battery=adp.DISAGGREGATE,
            #         contract=adp.DISCARD,
            #         car_type=adp.DISAGGREGATE,
            #         car_origin=adp.DISCARD,
            #     ),
            #     adp.AggLevel(
            #         temporal=3,
            #         spatial=3,
            #         battery=adp.DISAGGREGATE,
            #         contract=adp.DISCARD,
            #         car_type=adp.DISAGGREGATE,
            #         car_origin=adp.DISCARD,
            #     ),
            # ],
            # #### ONLY PAVs - BEST PERFORMANCE HiERARCHICAL AGGREGATION
            ConfigNetwork.AGGREGATION_LEVELS: [
                # adp.AggLevel(
                #     temporal=1,
                #     spatial=adp.DISAGGREGATE,
                #     battery=adp.DISAGGREGATE,
                #     contract=adp.DISCARD,
                #     car_type=adp.DISAGGREGATE,
                #     car_origin=adp.DISCARD,
                # ),
                adp.AggLevel(
                    temporal=0,
                    spatial=1,
                    battery=adp.DISAGGREGATE,
                    contract=adp.DISCARD,
                    car_type=adp.DISAGGREGATE,
                    car_origin=adp.DISCARD,
                ),
                adp.AggLevel(
                    temporal=1,
                    spatial=2,
                    battery=adp.DISAGGREGATE,
                    contract=adp.DISCARD,
                    car_type=adp.DISAGGREGATE,
                    car_origin=adp.DISCARD,
                ),
                adp.AggLevel(
                    temporal=2,
                    spatial=3,
                    battery=adp.DISAGGREGATE,
                    contract=adp.DISCARD,
                    car_type=adp.DISAGGREGATE,
                    car_origin=adp.DISCARD,
                ),
            ],
            # ConfigNetwork.LEVEL_TIME_LIST: [0.5, 1, 2, 3],
            # TIME LIST multiplies the time increment. E.g.:
            # time increment=5 then [1,2] = [5, 10]
            # time increment=1 then [1,3] = [1, 3]
            ConfigNetwork.LEVEL_TIME_LIST: [1, 2, 3],
            ConfigNetwork.LEVEL_CAR_ORIGIN: {
                Car.TYPE_FLEET: {adp.DISCARD: adp.DISCARD},
                Car.TYPE_HIRED: {0: 0, 1: 1, 2: 2, 3: 3, 4: 4},
            },
            ConfigNetwork.LEVEL_CAR_TYPE: {
                Car.TYPE_FLEET: {
                    adp.DISAGGREGATE: Car.TYPE_FLEET,
                    adp.DISCARD: Car.TYPE_FLEET,
                },
                Car.TYPE_HIRED: {
                    adp.DISAGGREGATE: Car.TYPE_HIRED,
                    adp.DISCARD: Car.TYPE_FLEET,
                },
            },
            ConfigNetwork.LEVEL_CONTRACT_DURATION: {
                Car.TYPE_FLEET: {adp.DISCARD: Car.INFINITE_CONTRACT_DURATION},
                Car.TYPE_HIRED: {
                    adp.DISAGGREGATE: adp.CONTRACT_DISAGGREGATE,
                    1: adp.CONTRACT_L1,
                    2: adp.CONTRACT_L2,
                    3: adp.CONTRACT_L3,
                    adp.DISCARD: Car.INFINITE_CONTRACT_DURATION,
                },
            },
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
            ConfigNetwork.DEMAND_SCENARIO: conf.SCENARIO_NYC,
            ConfigNetwork.TRIP_REJECTION_PENALTY: (("A", 0), ("B", 0)),
            ConfigNetwork.TRIP_BASE_FARE: (("A", 4.8), ("B", 2.4)),
            ConfigNetwork.TRIP_DISTANCE_RATE_KM: (("A", 1), ("B", 1)),
            ConfigNetwork.TRIP_TOLERANCE_DELAY_MIN: (("A", 5), ("B", 5)),
            ConfigNetwork.TRIP_MAX_PICKUP_DELAY: (("A", 5), ("B", 10)),
            ConfigNetwork.TRIP_CLASS_PROPORTION: (("A", 0), ("B", 1)),
            # -------------------------------------------------------- #
            # LEARNING ############################################### #
            # -------------------------------------------------------- #
            ConfigNetwork.DISCOUNT_FACTOR: 0.1,
            ConfigNetwork.HARMONIC_STEPSIZE: 1,
            ConfigNetwork.STEPSIZE_CONSTANT: 0.1,
            ConfigNetwork.STEPSIZE_RULE: adp.STEPSIZE_MCCLAIN,
            # ConfigNetwork.STEPSIZE_RULE: adp.STEPSIZE_CONSTANT,
            # -------------------------------------------------------- #
            # HIRING ################################################# #
            # -------------------------------------------------------- #
            ConfigNetwork.CONTRACT_DURATION_LEVEL: 1,
            ConfigNetwork.CONGESTION_PRICE: 0,
            # -------------------------------------------------------- #
            ConfigNetwork.MATCH_METHOD: ConfigNetwork.MATCH_DISTANCE,
            ConfigNetwork.MATCH_LEVEL: 2,
            ConfigNetwork.MAX_TARGETS: 1000,
        }
    )

    config.update(update_dict)

    return config


def alg_adp(
    plot_track,
    config,
    # PLOT ########################################################### #
    step_delay=PlotTrack.STEP_DELAY,
    # LOG ############################################################ #
    skip_steps=1,
    # TRIPS ########################################################## #
    classed_trips=True,
    # Create service rate and fleet status plots for each iteration
    save_plots=True,
    # Save .csv files for each iteration with fleet and demand statuses
    # throughtout all time steps
    save_df=True,
    # Save total reward, total service rate, and weights after iteration
    save_overall_stats=True,
    log_config_dict={
        # Write each vehicles status
        la.LOG_FLEET_ACTIVITY: False,
        # Write profit, service level, # trips, car/satus count
        la.LOG_STEP_SUMMARY: True,
        # ############# ADP ############################################
        # Log duals update process
        la.LOG_DUALS: False,
        la.LOG_VALUE_UPDATE: False,
        la.LOG_COSTS: False,
        la.LOG_SOLUTIONS: False,
        la.LOG_WEIGHTS: False,
        # Log .lp and .log from MIP models
        la.LOG_MIP: False,
        # Log time spent across every step in each code section
        la.LOG_TIMES: False,
        # Save fleet, demand, and delay plots
        la.SAVE_PLOTS: False,
        # Save fleet and demand dfs for live plot
        la.SAVE_DF: False,
        # Log level saved in file
        la.LEVEL_FILE: la.DEBUG,
        # Log level printed in screen
        la.LEVEL_CONSOLE: la.INFO,
        la.FORMATTER_FILE: la.FORMATTER_TERSE,
        # Log everything
        la.LOG_ALL: False,
        # Log chosen (if LOG_ALL, set to lowest, i.e., DEBUG)
        la.LOG_LEVEL: la.INFO,
    },
):
    # Set tabu size (vehicles cannot visit nodes in tabu)
    Car.SIZE_TABU = config.car_size_tabu

    print(f'### Saving experimental settings at: "{config.exp_settings}"')
    config.save()

    # ---------------------------------------------------------------- #
    # Episodes ####################################################### #
    # ---------------------------------------------------------------- #

    amod = AmodNetworkHired(config, online=True)
    print(
        f"### Nodes with no neighbors (within time increment) "
        f"({len(amod.unreachable_ods)})"
        f" = {amod.unreachable_ods}"
        f" --- #neighbors (avg, max, min) = {amod.stats_neighbors}"
    )

    episode_log = EpisodeLog(
        amod.config.save_progress, config=config, adp=amod.adp
    )
    if plot_track:
        plot_track.set_env(amod)

    # ---------------------------------------------------------------- #
    # Plot centers and guidelines #################################### #
    # ---------------------------------------------------------------- #
    if plot_track:
        plot_track.plot_centers(
            amod.points,
            mod.env.Point.Point.levels,
            mod.env.Point.Point.levels[config.demand_center_level],
            mod.env.Point.Point.levels[config.neighborhood_level],
            show_sp_lines=PlotTrack.SHOW_SP_LINES,
            show_lines=PlotTrack.SHOW_LINES,
        )

    if config.use_class_prob:
        try:

            print(
                "### Loading first-class probabilities "
                f"from '{config.path_class_prob}'..."
            )

            prob_dict = np.load(
                config.path_class_prob, allow_pickle=True
            ).item()
            time_bin = prob_dict["time_bin"]
            start_time = prob_dict["start"]
            end_time = prob_dict["end"]
            print(f"### bin={time_bin}, start={start_time}, end={end_time}")
        except Exception as e:
            print(f"Exception: '{e}'. Cannot load class probabilities.")
    else:
        prob_dict = None

    print(f"### Loading demand scenario '{config.demand_scenario}'...")

    try:
        if config.myopic or config.policy_random or config.policy_reactive:
            print("Ignore training.")
        else:
            # Load last episode
            episode_log.load_progress()
            print("Data loaded successfully.")

    except Exception as e:
        print(f"No previous episodes were saved (Exception: '{e}').")

    # ---------------------------------------------------------------- #
    # Process demand ################################################# #
    # ---------------------------------------------------------------- #

    if config.demand_scenario == conf.SCENARIO_UNBALANCED:

        origins, destinations = episode_log.get_od_lists(amod)

        # Get demand pattern from NY city
        step_trip_count = get_trip_count_step(
            conf.TRIP_FILES[0],
            step=config.time_increment,
            multiply_for=config.demand_resize_factor,
            earliest_step=config.demand_earliest_step_min,
            max_steps=config.demand_max_steps,
        )

    # ---------------------------------------------------------------- #
    # Experiment ##################################################### #
    # ---------------------------------------------------------------- #

    # Loop all episodes, pick up trips, and learn where they are
    if config.method == ConfigNetwork.METHOD_ADP_TRAIN:
        start = episode_log.n
    else:
        start = 0

    print(f" - Iterating from {start:>4} to {config.iterations:>4}...")

    for n in range(start, config.iterations):
        config.current_iteration = n

        t_update = 0
        t_mip = 0
        t_log = 0
        t_save_plots = 0
        t_add_record = 0

        if config.demand_scenario == conf.SCENARIO_UNBALANCED:

            # Sample ods for iteration n
            step_trip_list = get_trips_random_ods(
                amod.points,
                step_trip_count,
                offset_start=amod.config.offset_repositioning_steps,
                offset_end=amod.config.offset_termination_steps,
                origins=origins,
                destinations=destinations,
                classed=classed_trips,
            )

        elif config.demand_scenario == conf.SCENARIO_NYC:

            # Load a random .csv file with trips from NYC
            if config.train:
                folder = config.folder_training_files
                list_files = conf.get_file_paths(folder)
                trips_file_path = random.choice(list_files)
                test_i = n
                # print(f"  -> Trip file - {trips_file_path}")
            else:
                folder = config.folder_testing_files
                list_files = conf.get_file_paths(folder)
                # If testing, select different trip files
                test_i = n % len(list_files)
                trips_file_path = list_files[test_i]
                # print(f"  -> Trip file test ({test_i:02}) - {trips_file_path}")

            step_trip_list, step_trip_count = tp.get_ny_demand(
                config,
                trips_file_path,
                amod.points,
                seed=n,
                prob_dict=prob_dict,
                centroid_level=amod.config.centroid_level,
                unreachable_ods=amod.unreachable_ods,
            )

            # Save random data (trip samples)
            if amod.config.save_trip_data:
                df = tp.get_df_from_sampled_trips(step_trip_list)
                df.to_csv(
                    f"{config.sampled_tripdata_path}trips_{test_i:04}.csv",
                    index=False,
                )

            if amod.config.unbound_max_cars_trip_destinations:

                # Trip destination ids are unrestricted, i.e., cars can
                # always arrive at destinations
                all_trips = list(itertools.chain(*step_trip_list))
                id_destinations = set([t.d.id for t in all_trips])
                amod.unrestricted_parking_node_ids = id_destinations
            else:
                amod.unrestricted_parking_node_ids = set()

        # Log events of iteration n
        logger = la.get_logger(
            config.log_path(amod.adp.n),
            log_file=config.log_path(amod.adp.n),
            **log_config_dict,
        )

        logger.debug(
            "##################################"
            f" Iteration {n:04} "
            f"- Demand (min={min(step_trip_count)}"
            f", max={max(step_trip_count)})"
            f", step={config.time_increment}"
            f", earliest_step={config.demand_earliest_step_min}"
            f", max_steps={config.demand_max_steps}"
            f", offset_start={amod.config.offset_repositioning_steps}"
            f", offset_end={amod.config.offset_termination_steps}"
            f", steps={amod.config.time_steps}"
        )

        if plot_track:
            plot_track.opt_episode = n

        # Start saving data of each step in the adp_network
        step_log = StepLog(amod)

        # Resetting environment
        amod.reset(seed=n)

        # Save random data (initial positions)
        if amod.config.save_fleet_data:
            # Save car distribution
            df_cars = amod.get_fleet_df()
            df_cars.to_csv(
                f"{config.fleet_data_path}cars_{test_i:04}.csv", index=False
            )

        # ------------------------------------------------------------ #
        # Plot fleet current status ################################## #
        # ------------------------------------------------------------ #
        if plot_track:

            # Computing initial timestep
            plot_track.compute_movements(0)

        start_time = time.time()

        # Outstanding trips between steps (user backlogging)
        outstanding = list()

        # Trips from this iteration (make sure it can be used again)
        it_step_trip_list = deepcopy(step_trip_list)

        # Get decisions for optimal rebalancing
        new_fleet_size = None
        if config.method == ConfigNetwork.METHOD_OPTIMAL:
            (
                it_decisions,
                it_step_trip_list,
                new_fleet_size,
            ) = optimal_rebalancing(
                amod, it_step_trip_list, log_mip=log_config_dict[la.LOG_MIP]
            )

            print(f"MPC optimal fleet size: {new_fleet_size}")

        if (
            log_config_dict[la.SAVE_PLOTS]
            or log_config_dict[la.SAVE_DF]
            or log_config_dict[la.LOG_STEP_SUMMARY]
        ):
            logger.debug("  - Computing fleet status...")
            # Compute fleet status after making decision in step - 1
            # What each car is doing when trips are arriving?
            step_log.compute_fleet_status()

        # When step=0, no users have come from previous round
        # step_log.add_record(0, [], [])

        total_trips = 0

        # Iterate through all steps and match requests to cars
        for step, trip_list in enumerate(it_step_trip_list):

            # print(exe_times, len(decision_post))
            config.current_step = step
            # Add trips from last step (when user backlogging is enabled)
            trips = trip_list + outstanding
            outstanding = []

            if plot_track:
                # Update optimization time step
                plot_track.opt_step = step

                # Create trip dictionary of coordinates
                plot_track.trips_dict[step] = vi.compute_trips(trips)

            # ######################################################## #
            # TIME INCREMENT HAS PASSED ############################## #
            # ######################################################## #

            if amod.config.fav_fleet_size > 0:
                hired_cars = amod.step_favs.get(step, [])

                # Add hired fleet to model
                amod.hired_cars.extend(hired_cars)
                amod.available_hired.extend(hired_cars)
                amod.overall_hired.extend(hired_cars)

            # Loop cars and update their current status as well as the
            # the list of available vehicles (change available and
            # available_hired)
            t1 = time.time()
            # If policy is reactive, rebalancing cars can be rerouted
            # from the intermediate nodes along the shortest path
            # to the rebalancing target. Notice that, if level > 0,
            # miiddle points will correspond to corresponding hierarchi-
            # cal superior node.
            amod.update_fleet_status(
                step + 1, use_rebalancing_cars=amod.config.policy_reactive
            )
            t_update += time.time() - t1

            # Show the top highest vehicle count per position
            # amod.show_count_vehicles_top(step, 5)

            t1 = time.time()
            logger.debug("\n## Car attributes:")

            # Log both fleets
            for c in itertools.chain(amod.cars, amod.hired_cars):
                logger.debug(f"{c} - {c.attribute}")

            # What each vehicle is doing after update?
            la.log_fleet_activity(
                config.log_path(amod.adp.n),
                step + 1,
                skip_steps,
                step_log,
                filter_status=[],
                msg="post update",
            )
            t_log += time.time() - t1

            t1 = time.time()

            if config.policy_optimal:
                # print(
                #     f"it={step:04} - Playing decisions {len(it_decisions[step])}"
                # )
                revenue, serviced, rejected = play_decisions(
                    amod, trips, step + 1, it_decisions[step]
                )

            # ######################################################## #
            # METHOD - MPC - HORIZON ################################# #
            # ######################################################## #
            elif config.policy_mpc:

                # Predicted trips for next steps (exclusive)
                predicted_trips = it_step_trip_list[
                    step + 1 : step + amod.config.mpc_forecasting_horizon
                ]

                # Log events of iteration n
                logger = la.get_logger(
                    config.log_path(step + 1),
                    log_file=config.log_path(step + 1),
                    **log_config_dict,
                )

                # Trips within the same region are invalid
                decisions = mpc(
                    # Amod environment with configuration file
                    amod,
                    # Trips to be matched
                    trips,
                    # Predicted trips within the forecasting horizon
                    predicted_trips,
                    # Service step (+1 trip placement step)
                    step=step + 1,
                    log_mip=log_config_dict[la.LOG_MIP],
                )

                revenue, serviced, rejected = play_decisions(
                    amod, trips, step + 1, decisions
                )

            else:
                revenue, serviced, rejected = service_trips(
                    # Amod environment with configuration file
                    amod,
                    # Trips to be matched
                    trips,
                    # Service step (+1 trip placement step)
                    step + 1,
                    # Save mip .lp and .log of iteration n
                    iteration=n,
                    log_mip=log_config_dict[la.LOG_MIP],
                    log_times=log_config_dict[la.LOG_TIMES],
                    car_type_hide=Car.TYPE_FLEET,
                )

            if amod.config.separate_fleets:

                # Optimize
                revenue_fav, serviced_fav, rejected_fav = service_trips(
                    # Amod environment with configuration file
                    amod,
                    # Trips to be matched
                    rejected,
                    # Service step (+1 trip placement step)
                    step + 1,
                    # Save mip .lp and .log of iteration n
                    iteration=n,
                    car_type_hide=Car.TYPE_FLEET,
                    log_times=log_config_dict[la.LOG_TIMES],
                    log_mip=log_config_dict[la.LOG_MIP],
                )

                revenue += (revenue_fav,)
                serviced += (serviced_fav,)
                rejected = rejected_fav

            # ######################################################## #
            # METHOD - BACKLOG ####################################### #
            # ######################################################## #

            if amod.config.max_user_backlogging_delay > 0:

                expired = []
                for r in rejected:

                    # Add time increment to backlog delay
                    r.backlog_delay += amod.config.time_increment
                    r.times_backlogged += 1

                    # Max. backlog reached -> discard trip
                    if (
                        r.backlog_delay
                        > amod.config.max_user_backlogging_delay
                        or step + 1 == amod.config.time_steps
                    ):
                        expired.append(r)
                    else:
                        outstanding.append(r)

                rejected = expired
            t_mip += time.time() - t1

            # ######################################################## #
            # METHOD - REACTIVE REBALANCE ############################ #
            # ######################################################## #

            t_reactive_rebalance_1 = time.time()
            if amod.config.policy_reactive and (rejected or outstanding):
                # If reactive rebalance, send vehicles to rejected
                # user's origins
                logger.debug(
                    "####################"
                    f"[{n:04}]-[{step:04}] REACTIVE REBALANCE "
                    "####################"
                )
                logger.debug("Rejected requests (rebalancing targets):")
                for r in rejected:
                    logger.debug(f"{r}")

                # print(step, amod.available_fleet_size, len(rejected))
                # Update fleet headings to isolate Idle vehicles.
                # Only empty cars are considered for rebalancing.
                t1 = time.time()
                amod.update_fleet_status(step + 1)
                t_update += time.time() - t1

                t1 = time.time()
                # What each vehicle is doing?
                la.log_fleet_activity(
                    config.log_path(amod.adp.n),
                    step,
                    skip_steps,
                    step_log,
                    filter_status=[],
                    msg="before rebalancing",
                )
                t_log += time.time() - t1

                # Service idle vehicles
                rebal_costs, _, _ = service_trips(
                    # Amod environment with configuration file
                    amod,
                    # Trips to be matched
                    rejected + outstanding,
                    # Service step (+1 trip placement step)
                    step + 1,
                    # # Save mip .lp and .log of iteration n
                    iteration=n,
                    log_mip=log_config_dict[la.LOG_MIP],
                    log_times=log_config_dict[la.LOG_TIMES],
                    car_type_hide=Car.TYPE_FLEET,
                    reactive=True,
                )

                revenue -= rebal_costs
                logger.debug(f"\n# REB. COSTS: {rebal_costs:6.2f}")

            t_reactive_rebalance = time.time() - t_reactive_rebalance_1

            t1 = time.time()
            if (
                log_config_dict[la.SAVE_PLOTS]
                or log_config_dict[la.SAVE_DF]
                or log_config_dict[la.LOG_STEP_SUMMARY]
            ):
                logger.debug("  - Computing fleet status...")
                # Compute fleet status after making decision in step - 1
                # What each car is doing when trips are arriving?
                step_log.compute_fleet_status()
            t_save_plots += time.time() - t1

            t1 = time.time()
            # -------------------------------------------------------- #
            # Update log with iteration ############################## #
            # -------------------------------------------------------- #
            step_log.add_record(
                revenue, serviced, rejected, outstanding, trips=trips
            )
            t_add_record += time.time() - t1

            # -------------------------------------------------------- #
            # Plotting fleet activity ################################ #
            # -------------------------------------------------------- #

            t1 = time.time()
            # What each vehicle is doing?
            la.log_fleet_activity(
                config.log_path(amod.adp.n),
                step,
                skip_steps,
                step_log,
                filter_status=[],
                msg="after decision",
            )
            t_log += time.time() - t1

            if plot_track:

                logger.debug("Computing movements...")
                plot_track.compute_movements(step + 1)
                logger.debug("Finished computing...")

                time.sleep(step_delay)

            # print(step, "weighted value:", amod.adp.get_weighted_value.cache_info())
            # print(step, "preview decision:", amod.preview_decision.cache_info())
            # print(step, "preview decision:", amod.preview_move.cache_info())
            # amod.adp.get_weighted_value.cache_clear()
            # self.post_cost.cache_clear()

        # LAST UPDATE (Closing the episode)
        t1 = time.time()
        amod.update_fleet_status(step + 1)
        t_update += time.time() - t1

        # Save random data (fleet and trips)
        if amod.config.save_trip_data:

            df = tp.get_df_from_sampled_trips(
                it_step_trip_list,
                show_service_data=True,
                earliest_datetime=config.demand_earliest_datetime,
            )

            df.to_csv(
                f"{config.sampled_tripdata_path}trips_{test_i:04}_result.csv",
                index=False,
            )

        if amod.config.save_fleet_data:

            df_cars = amod.get_fleet_df()
            df_cars.to_csv(
                f"{config.fleet_data_path}cars_{test_i:04}_result.csv",
                index=False,
            )

        # -------------------------------------------------------------#
        # Compute episode info #########################################
        # -------------------------------------------------------------#

        logger.debug("  - Computing iteration...")

        t1 = time.time()
        episode_log.compute_episode(
            step_log,
            it_step_trip_list,
            time.time() - start_time,
            fleet_size=new_fleet_size,
            save_df=log_config_dict[la.SAVE_DF],
            plots=log_config_dict[la.SAVE_PLOTS],
            save_learning=amod.config.save_progress,
            save_overall_stats=save_overall_stats,
        )
        t_epi = t1 = time.time() - t1

        # Clean weight track
        amod.adp.reset_weight_track()

        logger.info(
            f"####### "
            f"[Episode {n+1:>5}] "
            f"- {episode_log.last_episode_stats()} "
            f"serviced={step_log.serviced}, "
            f"rejected={step_log.rejected}, "
            f"total={step_log.total} "
            f"t(episode={t_epi:.2f}, "
            f"t_log={t_log:.2f}, "
            f"t_mip={t_mip:.2f}, "
            f"t_save_plots={t_save_plots:.2f}, "
            f"t_up={t_update:.2f}, "
            f"t_add_record={t_add_record:.2f})"
            f"#######"
        )

        # If True, saves time details in file times.csv
        # if log_config_dict[la.LOG_TIMES]:

        # logger.debug("weighted values:", len(amod.adp.weighted_values))
        # logger.debug("get_state:", amod.adp.get_state.cache_info())
        # logger.debug(
        #     "preview_decision:", amod.preview_decision.cache_info()
        # )
        # logger.debug(f"Rebalance: {amod.get_zone_neighbors.cache_info()}")
        # logger.debug("post_cost:", amod.post_cost.cache_info())

        # Increasingly let cars to be idle
        if config.idle_annealing is not None:
            # By the end of all iterations, cars cannot be forced to
            # rebalance anymore
            config.config[ConfigNetwork.IDLE_ANNEALING] += 1  # 1/episodes

        # pprint({k:v for k, v in amod.beta_ab.items() if v["a"]>2})
    # Plot overall performance (reward, service rate, and weights)
    episode_log.compute_learning()

    return amod.adp.reward


if __name__ == "__main__":

    instance_name = None
    # instance_name = f"{conf.FOLDER_OUTPUT}BA_LIN_C1_V=0400-0000(R)_I=5_L[3]=(01-0-, 02-0-, 03-0-)_R=([1-6][L(05)]_T=[06h,+30m+06h+30m]_0.10(S)_1.00_0.10_A_2.40_10.00_0.00_0.00_0.00_B_2.40_10.00_0.00_0.00_1.00/exp_settings.json"
    if instance_name:

        log_mip = False
        log_times = False
        save_plots = False
        save_df = False
        log_all = False
        log_level = la.INFO
        print(f'Loading settings from "{instance_name}"')
        start_config = ConfigNetwork.load(instance_name)
    else:
        # Default is training
        method = ConfigNetwork.METHOD_ADP_TEST
        save_progress_interval = None

        # Isolate arguments (first is filename)
        args = sys.argv[1:]

        try:
            test_label = args[0]
        except:
            test_label = "TESTDE"

        try:
            iterations = args.index("-n")
            n_iterations = int(args[iterations + 1])
        except:
            n_iterations = 400

        try:
            ibacklog = args.index("-backlog")
            backlog_delay_min = int(args[ibacklog + 1])
        except:
            backlog_delay_min = 0

        # Enable logs
        log_all = "-log_all" in args
        log_mip = "-log_mip" in args
        log_times = "-log_times" in args
        log_fleet = "-log_fleet" in args
        log_trips = "-log_trips" in args
        log_summary = "-log_summary" in args

        # Save supply and demand plots and dataframes
        save_plots = "-save_plots" in args
        save_df = "-save_df" in args

        # Optimization methods
        myopic = "-myopic" in args
        policy_random = "-random" in args
        policy_reactive = "-reactive" in args
        train = "-train" in args
        test = "-test" in args
        optimal = "-optimal" in args
        policy_mpc = "-mpc" in args
        mpc_horizon = 5

        # Progress file will be updated every X iterations
        save_progress_interval = None

        if policy_random:
            method = ConfigNetwork.METHOD_RANDOM

        elif policy_reactive:
            method = ConfigNetwork.METHOD_REACTIVE

        elif myopic:
            method = ConfigNetwork.METHOD_MYOPIC

        elif train:
            method = ConfigNetwork.METHOD_ADP_TRAIN
            try:
                i = int(args.index("-train"))
                save_progress_interval = int(args[i + 1])

            except:
                save_progress_interval = 1

        elif test:
            method = ConfigNetwork.METHOD_ADP_TEST

        elif optimal:
            method = ConfigNetwork.METHOD_OPTIMAL
            n_iterations = 1

        elif policy_mpc:
            method = ConfigNetwork.METHOD_MPC
            try:
                i = int(args.index("-mpc"))
                mpc_horizon = int(args[i + 1])
            except:
                mpc_horizon = 5
            n_iterations = 1

        else:
            raise ("Error! Which method?")

        print(
            f"### method={method}, save progress={(save_progress_interval if  save_progress_interval else 'no')}"
        )

        try:
            fleet_size_i = args.index(f"-{ConfigNetwork.FLEET_SIZE}")
            fleet_size = int(args[fleet_size_i + 1])
        except:
            fleet_size = 100

        try:
            fav_fleet_size_i = args.index(f"-{ConfigNetwork.FAV_FLEET_SIZE}")
            fav_fleet_size = int(args[fav_fleet_size_i + 1])
        except:
            fav_fleet_size = 0

        try:
            log_level_i = args.index("-level")
            log_level_label = args[log_level_i + 1]
            log_level = la.levels[log_level_label]
        except:
            log_level = la.INFO

        print(f"###### STARTING EXPERIMENTS. METHOD: {method}")

        start_config = get_sim_config(
            {
                # ConfigNetwork.CASE_STUDY: "N08Z08SD02",
                ConfigNetwork.CASE_STUDY: "N08Z08CD02",
                # Cars can rebalance/stay/travel to trip destinations
                # indiscriminately
                ConfigNetwork.UNBOUND_MAX_CARS_TRIP_DESTINATIONS: False,
                # All trip decisions can be realized
                ConfigNetwork.UNBOUND_MAX_CARS_TRIP_DECISIONS: True,
                ConfigNetwork.PATH_CLASS_PROB: "distr/class_prob_distribution_p5min_6h.npy",
                ConfigNetwork.ITERATIONS: n_iterations,
                ConfigNetwork.TEST_LABEL: test_label,
                ConfigNetwork.DISCOUNT_FACTOR: 1,
                ConfigNetwork.FLEET_SIZE: fleet_size,
                # DEMAND ############################################# #
                ConfigNetwork.UNIVERSAL_SERVICE: False,
                ConfigNetwork.DEMAND_RESIZE_FACTOR: 0.1,
                ConfigNetwork.DEMAND_TOTAL_HOURS: 6,
                ConfigNetwork.DEMAND_EARLIEST_HOUR: 6,
                ConfigNetwork.OFFSET_TERMINATION_MIN: 30,
                ConfigNetwork.OFFSET_REPOSITIONING_MIN: 30,
                ConfigNetwork.TIME_INCREMENT: 5,
                ConfigNetwork.DEMAND_SAMPLING: True,
                # Service quality
                ConfigNetwork.MATCHING_DELAY: 15,
                ConfigNetwork.MAX_USER_BACKLOGGING_DELAY: backlog_delay_min,
                ConfigNetwork.SQ_GUARANTEE: False,
                ConfigNetwork.RECHARGE_COST_DISTANCE: 0.1,
                ConfigNetwork.APPLY_BACKLOG_REJECTION_PENALTY: True,
                ConfigNetwork.TRIP_REJECTION_PENALTY: (("A", 2.5), ("B", 2.5)),
                ConfigNetwork.TRIP_OUTSTANDING_PENALTY: (
                    ("A", 0.25),
                    ("B", 0.25),
                ),
                ConfigNetwork.TRIP_BASE_FARE: (("A", 7.5), ("B", 2.5)),
                ConfigNetwork.TRIP_DISTANCE_RATE_KM: (("A", 1), ("B", 1)),
                ConfigNetwork.TRIP_TOLERANCE_DELAY_MIN: (("A", 0), ("B", 0)),
                ConfigNetwork.TRIP_MAX_PICKUP_DELAY: (("A", 15), ("B", 15)),
                ConfigNetwork.TRIP_CLASS_PROPORTION: (("A", 0), ("B", 1)),
                # ADP EXECUTION ###################################### #
                ConfigNetwork.METHOD: method,
                ConfigNetwork.SAVE_PROGRESS: save_progress_interval,
                ConfigNetwork.ADP_IGNORE_ZEROS: True,
                ConfigNetwork.LINEARIZE_INTEGER_MODEL: False,
                ConfigNetwork.USE_ARTIFICIAL_DUALS: False,
                # MPC ################################################ #
                ConfigNetwork.MPC_FORECASTING_HORIZON: mpc_horizon,
                ConfigNetwork.MPC_USE_PERFORMANCE_TO_GO: False,
                ConfigNetwork.MPC_REBALANCE_TO_NEIGHBORS: False,
                ConfigNetwork.MPC_USE_TRIP_ODS_ONLY: True,
                # EXPLORATION ######################################## #
                # ANNEALING + THOMPSON
                # If zero, cars increasingly gain the right of stay
                # still. This obliges them to rebalance consistently.
                # If None, disabled.
                ConfigNetwork.IDLE_ANNEALING: None,
                ConfigNetwork.ACTIVATE_THOMPSON: False,
                ConfigNetwork.MAX_TARGETS: 12,
                # When cars start in the last visited point, the model takes
                # a long time to figure out the best time
                ConfigNetwork.FLEET_START: conf.FLEET_START_RANDOM,
                # ConfigNetwork.FLEET_START: conf.FLEET_START_REJECTED_TRIP_ORIGINS,
                # ConfigNetwork.FLEET_START: conf.FLEET_START_PARKING_LOTS,
                # ConfigNetwork.LEVEL_PARKING_LOTS: 2,
                # ConfigNetwork.FLEET_START: conf.FLEET_START_LAST_TRIP_ORIGINS,
                ConfigNetwork.CAR_SIZE_TABU: 0,
                # REBALANCING ##########################################
                # If REACHABLE_NEIGHBORS is True, then PENALIZE_REBALANCE
                # is False
                ConfigNetwork.PENALIZE_REBALANCE: False,
                # All rebalancing finishes within time increment
                ConfigNetwork.REBALANCING_TIME_RANGE_MIN: (0, 10),
                # Consider only rebalance targets from sublevel
                ConfigNetwork.REBALANCE_SUB_LEVEL: None,
                # Rebalance to at most max targets
                ConfigNetwork.REBALANCE_MAX_TARGETS: None,
                # Remove nodes that dont have at least min. neighbors
                ConfigNetwork.MIN_NEIGHBORS: 1,
                ConfigNetwork.REACHABLE_NEIGHBORS: False,
                ConfigNetwork.N_CLOSEST_NEIGHBORS: ((1, 6), (2, 6), (3, 6),),
                ConfigNetwork.CENTROID_LEVEL: 1,
                # FLEET ############################################## #
                # Car operation
                ConfigNetwork.MAX_CARS_LINK: 5,
                ConfigNetwork.MAX_IDLE_STEP_COUNT: None,
                # FAV configuration
                # Functions
                ConfigNetwork.DEPOT_SHARE: 0.01,
                ConfigNetwork.FAV_DEPOT_LEVEL: None,
                ConfigNetwork.FAV_FLEET_SIZE: fav_fleet_size,
                ConfigNetwork.SEPARATE_FLEETS: False,
                ConfigNetwork.MAX_CONTRACT_DURATION: True,
                # mean, std, clip_a, clip_b
                # ConfigNetwork.FAV_EARLIEST_FEATURES = (8, 1, 5, 9)
                # ConfigNetwork.FAV_AVAILABILITY_FEATURES = (2, 1, 1, 4)
                # ConfigNetwork.PARKING_RATE_MIN = 1.50/60 # 1.50/h
                # ConfigNetwork.PARKING_RATE_MIN = 0.1*20/60
                # ,  # = rebalancing 1 min
                ConfigNetwork.PARKING_RATE_MIN: 0,  # = rebalancing 1 min
                # Saving
                ConfigNetwork.USE_SHORT_PATH: False,
                ConfigNetwork.SAVE_TRIP_DATA: log_trips,
                ConfigNetwork.SAVE_FLEET_DATA: log_fleet,
                # Load 1st class probabilities dictionary
                ConfigNetwork.USE_CLASS_PROB: True,
                ConfigNetwork.ENABLE_RECHARGING: False,
                # PLOT ############################################### #
                ConfigNetwork.PLOT_FLEET_XTICKS_LABELS: [
                    "",
                    "6AM",
                    "",
                    "7AM",
                    "",
                    "8AM",
                    "",
                    "9AM",
                    "",
                    "10AM",
                    "",
                    "11AM",
                    "",
                    "12AM",
                    "",
                ],
                ConfigNetwork.PLOT_FLEET_X_MIN: 0,
                ConfigNetwork.PLOT_FLEET_X_MAX: 84,
                ConfigNetwork.PLOT_FLEET_X_NUM: 15,
                ConfigNetwork.PLOT_FLEET_OMIT_CRUISING: False,
                ConfigNetwork.PLOT_DEMAND_Y_MAX: 3500,
                ConfigNetwork.PLOT_DEMAND_Y_NUM: 8,
                ConfigNetwork.PLOT_DEMAND_Y_MIN: 0,
            }
        )

    # Toggle what is going to be logged
    log_config = {
        # Write each vehicles status
        la.LOG_FLEET_ACTIVITY: False,
        # Write profit, service level, # trips, car/satus count
        la.LOG_STEP_SUMMARY: log_summary,
        # ############# ADP ############################################
        # Log duals update process
        la.LOG_WEIGHTS: False,
        la.LOG_VALUE_UPDATE: False,
        la.LOG_DUALS: False,
        la.LOG_COSTS: False,
        la.LOG_SOLUTIONS: False,
        la.LOG_ATTRIBUTE_CARS: False,
        la.LOG_DECISION_INFO: False,
        # Log .lp and .log from MIP models
        la.LOG_MIP: log_mip,
        # Log time spent across every step in each code section
        la.LOG_TIMES: log_times,
        # Save fleet, demand, and delay plots
        la.SAVE_PLOTS: save_plots,
        # Save fleet and demand dfs for live plot
        la.SAVE_DF: save_df,
        # Log level saved in file
        la.LEVEL_FILE: la.DEBUG,
        # Log level printed in screen
        la.LEVEL_CONSOLE: la.INFO,
        la.FORMATTER_FILE: la.FORMATTER_TERSE,
        # Log everything
        la.LOG_ALL: log_all,
        # Log chosen (if LOG_ALL, set to lowest, i.e., DEBUG)
        la.LOG_LEVEL: (log_level if not log_all else la.DEBUG),
    }

    alg_adp(None, start_config, log_config_dict=log_config)
