import os
import sys
from copy import deepcopy
import time
import random
from collections import defaultdict
from pprint import pprint
import numpy as np
import itertools

# Adding project folder to import modules
root = os.getcwd().replace("\\", "/")
sys.path.append(root)

from mod.env.matching import service_trips, optimal_rebalancing, play_decisions
import mod.util.log_util as la

from mod.env.amod.AmodNetworkHired import AmodNetworkHired
from mod.env.visual import StepLog, EpisodeLog
import mod.env.adp.adp as adp

import mod.env.visual as vi
from mod.env.config import ConfigNetwork

import mod.env.config as conf


from mod.env.car import Car
from mod.env.trip import get_trip_count_step, get_trips_random_ods
import mod.env.trip as tp
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
            ConfigNetwork.PENALIZE_REBALANCE: True,
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
                    temporal=1,
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
                    temporal=1,
                    spatial=3,
                    battery=adp.DISAGGREGATE,
                    contract=adp.DISCARD,
                    car_type=adp.DISAGGREGATE,
                    car_origin=adp.DISCARD,
                ),
            ],
            # ConfigNetwork.LEVEL_TIME_LIST: [0.5, 1, 2, 3],
            ConfigNetwork.LEVEL_TIME_LIST: [1, 5, 7, 10],
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
    log_config_dict=None,
):
    # Set tabu size (vehicles cannot visit nodes in tabu)
    Car.SIZE_TABU = config.car_size_tabu

    print(f'Saving experimental settings at: "{config.exp_settings}"')
    config.save()

    # ---------------------------------------------------------------- #
    # Episodes ####################################################### #
    # ---------------------------------------------------------------- #

    amod = AmodNetworkHired(config, online=True)
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
            nw.Point.levels,
            nw.Point.levels[config.demand_center_level],
            nw.Point.levels[config.neighborhood_level],
            show_sp_lines=PlotTrack.SHOW_SP_LINES,
            show_lines=PlotTrack.SHOW_LINES,
        )

    if config.use_class_prob:
        print("Loading first-class probabilities...")
        prob_dict = np.load(conf.FIST_CLASS_PROB, allow_pickle=True).item()
        time_bin = prob_dict["time_bin"]
        start_time = prob_dict["start"]
        end_time = prob_dict["end"]
        print(f"bin={time_bin}, start={start_time}, end={end_time}")
    else:
        prob_dict = None

    print(f"Loading demand scenario '{config.demand_scenario}'...")

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
                trips_file_path = random.choice(conf.PATHS_TRAINING_TRIPS)
                test_i = n
                # print(f"  -> Trip file - {trips_file_path}")
            else:
                # If testing, select different trip files
                test_i = n % len(conf.PATHS_TESTING_TRIPS)
                trips_file_path = conf.PATHS_TESTING_TRIPS[test_i]
                # print(f"  -> Trip file test ({test_i:02}) - {trips_file_path}")

            step_trip_list, step_trip_count = tp.get_ny_demand(
                config,
                trips_file_path,
                amod.points,
                seed=n,
                prob_dict=prob_dict,
                centroid_level=amod.config.centroid_level,
            )

            # Save random data (trip samples)
            if amod.config.save_trip_data:
                df = tp.get_df(step_trip_list)
                df.to_csv(
                    f"{config.sampled_tripdata_path}trips_{test_i:04}.csv",
                    index=False,
                )

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
        if config.method == ConfigNetwork.METHOD_OPTIMAL:
            it_decisions, it_step_trip_list = optimal_rebalancing(
                amod, it_step_trip_list
            )

        # Iterate through all steps and match requests to cars
        for step, trips in enumerate(it_step_trip_list):
            config.current_step = step

            # Add trips from last step (when user backlogging is enabled)
            trips.extend(outstanding)
            outstanding = []

            logger.debug(
                "###########################################"
                "###########################################"
                "\n###########################################"
                f" (step={step+1}, trips={len(trips)}) "
                "###########################################"
                "\n###########################################"
                "###########################################"
            )

            t1 = time.time()
            for t in trips:
                logger.debug(f"  - {t}")
            t_log += time.time() - t1

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
            # to the rebalancing target.
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

            # print(
            #     f"#{step:>3} - hired={len(amod.hired_cars)}"
            #     f" - expired={len(amod.expired_contract_cars)}"
            #     f" - available={len(amod.available_hired)}"
            #     f" - favs={len(amod.step_favs.get(step, [])):>2}"
            #     f" - trips={len(trips):>2}")

            # count_car_point = defaultdict(int)
            # for c in amod.cars:
            #     count_car_point[c.point.id] += 1

            # for p, count in count_car_point.items():
            #     if count > 5:
            #         print(f"{step} - link {p} has {count} cars")
            # print(len(amod.cars), len(amod.available), len(amod.hired_cars), len(amod.available_hired))
            # Optimize

            # for tt in trips:
            #     print(tt.info())

            t1 = time.time()
            if config.method == ConfigNetwork.METHOD_OPTIMAL:
                print(
                    f"it={step:04} - Playing decisions {len(it_decisions[step])}"
                )
                revenue, serviced, rejected = play_decisions(
                    amod, trips, step + 1, it_decisions[step]
                )
                print(revenue, len(serviced), len(rejected))

            else:
                revenue, serviced, rejected = service_trips(
                    # Amod environment with configuration file
                    amod,
                    # Trips to be matched
                    trips,
                    # Service step (+1 trip placement step)
                    step + 1,
                    # # Save mip .lp and .log of iteration n
                    iteration=n,
                    log_mip=log_config_dict[la.LOG_MIP],
                    log_times=log_config_dict[la.LOG_TIMES],
                    car_type_hide=Car.TYPE_FLEET,
                )

            if amod.config.separate_fleets:
                # print(f"Trips: {len(trips)} - Revenue: {revenue} - Serviced: {len(serviced)} - Rejected: {len(rejected)}")

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

            # print(f"Revenue FAV: {revenue_fav} - Serviced FAV: {len(serviced_fav)} - Rejected FAV: {len(rejected_fav)}")

            if amod.config.allow_user_backlogging:

                expired = []
                for r in rejected:
                    max_delay = r.update_delay(amod.config.time_increment)
                    if max_delay <= 0:
                        expired.append(r)
                    else:
                        outstanding.append(r)

                rejected = expired
            t_mip += time.time() - t1

            # If reactive rebalance, send vehicles to rejected
            # user's origins
            t_reactive_rebalance_1 = time.time()
            if amod.config.policy_reactive and rejected:
                logger.debug(
                    "####################"
                    f"[{n:04}]-[{step:04}] REACTIVE REBALANCE "
                    "####################"
                )
                logger.debug("Rejected requests (rebalancing targets):")
                for r in rejected:
                    logger.debug(f"{r}")

                # print(step, amod.available_fleet_size,  len(rejected))
                # Update fleet headings to isolate Idle vehicles
                amod.update_fleet_status(step + 1)

                # Service idle vehicles
                rebal_costs, _, _ = service_trips(
                    # Amod environment with configuration file
                    amod,
                    # Trips to be matched
                    rejected,
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
            # What each vehicle is doing?
            la.log_fleet_activity(
                config.log_path(amod.adp.n),
                int((step + 1) / config.time_max_cars_link),
                skip_steps,
                step_log,
                filter_status=[],
                msg="after decision",
            )
            t_log += time.time() - t1

            t1 = time.time()
            if log_config_dict[la.SAVE_PLOTS] or log_config_dict[la.SAVE_DF]:
                logger.debug("  - Computing fleet status...")
                # Compute fleet status after making decision in step - 1
                # What each car is doing when trips are arriving?
                step_log.compute_fleet_status(step + 1)
            t_save_plots += time.time() - t1

            t1 = time.time()
            # -------------------------------------------------------- #
            # Update log with iteration ############################## #
            # -------------------------------------------------------- #
            step_log.add_record(revenue, serviced, rejected)
            t_add_record += time.time() - t1

            # -------------------------------------------------------- #
            # Plotting fleet activity ################################ #
            # -------------------------------------------------------- #

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

            amod.print_fleet_stats()  # filter_status=[Car.ASSIGN])

        # LAST UPDATE (Closing the episode)
        amod.update_fleet_status(step + 1)

        # Save random data (fleet and trips)
        if amod.config.save_trip_data:

            df = tp.get_df(
                it_step_trip_list,
                show_service_data=True,
                earliest_datetime=config.demand_earliest_datetime,
            )

            df.to_csv(
                f"{config.sampled_tripdata_path}trips_result_{test_i:04}.csv",
                index=False,
            )

        if amod.config.save_fleet_data:

            df_cars = amod.get_fleet_df()
            df_cars.to_csv(
                f"{config.fleet_data_path}cars_result_{test_i:04}.csv",
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
        if log_config_dict[la.LOG_TIMES]:
            print("weighted values:", len(amod.adp.weighted_values))
            # print("get_state:", amod.adp.get_state.cache_info())
            # print("preview_decision:", amod.preview_decision.cache_info())
            # print("post_cost:", amod.post_cost.cache_info())

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

    backlog = "-backlog" in args

    # Enable logs
    log_adp = "-log_adp" in args
    log_mip = "-log_mip" in args
    log_times = "-log_times" in args
    log_fleet = "-log_fleet" in args
    log_trips = "-log_trips" in args

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

    if policy_random:
        print("RANDOM")
        method = ConfigNetwork.METHOD_RANDOM

    elif policy_reactive:
        print("REACTIVE")
        method = ConfigNetwork.METHOD_REACTIVE

    elif myopic:
        print("MYOPIC")
        method = ConfigNetwork.METHOD_MYOPIC

    elif train:
        print("SAVING PROGRESS")
        try:
            i = int(args.index("-train"))
            save_progress_interval = int(args[i + 1])
        except:
            save_progress_interval = 1
        print(f"Saving progress every {save_progress_interval} iteration.")
        method = ConfigNetwork.METHOD_ADP_TRAIN

    elif test:
        method = ConfigNetwork.METHOD_ADP_TEST
        save_progress_interval = None
        print("Progress will not be saved!")

    elif optimal:
        method = ConfigNetwork.METHOD_OPTIMAL
        n_iterations = 1

    else:
        raise ("Error! Which method?")

    print("METHOD:", method)

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

    instance_name = None  # f"{conf.FOLDER_OUTPUT}hiring_LIN_cars=0000-0300(L)_t=1_levels[5]=(1-0, 1-0, 1-0, 3-300, 3-600)_rebal=([0-8, 1-8][tabu=10])[L(05)][P]_[05h,+30m+04h+60m]_0.10(S)_1.00_0.10/exp_settings.json"
    if instance_name:
        print(f'Loading settings from "{instance_name}"')
        start_config = ConfigNetwork.load(instance_name)
    else:
        start_config = get_sim_config(
            {
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
                ConfigNetwork.TIME_INCREMENT: 10,
                ConfigNetwork.DEMAND_SAMPLING: True,
                # Service quality
                ConfigNetwork.MATCHING_DELAY: 15,
                ConfigNetwork.ALLOW_USER_BACKLOGGING: backlog,
                ConfigNetwork.SQ_GUARANTEE: False,
                # ConfigNetwork.TRIP_REJECTION_PENALTY: {
                #     "A": 4.8,
                #     "B": 2.4,
                # },
                ConfigNetwork.TRIP_REJECTION_PENALTY: (("A", 4.8), ("B", 0)),
                ConfigNetwork.TRIP_BASE_FARE: (("A", 4.8), ("B", 2.4)),
                ConfigNetwork.TRIP_DISTANCE_RATE_KM: (("A", 1), ("B", 1)),
                ConfigNetwork.TRIP_TOLERANCE_DELAY_MIN: (("A", 5), ("B", 0)),
                ConfigNetwork.TRIP_MAX_PICKUP_DELAY: (("A", 5), ("B", 15)),
                ConfigNetwork.TRIP_CLASS_PROPORTION: (("A", 0), ("B", 1)),
                # ADP EXECUTION ###################################### #
                ConfigNetwork.METHOD: method,
                ConfigNetwork.SAVE_PROGRESS: save_progress_interval,
                ConfigNetwork.ADP_IGNORE_ZEROS: True,
                ConfigNetwork.LINEARIZE_INTEGER_MODEL: False,
                ConfigNetwork.USE_ARTIFICIAL_DUALS: False,
                # EXPLORATION ######################################## #
                # ANNEALING + THOMPSON
                # If zero, cars increasingly gain the right of stay
                # still. This obliges them to rebalance consistently.
                # If None, disabled.
                ConfigNetwork.IDLE_ANNEALING: None,
                ConfigNetwork.ACTIVATE_THOMPSON: False,
                ConfigNetwork.MAX_TARGETS: 6,
                # When cars start in the last visited point, the model takes
                # a long time to figure out the best time
                ConfigNetwork.FLEET_START: conf.FLEET_START_RANDOM,
                ConfigNetwork.CAR_SIZE_TABU: 0,
                # If REACHABLE_NEIGHBORS is True, then PENALIZE_REBALANCE
                # is False
                ConfigNetwork.PENALIZE_REBALANCE: True,
                ConfigNetwork.REACHABLE_NEIGHBORS: False,
                ConfigNetwork.N_CLOSEST_NEIGHBORS: (
                    (1, 6),
                    # (2, 6),
                    # (3, 3),
                    # (3, 4),
                ),
                ConfigNetwork.CENTROID_LEVEL: 1,
                # FLEET ############################################## #
                # Car operation
                ConfigNetwork.MAX_CARS_LINK: None,
                ConfigNetwork.MAX_IDLE_STEP_COUNT: None,
                ConfigNetwork.TIME_MAX_CARS_LINK: 5,
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
                ConfigNetwork.USE_CLASS_PROB: False,
                ConfigNetwork.ENABLE_RECHARGING: False,
            }
        )

    # Toggle what is going to be logged
    log_config = {
        la.LOG_DUALS: True,
        la.LOG_FLEET_ACTIVITY: True,
        la.LOG_VALUE_UPDATE: False,
        la.LOG_COSTS: True,
        la.LOG_SOLUTIONS: False,
        la.LOG_WEIGHTS: False,
        la.LOG_MIP: log_mip,
        la.LOG_TIMES: log_times,
        la.SAVE_PLOTS: save_plots,
        la.SAVE_DF: save_df,
        la.LOG_ALL: log_adp,
        la.LOG_LEVEL: (log_level if not log_adp else la.DEBUG),
        la.LEVEL_FILE: la.DEBUG,
        la.LEVEL_CONSOLE: la.INFO,
        la.FORMATTER_FILE: la.FORMATTER_TERSE,
    }

    print("PROGRESS", start_config.save_progress)

    alg_adp(None, start_config, log_config_dict=log_config)
