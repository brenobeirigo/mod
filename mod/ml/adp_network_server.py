import os
import sys
from copy import deepcopy
import time
import random
from collections import defaultdict
from pprint import pprint
import numpy as np
import json
# Adding project folder to import modules
root = os.getcwd().replace("\\", "/")
sys.path.append(root)

from mod.env.matching import service_trips
import mod.util.log_util as la

from mod.env.amod.AmodNetworkHired import AmodNetworkHired
from mod.env.visual import StepLog, EpisodeLog
import mod.env.adp.adp as adp

import mod.env.visual as vi
from mod.env.config import ConfigNetwork

import mod.env.config as conf


from mod.env.car import Car, HiredCar
from mod.env.trip import get_trip_count_step, get_trips_random_ods
import mod.env.trip as tp
import mod.env.network as nw

from mod.env.simulator import PlotTrack
from mod.env.trip import ClassedTrip


# Reproducibility of the experiments
random.seed(1)


def move(arrival_t, current_t, lin, next_lin):
    # Subtract
    base_lin = np.zeros(len(lin))
    ones = np.ones(len(lin) - current_t)
    sub_range_ones = np.arange(current_t, len(lin))
    np.put(
        base_lin,
        sub_range_ones,
        ones
    )
    lin-=base_lin
    
    # Add
    base_next_lin = np.zeros(len(next_lin))
    ones = np.ones(len(next_lin) - arrival_t)
    add_range_ones = np.arange(arrival_t, len(lin))
    np.put(
        base_next_lin,
        add_range_ones,
        ones
    )
    next_lin+=base_next_lin


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
            ConfigNetwork.N_CLOSEST_NEIGHBORS: (
                (0, 8),
                (1, 8),
            ),
            # Only used when max idle step count is not None
            ConfigNetwork.N_CLOSEST_NEIGHBORS_EXPLORE: (
                (2, 16),
                (3, 16)
            ),
            # ConfigNetwork.REBALANCE_REACH: 2,
            ConfigNetwork.REBALANCE_MULTILEVEL: False,
            ConfigNetwork.LEVEL_DIST_LIST: [0, 60, 300, 600],
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
            ConfigNetwork.AGGREGATION_LEVELS: [

                adp.AggLevel(
                    temporal=1,
                    spatial=adp.DISAGGREGATE,
                    battery=adp.DISAGGREGATE,
                    contract=adp.DISCARD,
                    car_type=adp.DISAGGREGATE,
                    car_origin=adp.DISCARD,
                ),

                adp.AggLevel(
                    temporal=3,
                    spatial=2,
                    battery=adp.DISAGGREGATE,
                    contract=adp.DISCARD,
                    car_type=adp.DISAGGREGATE,
                    car_origin=adp.DISCARD,
                ),

                adp.AggLevel(
                    temporal=3,
                    spatial=3,
                    battery=adp.DISAGGREGATE,
                    contract=adp.DISCARD,
                    car_type=adp.DISAGGREGATE,
                    car_origin=adp.DISCARD,
                ),
            ],

            ConfigNetwork.LEVEL_TIME_LIST: [0.5, 1, 2, 3],
            ConfigNetwork.LEVEL_CAR_ORIGIN: {
                Car.TYPE_FLEET: {adp.DISCARD: adp.DISCARD},
                Car.TYPE_HIRED: {0: 0, 1: 1, 2: 2, 3: 3, 4: 4},
            },
            ConfigNetwork.LEVEL_CAR_TYPE: {
                Car.TYPE_FLEET: {
                    adp.DISAGGREGATE: Car.TYPE_FLEET,
                    adp.DISCARD: Car.TYPE_FLEET
                },
                Car.TYPE_HIRED: {
                    adp.DISAGGREGATE: Car.TYPE_HIRED,
                    adp.DISCARD: Car.TYPE_FLEET
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
            ConfigNetwork.TRIP_BASE_FARE: {
                tp.ClassedTrip.SQ_CLASS_1: 2.4,
                tp.ClassedTrip.SQ_CLASS_2: 2.4,
            },
            # -------------------------------------------------------- #
            # LEARNING ############################################### #
            # -------------------------------------------------------- #
            ConfigNetwork.DISCOUNT_FACTOR: 1,
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
            amod.config.min_contract_duration,
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
    episodes=450,
    enable_charging=False,
    # LOG ############################################################ #
    skip_steps=1,
    # HIRING ######################################################### #
    enable_hiring=False,
    contract_duration_h=2,
    sq_guarantee=False,
    universal_service=False,
    # TRIPS ########################################################## #
    classed_trips=True,
    # Create service rate and fleet status plots for each iteration
    save_plots=True,
    # Save .csv files for each iteration with fleet and demand statuses
    # throughtout all time steps
    save_df=True,
    # Save total reward, total service rate, and weights after iteration
    save_overall_stats=True,
    # Update value functions (dictionary in progress.npy file)
    # after n iterations (default n=1)
    save_progress=1,
    log_config_dict= {
        # la.LOG_DUALS: True,
        # la.LOG_FLEET_ACTIVITY: True,
        # la.LOG_VALUE_UPDATE: True,
        # la.LOG_COSTS: True,
        # la.LOG_SOLUTIONS: True,
        # la.LOG_WEIGHTS: True,
        # la.LOG_ALL: True,
        # la.LOG_LEVEL: la.DEBUG,
        # la.LEVEL_FILE: la.DEBUG,
        # la.LEVEL_CONSOLE: la.INFO,
        # la.FORMATTER_FILE: la.FORMATTER_TERSE,
    },
    log_mip=False,
    # If True, saves time details in file times.csv
    log_times=False,
    linearize_integer_model=False,
    use_artificial_duals=False,
):

    # Set tabu size (vehicles cannot visit nodes in tabu)
    Car.SIZE_TABU = config.car_size_tabu

    print(f"Saving experimental settings at: \"{config.exp_settings}\"")
    config.save()

    # ---------------------------------------------------------------- #
    # Episodes ####################################################### #
    # ---------------------------------------------------------------- #

    amod = AmodNetworkHired(config)
    episode_log = EpisodeLog(config=config, adp=amod.adp)
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

    print(f"Loading demand scenario '{config.demand_scenario}'...")

    try:
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
    for n in range(episode_log.n, episodes):

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
            trips_file_path = random.choice(conf.TRIP_FILES)

            step_trip_list, step_trip_count = tp.get_ny_demand(
                config, trips_file_path, amod.points
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
        amod.reset()

        # ------------------------------------------------------------ #
        # Plot fleet current status ################################## #
        # ------------------------------------------------------------ #
        if plot_track:

            # Computing initial timestep
            plot_track.compute_movements(0)

        start_time = time.time()

        outstanding = list()

        # Iterate through all steps and match requests to cars
        for step, trips in enumerate(deepcopy(step_trip_list)):

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

            for t in trips:
                logger.debug(f'  - {t}')

            if plot_track:
                # Update optimization time step
                plot_track.opt_step = step

                # Create trip dictionary of coordinates
                plot_track.trips_dict[step] = vi.compute_trips(trips)

            # ######################################################## #
            # TIME INCREMENT HAS PASSED ############################## #
            # ######################################################## #

            # Loop cars and update their current status as well as the
            # the list of available vehicles (change available and
            # available_hired)
            amod.update_fleet_status(step + 1)

            logger.debug("\n## Car attributes:")
            for c in amod.cars:
                logger.debug(f'{c} - {c.attribute}')
            # What each vehicle is doing after update?
            la.log_fleet_activity(
                config.log_path(amod.adp.n),
                step + 1,
                skip_steps,
                step_log,
                filter_status=[],
                msg="post update",
            )

            # print(
            #     f"#{step:>3} - hired={len(amod.hired_cars)}"
            #     f" - expired={len(amod.expired_contract_cars)}"
            #     f" - available={len(amod.available_hired)}"
            #     f" - favs={len(amod.step_favs.get(step, [])):>2}"
            #     f" - trips={len(trips):>2}")

            if enable_hiring:

                hired_cars = amod.step_favs.get(step, [])

                # if trips:

                #     hired_cars = hire_cars_trip_regions(
                #         amod, trips, contract_duration_h, step
                #     )

                #     logger.debug(
                #         f"**** Hiring {len(hired_cars)} in the trip regions."
                #     )

                # else:

                #     hired_cars = hire_cars_centers(
                #         amod,
                #         contract_duration_h,
                #         step,
                #         rc_level=config.level_rc,
                #     )

                #     logger.debug(
                #         f"**** Hiring {len(hired_cars)} in region centers."
                #     )

                # Add hired fleet to model
                amod.hired_cars.extend(hired_cars)
                amod.available_hired.extend(hired_cars)

            # count_car_point = defaultdict(int)
            # for c in amod.cars:
            #     count_car_point[c.point.id] += 1

            # for p, count in count_car_point.items():
            #     if count > 5:
            #         print(f"{step} - link {p} has {count} cars")
            # print(len(amod.cars), len(amod.available), len(amod.hired_cars), len(amod.available_hired))
            # Optimize
            revenue, serviced, rejected = service_trips(
                # Amod environment with configuration file
                amod,
                # Trips to be matched
                trips,
                # Service step (+1 trip placement step)
                step + 1,
                # Guarantee lowest pickup delay for a share of users
                # sq_guarantee=sq_guarantee,
                # All users are picked up
                universal_service=universal_service,
                # Allow recharging
                charge=enable_charging,
                # # Save mip .lp and .log of iteration n
                iteration=n,
                log_mip=log_mip,
                # Use hierarchical aggregation to update values
                use_artificial_duals=use_artificial_duals,
                # linearize_integer_model=linearize_integer_model,
                log_times=log_times,
                car_type_hide=Car.TYPE_FLEET
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
                    # Guarantee lowest pickup delay for a share of users
                    # sq_guarantee=sq_guarantee,
                    # All users are picked up
                    universal_service=universal_service,
                    # Allow recharging
                    charge=enable_charging,
                    # # Save mip .lp and .log of iteration n
                    iteration=n,
                    log_mip=log_mip,
                    # Use hierarchical aggregation to update values
                    use_artificial_duals=use_artificial_duals,
                    # linearize_integer_model=linearize_integer_model,
                    log_times=log_times,
                    car_type_hide=Car.TYPE_FLEET
                )

                revenue += revenue_fav,
                serviced += serviced_fav,
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

            # What each vehicle is doing?
            la.log_fleet_activity(
                config.log_path(amod.adp.n),
                int((step+1)/config.time_max_cars_link),
                skip_steps,
                step_log,
                filter_status=[],
                msg="after decision",
            )

            if save_plots or save_df:
                logger.debug("  - Computing fleet status...")
                # Compute fleet status after making decision in step - 1
                # What each car is doing when trips are arriving?
                step_log.compute_fleet_status(step+1)

            # Virtual hired cars are discarded
            # if enable_hiring:

            #     logger.debug(
            #         f"Total hired: {len(amod.hired_cars):4} "
            #         f"(available={len(amod.available_hired)})"
            #     )

            #     amod.discard_excess_hired()

            #     logger.debug(
            #         f"Total hired: {len(amod.hired_cars):4} "
            #         f"(available={len(amod.available_hired)})"
            #         " AFTER DISCARDING"
            #     )

            # -------------------------------------------------------- #
            # Update log with iteration ############################## #
            # -------------------------------------------------------- #
            step_log.add_record(revenue, serviced, rejected)

            # -------------------------------------------------------- #
            # Plotting fleet activity ################################ #
            # -------------------------------------------------------- #

            if plot_track:

                logger.debug("Computing movements...")
                plot_track.compute_movements(step + 1)
                logger.debug("Finished computing...")

                time.sleep(step_delay)

        amod.update_fleet_status(step + 1)

        # -------------------------------------------------------------#
        # Compute episode info #########################################
        # -------------------------------------------------------------#

        logger.debug("  - Computing iteration...")

        episode_log.compute_episode(
            step_log,
            time.time() - start_time,
            save_df=save_df,
            plots=save_plots,
            save_learning=save_progress,
            save_overall_stats=save_overall_stats,
        )

        # Clean weight track
        amod.adp.reset_weight_track()

        logger.info(
            f"####### "
            f"[Episode {n+1:>5}] "
            f"- {episode_log.last_episode_stats()} "
            f"#######"
        )

        # If True, saves time details in file times.csv
        if log_times:
            print("weighted values:", len(amod.adp.weighted_values))
            # print("get_state:", amod.adp.get_state.cache_info())
            # print("preview_decision:", amod.preview_decision.cache_info())
            # print("post_cost:", amod.post_cost.cache_info())

    # Plot overall performance (reward, service rate, and weights)
    episode_log.compute_learning()

    return amod.adp.reward


if __name__ == "__main__":

    # Isolate arguments (first is filename)
    args = sys.argv[1:]

    try:
        test_label = args[0]
    except:
        test_label = "TESTDE"

    hire = "-hire" in args
    log_adp = "-log_adp" in args
    log_mip = "-log_mip" in args
    save_plots = "-save_plots" in args
    save_df = "-save_df" in args
    log_times = "-log_times" in args
    use_duals = "-myopic" in args

    try:
        save_progress = int(args.index("-save_progress"))
    except:
        save_progress = 1

    try:
        iterations = args.index("-n")
        n_iterations = int(args[iterations + 1])
    except:
        n_iterations = 400

    try:
        fleet_size_i = args.index(f"-{ConfigNetwork.FLEET_SIZE}")
        fleet_size = int(args[fleet_size_i + 1])
    except:
        fleet_size = 100

    try:
        myopic_i = args.index(f"-{ConfigNetwork.MYOPIC}")
        myopic = bool(args[fleet_size_i + 1])
    except:
        myopic = False

    try:
        log_level_i = args.index("-level")
        log_level_label = args[log_level_i + 1]
        log_level = la.levels[log_level_label]
    except:
        log_level = la.INFO

    print("###### STARTING EXPERIMENTS")

    instance_name = None # f"{conf.FOLDER_OUTPUT}hiring_LIN_cars=0000-0300(L)_t=1_levels[5]=(1-0, 1-0, 1-0, 3-300, 3-600)_rebal=([0-8, 1-8][tabu=10])[L(05)][P]_[05h,+30m+04h+60m]_0.10(S)_1.00_0.10/exp_settings.json"
    if instance_name:
        print(f"Loading settings from \"{instance_name}\"")
        start_config = ConfigNetwork.load(instance_name)
    else:
        start_config = get_sim_config(
            {
                ConfigNetwork.TEST_LABEL: test_label,
                ConfigNetwork.DISCOUNT_FACTOR: 1,
                ConfigNetwork.PENALIZE_REBALANCE: True,
                ConfigNetwork.FLEET_SIZE: fleet_size,
                ConfigNetwork.DEMAND_RESIZE_FACTOR: 0.1,
                ConfigNetwork.DEMAND_TOTAL_HOURS: 4,
                ConfigNetwork.DEMAND_EARLIEST_HOUR: 5,
                ConfigNetwork.OFFSET_TERMINATION_MIN: 60,
                ConfigNetwork.OFFSET_REPOSITIONING_MIN: 30,
                ConfigNetwork.TIME_INCREMENT: 1,
                ConfigNetwork.DEMAND_SAMPLING: True,
                ConfigNetwork.SQ_GUARANTEE: False,
                ConfigNetwork.MAX_CARS_LINK: 5,
                # 10 steps = 5 min
                ConfigNetwork.TIME_MAX_CARS_LINK: 5,
                ConfigNetwork.LINEARIZE_INTEGER_MODEL: False,
                ConfigNetwork.USE_ARTIFICIAL_DUALS: False,
                # Controlling user matching
                ConfigNetwork.MATCHING_DELAY: 15,
                ConfigNetwork.ALLOW_USER_BACKLOGGING: False,
                ConfigNetwork.MAX_IDLE_STEP_COUNT: None,
                # When cars start in the last visited point, the model takes
                # a long time to figure out the best time
                ConfigNetwork.FLEET_START: conf.FLEET_START_LAST,
                ConfigNetwork.CAR_SIZE_TABU: 10,
                ConfigNetwork.REACHABLE_NEIGHBORS: False,
                ConfigNetwork.ADP_IGNORE_ZEROS: True,
                ConfigNetwork.DEPOT_SHARE: None,
                ConfigNetwork.FAV_DEPOT_LEVEL: 2,
                ConfigNetwork.FAV_FLEET_SIZE: 0,
                ConfigNetwork.SEPARATE_FLEETS: False,
                ConfigNetwork.MYOPIC: myopic,
                # ConfigNetwork.PARKING_RATE_MIN = 1.50/60 # 1.50/h
                # ConfigNetwork.PARKING_RATE_MIN = 0.1*20/60,  # = rebalancing 1 min
                ConfigNetwork.PARKING_RATE_MIN: 0  # = rebalancing 1 min
            }
        )

    ClassedTrip.q_classes = dict(A=1.0, B=0.9)
    ClassedTrip.sq_level_class = dict(A=[0, 0], B=[0, 0])
    # min_max_time_class = dict(A=dict(min=3, max=3), B=dict(min=3, max=6))
    ClassedTrip.min_max_time_class = dict(A=dict(min=1, max=3), B=dict(min=4,max=9))
    ClassedTrip.class_proportion = dict(A=0.0, B=1)

    # Toggle what is going to be logged
    log_config = {
        la.LOG_DUALS: True,
        la.LOG_FLEET_ACTIVITY: True,
        la.LOG_VALUE_UPDATE: False,
        la.LOG_COSTS: True,
        la.LOG_SOLUTIONS: False,
        la.LOG_WEIGHTS: False,
        la.LOG_ALL: log_adp,
        la.LOG_LEVEL: (log_level if not log_adp else la.DEBUG),
        la.LEVEL_FILE: la.DEBUG,
        la.LEVEL_CONSOLE: la.INFO,
        la.FORMATTER_FILE: la.FORMATTER_TERSE,
    }

    alg_adp(
        None,
        start_config,
        episodes=n_iterations,
        contract_duration_h=2,
        sq_guarantee=hire,
        enable_hiring=hire,
        universal_service=False,
        log_config_dict=log_config,
        log_mip=log_mip,
        log_times=log_times,
        save_plots=save_plots,
        save_progress=save_progress,
        save_df=save_df,
        use_artificial_duals=start_config.use_artificial_duals
    )
