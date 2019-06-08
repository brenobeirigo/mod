import os
import sys
import numpy as np
from pprint import pprint
from threading import Thread
import time

# Adding project folder to import modules
root = os.getcwd().replace("\\", "/")
sys.path.append(root)

from mod.env.amod import AmodNetwork
from mod.env.visual import StepLog, EpisodeLog
import mod.env.visual as vi
from mod.env.config import ConfigNetwork, NY_TRIPS_EXCERPT_DAY
from mod.env.match import adp_network
from mod.env.car import Car
from mod.env.trip import get_trip_count_step, get_trips_random_ods
import mod.env.network as nw

from mod.env.simulator import PlotTrack


run_plot = PlotTrack(0, 0, 0)


def sim(plot_track):

    step_delay = PlotTrack.STEP_DELAY
    enable_plot = PlotTrack.ENABLE_PLOT

    print(step_delay, enable_plot)
    # Pull graph info
    region, label, node_count, center_count, edge_count = nw.query_info()

    info = (
        "##############################################################"
        f"### Region: {region} G(V={node_count}, E={edge_count})"
        f"### Center count: {center_count}"
    )

    print(info)
    # -----------------------------------------------------------------#
    # Amod environment #################################################
    # -----------------------------------------------------------------#

    config = ConfigNetwork()
    config.update(
        {
            ConfigNetwork.NAME: label,
            # Fleet
            ConfigNetwork.FLEET_SIZE: 1500,
            ConfigNetwork.BATTERY_LEVELS: 20,
            # Time - Increment (min)
            ConfigNetwork.TIME_INCREMENT: 1,
            ConfigNetwork.OFFSET_REPOSIONING: 2,
            ConfigNetwork.OFFSET_TERMINATION: 30,
            # NETWORK ##################################################
            # Region centers are created in steps of how much time?
            ConfigNetwork.STEP_SECONDS: 30,
            # Demand spawn from how many centers?
            ConfigNetwork.ORIGIN_CENTERS: 2,
            # Demand arrives in how many centers?
            ConfigNetwork.DESTINATION_CENTERS: 2,
            # OD level extension
            ConfigNetwork.DEMAND_CENTER_LEVEL: 3,
            # Cars rebalance to up to #region centers
            ConfigNetwork.N_CLOSEST_NEIGHBORS: 2,
            # Cars can access locations within region centers
            # established in which neighborhood level?
            ConfigNetwork.NEIGHBORHOOD_LEVEL: 1,
            # ConfigNetwork.LEVEL_DIST_LIST: [0, 30, 60, 120, 180, 300, 600],
            # How many levels separated by step seconds? If None, ad-hoc
            # LEVEL_DIST_LIST must be filled
            ConfigNetwork.AGGREGATION_LEVELS: 8,
            ConfigNetwork.SPEED: 30,
        }
    )

    # "centers":{"30":684,"60":235,"90":119,"120":73,"150":50,"180":37,"210":32,"240":24,"270":19,"300":16,"330":13,"360":11,"390":9,"420":9,"450":8,"480":7,"510":6,"540":5,"570":5,"600":4}

    # ---------------------------------------------------------------- #
    # Episodes ####################################################### #
    # ---------------------------------------------------------------- #
    episodes = 600
    episodeLog = EpisodeLog(config=config)
    amod = AmodNetwork(config)

    plot_track.env = amod

    step_car_path = plot_track.step_car_path_dict

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

    origins, destinations = episodeLog.get_od_lists(amod)

    # Get demand pattern from NY city
    step_trip_count = get_trip_count_step(
        NY_TRIPS_EXCERPT_DAY,
        step=config.time_increment,
        multiply_for=config.demand_resize_factor,
        earliest_step=config.demand_earliest_step_min,
        max_steps=config.demand_max_steps,
    )

    print(
        f"### DEMAND ###"
        f" - min: {min(step_trip_count)}"
        f" - max: {max(step_trip_count)}"
    )

    # ---------------------------------------------------------------- #
    # Experiment ##################################################### #
    # ---------------------------------------------------------------- #

    # Loop all episodes, pick up trips, and learn where they are
    for n in range(episodeLog.n, episodes):

        plot_track.plot_episode = n

        # Create all episode trips
        step_trip_list = get_trips_random_ods(
            amod.points,
            step_trip_count,
            offset_start=amod.config.offset_repositioning,
            offset_end=amod.config.offset_termination,
            origins=origins,
            destinations=destinations,
        )

        # Start saving data of each step in the adp_network
        step_log = StepLog(amod)

        # Resetting environment
        amod.reset()

        # ---------step_car_path--------------------------- #
        # Plot fleet current status ################################## #
        # ------------------------------------------------------------ #
        if enable_plot:

            # Computing initial timestep
            plot_track.compute_movements(0)

        # Iterate through all steps and match requests to cars
        for step, trips in enumerate(step_trip_list):

            if enable_plot:
                plot_track.opt_step = step
                plot_track.trips_dict[step] = vi.compute_trips(trips)

            revenue, serviced, rejected = adp_network(
                amod,
                trips,
                step + 1,
                neighborhood_level=config.neighborhood_level,
                n_neighbors=config.n_neighbors,
                # agg_level=amod.config.incumbent_aggregation_level,
            )

            # ---------------------------------------------------------#
            # Update log with iteration ################################
            # ---------------------------------------------------------#
            step_log.add_record(revenue, serviced, rejected)

            # Show time step statistics
            step_log.show_info()

            # What each vehicle is doing?
            # amod.print_fleet_stats()

            # ---------------------------------------------------------#
            # Plotting fleet activity ################################ #
            # ---------------------------------------------------------#
            print("Computing movements...")
            if enable_plot:
                plot_track.compute_movements(step + 1)
            print("Finished computing...")

            time.sleep(step_delay)

        # -------------------------------------------------------------#
        # Compute episode info #########################################
        # -------------------------------------------------------------#
        # print(
        #     f"####### "
        #     f"[Episode {n:>5}] "
        #     f"- {episodeLog.last_episode_stats()} "
        #     f"#######"
        # )
        episodeLog.compute_episode(
            step_log, weights=amod.get_weights(), progress=True
        )

    episodeLog.compute_learning()


run_plot.start_animation(sim)
