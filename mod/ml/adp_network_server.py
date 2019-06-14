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
    # -----------------------------------------------------------------#
    # Amod environment #################################################
    # -----------------------------------------------------------------#

    config.update(
        {
            # Network
            ConfigNetwork.NAME: label,
            ConfigNetwork.REGION: region,
            ConfigNetwork.NODE_COUNT: node_count,
            ConfigNetwork.EDGE_COUNT: edge_count,
            ConfigNetwork.CENTER_COUNT: center_count,
            # Fleet
            ConfigNetwork.FLEET_SIZE: 400,
            ConfigNetwork.BATTERY_LEVELS: 20,
            # Time - Increment (min)
            ConfigNetwork.TIME_INCREMENT: 1,
            ConfigNetwork.OFFSET_REPOSIONING: 15,
            ConfigNetwork.OFFSET_TERMINATION: 30,
            # NETWORK ##################################################
            # Region centers are created in steps of how much time?
            ConfigNetwork.STEP_SECONDS: 30,
            # Demand spawn from how many centers?
            ConfigNetwork.ORIGIN_CENTERS: 3,
            # Demand arrives in how many centers?
            ConfigNetwork.DESTINATION_CENTERS: 3,
            # OD level extension
            ConfigNetwork.DEMAND_CENTER_LEVEL: 3,
            # Cars rebalance to up to #region centers
            ConfigNetwork.N_CLOSEST_NEIGHBORS: 8,
            # Cars can access locations within region centers
            # established in which neighborhood level?
            ConfigNetwork.NEIGHBORHOOD_LEVEL: 3,
            # Cars can rebalance to neighbor centers of level:
            ConfigNetwork.REBALANCE_LEVEL: 1,
            ConfigNetwork.LEVEL_DIST_LIST: [0, 30, 60, 120, 300],
            # How many levels separated by step seconds? If None, ad-hoc
            # LEVEL_DIST_LIST must be filled
            ConfigNetwork.AGGREGATION_LEVELS: 5,
            ConfigNetwork.SPEED: 30,
            # Demand
            ConfigNetwork.DEMAND_TOTAL_HOURS: 6,
            ConfigNetwork.DEMAND_EARLIEST_HOUR: 14,
        }
    )
    return config


run_plot = PlotTrack(get_sim_config())


def sim(plot_track, config):

    step_delay = PlotTrack.STEP_DELAY
    enable_plot = PlotTrack.ENABLE_PLOT

    # ---------------------------------------------------------------- #
    # Episodes ####################################################### #
    # ---------------------------------------------------------------- #
    episodes = 1500
    episode_log = EpisodeLog(config=config)
    amod = AmodNetwork(config)

    plot_track.env = amod

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

    origins, destinations = episode_log.get_od_lists(amod)

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
    for n in range(episode_log.n, episodes):

        plot_track.opt_episode = n

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

        # ------------------------------------------------------------ #
        # Plot fleet current status ################################## #
        # ------------------------------------------------------------ #
        if enable_plot:

            # Computing initial timestep
            plot_track.compute_movements(0)

        # Iterate through all steps and match requests to cars
        for step, trips in enumerate(step_trip_list):

            if enable_plot:
                # Update optimization time step
                plot_track.opt_step = step

                # Assign trips at step
                plot_track.trips_dict[step] = vi.compute_trips(trips)

            # Loop cars and update their current status as well as the
            # the list of available vehicles.
            amod.update_fleet_status(step)

            # Compute fleet status
            step_log.compute_fleet_status()

            # Stats summary
            # print(" - Pre-decision statuses:")
            # amod.print_fleet_stats_summary()

            # Optimize
            revenue, serviced, rejected = adp_network(
                amod,
                trips,
                step + 1,
                neighborhood_level=config.neighborhood_level,
                n_neighbors=config.n_neighbors,
            )

            # ---------------------------------------------------------#
            # Update log with iteration ################################
            # ---------------------------------------------------------#
            step_log.add_record(revenue, serviced, rejected)

            # Show time step statistics
            step_log.show_info()

            # What each vehicle is doing?
            # amod.print_fleet_stats()

            # Stats summary
            # print(" - Post-decision statuses")
            # amod.print_fleet_stats_summary()

            # ---------------------------------------------------------#
            # Plotting fleet activity ################################ #
            # ---------------------------------------------------------#
            # print("Computing movements...")
            if enable_plot:
                plot_track.compute_movements(step + 1)
                # print("Finished computing...")

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
        episode_log.compute_episode(
            step_log, weights=amod.get_weights(), progress=True
        )

    episode_log.compute_learning()


run_plot.start_animation(sim)
