import os
import sys
import time

# Adding project folder to import modules
import mod.env.Point

root = os.getcwd().replace("\\", "/")
sys.path.append(root)
import mod.util.log_util as la

from mod.env.amod.AmodNetworkHired import AmodNetworkHired
from mod.env.visual import StepLog, EpisodeLog

from mod.env.adp.alg.Iteration import Iteration
from mod.env.adp.alg.Step import Step

import mod.env.visual as vi
from mod.env.config import ConfigNetwork

from mod.env.fleet.Car import Car
import mod.env.network as nw
from mod.env.simulator import PlotTrack


class ValueIteration:
    def __init__(
        self,
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

        self.plot_track = plot_track
        self.config = config
        self.step_delay = step_delay
        self.skip_steps = skip_steps
        self.classed_trips = classed_trips
        self.save_plots = save_plots
        self.save_df = save_df
        self.save_overall_stats = save_overall_stats
        self.log_config_dict = (
            log_config_dict if log_config_dict else la.get_standard()
        )
        self.config.log_config_dict = self.log_config_dict

        self.amod = AmodNetworkHired(self.config, online=True)
        self.print_amod_node_reachability_stats()
        self.episode_log = EpisodeLog(self.amod)
        self.first_iteration = 0

    def print_amod_node_reachability_stats(self):
        print(
            f"### Nodes with no neighbors (within time increment) "
            f"({len(self.amod.unreachable_ods)})"
            f" = {self.amod.unreachable_ods}"
            f" --- #neighbors (avg, max, min) = {self.amod.stats_neighbors}"
        )

    def comeca(self):

        # Set tabu size (vehicles cannot visit nodes in tabu)
        Car.SIZE_TABU = self.config.car_size_tabu

        print(
            f'### Saving experimental settings at: "{self.config.exp_settings}"'
        )
        self.config.save()

        # ---------------------------------------------------------------- #
        # Episodes ####################################################### #
        # ---------------------------------------------------------------- #

        if self.plot_track:
            self.plot_track.set_env(self.amod)

        # ---------------------------------------------------------------- #
        # Plot centers and guidelines #################################### #
        # ---------------------------------------------------------------- #
        if self.plot_track:
            self.plot_track.plot_centers(
                self.amod.points,
                mod.env.Point.Point.levels,
                mod.env.Point.Point.levels[self.config.demand_center_level],
                mod.env.Point.Point.levels[self.config.neighborhood_level],
                show_sp_lines=PlotTrack.SHOW_SP_LINES,
                show_lines=PlotTrack.SHOW_LINES,
            )

        print(
            f"### Loading demand scenario '{self.config.demand_scenario}'..."
        )

        self.load_progress()

        print(
            f" - Iterating from {self.first_iteration:>4} to {self.config.iterations:>4}..."
        )

    def load_progress(self):
        try:
            if self.config.ignore_training:
                print("Ignore training.")
            else:
                # Load last episode
                self.episode_log.load_progress()
                print("Data loaded successfully.")

                # Loop all episodes, pick up trips, and learn where they are
                if self.config.train:
                    self.first_iteration = self.episode_log.n

        except Exception as e:
            print(f"No previous episodes were saved (Exception: '{e}').")

    def init(self):

        self.comeca()

        for n in range(self.first_iteration, self.config.iterations):

            t_start_episode = time.time()

            iteration = Iteration(n, self.amod)

            if self.plot_track:
                self.plot_track.opt_episode = n

            # Start saving data of each step in the adp_network
            step_log = StepLog(self.amod)

            # Resetting environment
            self.amod.reset(seed=n)
            iteration.save_fleet_data()

            self.plot_init_episodes()

            self.execute_timesteps(iteration, step_log)

            iteration.save_trip_data()
            iteration.save_fleet_data_result()

            total_execution_time = time.time() - t_start_episode
            iteration.compute(
                step_log,
                self.episode_log,
                total_execution_time,
                save_overall_stats=self.save_overall_stats,
            )

            # Clean weight track
            self.amod.adp.reset_weight_track()

            iteration.log_summary_stats(step_log, self.episode_log)

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
            if self.config.idle_annealing is not None:
                # By the end of all iterations, cars cannot be forced to
                # rebalance anymore
                self.config.config[
                    ConfigNetwork.IDLE_ANNEALING
                ] += 1  # 1/episodes

        # Plot overall performance (reward, service rate, and weights)
        self.episode_log.compute_learning()

        return self.amod.adp.reward

    def execute_timesteps(self, iteration, step_log):

        # Start computation
        self.compute_fleet_status(iteration, step_log)

        current_step = None
        outstanding_trips_from_previous_step = []
        # Iterate through all steps and match requests to cars
        for step_index, trip_list in enumerate(iteration.it_step_trip_list):
            trips = trip_list + outstanding_trips_from_previous_step
            current_step = Step(iteration, self.amod, step_index, trips)

            self.config.current_step = current_step.step

            self.plot_create_dict_coordinates(current_step)

            # ######################################################## #
            # TIME INCREMENT HAS PASSED ############################## #
            # ######################################################## #
            self.amod.hire_favs_available_at_step(current_step.step)

            iteration.update_amod_fleet_status(current_step)

            # Show the top highest vehicle count per position
            # amod.show_count_vehicles_top(step, 5)

            iteration.log_pre_decision(current_step, step_log)

            iteration.assign(current_step, step_log)

            self.compute_fleet_status(iteration, step_log)

            # -------------------------------------------------------- #
            # Update log with iteration ############################## #
            # -------------------------------------------------------- #
            t1 = time.time()
            step_log.add_record(current_step)
            iteration.execution_time_dict["t_add_record"] += time.time() - t1

            iteration.log_post_decision(current_step, step_log)

            self.plot_fleet_activity(current_step, iteration)

            # print(step, "weighted value:", amod.adp.get_weighted_value.cache_info())
            # print(step, "preview decision:", amod.preview_decision.cache_info())
            # print(step, "preview decision:", amod.preview_move.cache_info())
            # amod.adp.get_weighted_value.cache_clear()
            # self.post_cost.cache_clear()

            outstanding_trips_from_previous_step = current_step.outstanding

        iteration.update_amod_fleet_status(current_step)

    def plot_fleet_activity(self, current_step, iteration):
        # -------------------------------------------------------- #
        # Plotting fleet activity ################################ #
        # -------------------------------------------------------- #
        if self.plot_track:
            iteration.logger.debug("Computing movements...")
            self.plot_track.compute_movements(current_step.step + 1)
            iteration.logger.debug("Finished computing...")

            time.sleep(self.step_delay)

    def plot_init_episodes(self):
        # ------------------------------------------------------------ #
        # Plot fleet current status ################################## #
        # ------------------------------------------------------------ #
        if self.plot_track:
            # Computing initial timestep
            self.plot_track.compute_movements(0)

    def plot_create_dict_coordinates(self, current_step):
        if self.plot_track:
            # Update optimization time step
            self.plot_track.opt_step = current_step.step

            # Create trip dictionary of coordinates
            self.plot_track.trips_dict[current_step.step] = vi.compute_trips(
                current_step.trips
            )

    def compute_fleet_status(self, iteration, step_log):
        t1 = time.time()
        if self.log_fleet_status():
            # Compute fleet status after making decision in step - 1
            # What each car is doing when trips are arriving?
            iteration.logger.debug("  - Computing fleet status...")
            step_log.compute_fleet_status()
        iteration.execution_time_dict["t_save_plots"] += time.time() - t1

    def log_fleet_status(self):
        return (
            self.log_config_dict[la.SAVE_PLOTS]
            or self.log_config_dict[la.SAVE_DF]
            or self.log_config_dict[la.LOG_STEP_SUMMARY]
        )

