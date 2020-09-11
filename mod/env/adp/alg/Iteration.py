import itertools
import time
from collections import defaultdict
from copy import deepcopy

import mod.env.config as conf
import mod.env.demand.trip_util as tp
import mod.util.log_util as la
from mod.env.adp.alg.Scenario import ScenarioUmbalanced, ScenarioNYC
from mod.env.matching import optimal_rebalancing


class Iteration:
    def __init__(self, n, amod):
        self.n = n
        self.amod = amod

        # Times
        self.execution_time_dict = defaultdict(float)

        self.amod.config.current_iteration = n

        self.load_tripdata_scenario(n)

        self.save_sampled_tripdata_from_iteration()

        self.amod.bound_max_cars_at_trip_destinations_from(
            self.scenario.step_trip_list
        )

        self.logger = self.get_iteration_logger()

        self.logger.debug(self.__str__())

        # Trips from this iteration (make sure it can be used again)
        self.it_step_trip_list = deepcopy(self.scenario.step_trip_list)

        # From optimal mpc
        self.new_fleet_size = self.amod.fleet_size
        self.it_decisions = []
        if self.amod.config.policy_optimal:
            self.mpc_optimal()

    def compute(
        self,
        step_log,
        episode_log,
        total_execution_time,
        save_overall_stats=True,
    ):
        self.logger.debug("  - Computing iteration...")
        t1 = time.time()
        episode_log.compute_episode(
            step_log,
            self.it_step_trip_list,
            total_execution_time,
            fleet_size=self.new_fleet_size,
            save_df=self.amod.config.log_config_dict[la.SAVE_DF],
            plots=self.amod.config.log_config_dict[la.SAVE_PLOTS],
            save_learning=self.amod.config.save_progress,
            save_overall_stats=save_overall_stats,
        )
        self.execution_time_dict["t_epi"] = time.time() - t1

    def assign(self, current_step, step_log):
        t1 = time.time()
        if self.amod.config.policy_optimal:
            current_step.mpc_optimal_play_decisions(self.it_decisions)

        elif self.amod.config.policy_mpc:
            current_step.mpc_method(self.it_step_trip_list)
        else:
            current_step.adp_method()
        if self.amod.config.separate_fleets:
            current_step.adp_separate_fleet()
        # ######################################################## #
        # METHOD - BACKLOG ####################################### #
        # ######################################################## #
        if self.amod.config.max_user_backlogging_delay > 0:
            current_step.backlog_users()
        self.execution_time_dict["t_mip"] += time.time() - t1
        # ######################################################## #
        # METHOD - REACTIVE REBALANCE ############################ #
        # ######################################################## #
        self.log_pre_rebalancing(current_step, step_log)

        if current_step.apply_reactive_rebalancing():
            t_reactive_rebalance_1 = time.time()
            current_step.rebalance_to_not_serviced(self)
            self.execution_time_dict["t_reactive_rebalance"] += (
                time.time() - t_reactive_rebalance_1
            )

    def log_pre_rebalancing(self, current_step, step_log, skip_steps=1):
        t1 = time.time()
        # What each vehicle is doing?
        la.log_fleet_activity(
            self.amod.config.log_path(self.amod.adp.n),
            current_step.step,
            skip_steps,
            step_log,
            filter_status=[],
            msg="before rebalancing",
        )
        self.execution_time_dict["t_log"] += time.time() - t1

    def log_post_decision(self, current_step, step_log, skip_steps=1):
        t1 = time.time()
        # What each vehicle is doing?
        la.log_fleet_activity(
            self.amod.config.log_path(self.amod.adp.n),
            current_step.step,
            skip_steps,
            step_log,
            filter_status=[],
            msg="after decision",
        )
        self.execution_time_dict["t_log"] += time.time() - t1

    def log_pre_decision(self, current_step, step_log, skip_steps=1):
        t1 = time.time()
        self.logger.debug("\n## Car attributes:")
        # Log both fleets
        for c in itertools.chain(self.amod.cars, self.amod.hired_cars):
            self.logger.debug(f"{c} - {c.attribute}")
        # What each vehicle is doing after update?
        la.log_fleet_activity(
            self.amod.config.log_path(self.amod.adp.n),
            current_step.step + 1,
            skip_steps,
            step_log,
            filter_status=[],
            msg="post update",
        )
        self.execution_time_dict["t_log"] += time.time() - t1

    def update_amod_fleet_status(self, current_step):
        # Loop cars and update their current status as well as the
        # the list of available vehicles (change available and
        # available_hired)
        t1 = time.time()
        # If policy is reactive, rebalancing cars can be rerouted
        # from the intermediate nodes along the shortest path
        # to the rebalancing target. Notice that, if level > 0,
        # middle points will correspond to corresponding hierarchi-
        # cal superior node.
        self.amod.update_fleet_status(
            current_step.step + 1,
            use_rebalancing_cars=self.amod.config.policy_reactive,
        )
        self.execution_time_dict["t_update"] += time.time() - t1

    def get_iteration_logger(self):
        return la.get_logger(
            self.amod.config.log_path(self.amod.adp.n),
            log_file=self.amod.config.log_path(self.amod.adp.n),
            **self.amod.config.log_config_dict,
        )

    def load_tripdata_scenario(self, n):
        if self.amod.config.demand_scenario == conf.SCENARIO_UNBALANCED:
            self.trips_file_path = conf.TRIP_FILES[0]
            self.test_i = n
            self.scenario = ScenarioUmbalanced(self.amod, self.trips_file_path)

        elif self.amod.config.demand_scenario == conf.SCENARIO_NYC:
            (
                self.test_i,
                self.trips_file_path,
            ) = self.amod.config.get_demand_file_index(self.n)
            self.scenario = ScenarioNYC(
                self.amod, self.trips_file_path, self.n
            )

    def save_sampled_tripdata_from_iteration(self):
        if self.amod.config.save_trip_data:
            df = tp.get_df_from_sampled_trips(self.scenario.step_trip_list)
            df.to_csv(
                f"{self.amod.config.sampled_tripdata_path}trips_{self.test_i:04}.csv",
                index=False,
            )

    def save_fleet_data(self):
        # Save random data (initial positions)
        if self.amod.config.save_fleet_data:
            # Save car distribution
            df_cars = self.amod.get_fleet_df()
            df_cars.to_csv(
                f"{self.amod.config.fleet_data_path}cars_{self.test_i:04}.csv",
                index=False,
            )

    def save_fleet_data_result(self):
        if self.amod.config.save_fleet_data:
            df_cars = self.amod.get_fleet_df()
            df_cars.to_csv(
                f"{self.amod.config.fleet_data_path}cars_{self.n:04}_result.csv",
                index=False,
            )

    def log(self, s):
        self.logger.debug(s)

    def mpc_optimal(self):
        (
            self.it_decisions,
            it_step_trip_list_distinct_od_areas,
            self.new_fleet_size,
        ) = optimal_rebalancing(
            self.amod,
            self.it_step_trip_list,
            log_mip=self.amod.config.log_config_dict[la.LOG_MIP],
        )

        self.it_step_trip_list = it_step_trip_list_distinct_od_areas

        print(f"MPC optimal fleet size: {self.new_fleet_size}")

    def save_trip_data(self):
        if self.amod.config.save_trip_data:
            df = tp.get_df_from_sampled_trips(
                self.it_step_trip_list,
                show_service_data=True,
                earliest_datetime=self.amod.config.demand_earliest_datetime,
            )

            df.to_csv(
                f"{self.amod.config.sampled_tripdata_path}trips_{self.n:04}_result.csv",
                index=False,
            )

    def log_summary_stats(self, step_log, episode_log):
        self.logger.info(
            f"####### "
            f"[Episode {self.n + 1:>5}] "
            f"- {episode_log.last_episode_stats()} "
            f"serviced={step_log.serviced}, "
            f"rejected={step_log.rejected}, "
            f"total={step_log.total} "
            f"t(episode={self.execution_time_dict['t_epi']:.2f}, "
            f"t_log={self.execution_time_dict['t_log']:.2f}, "
            f"t_mip={self.execution_time_dict['t_mip']:.2f}, "
            f"t_save_plots={self.execution_time_dict['t_save_plots']:.2f}, "
            f"t_up={self.execution_time_dict['t_update']:.2f}, "
            f"t_add_record={self.execution_time_dict['t_add_record']:.2f})"
            f"#######"
        )

    def __str__(self):
        return (
            "##################################"
            f" Iteration {self.n:04} "
            f"- Demand (min={min(self.scenario.step_trip_count)}"
            f", max={max(self.scenario.step_trip_count)})"
            f", step={self.amod.config.time_increment}"
            f", earliest_step={self.amod.config.demand_earliest_step_min}"
            f", max_steps={self.amod.config.demand_max_steps}"
            f", offset_start={self.amod.config.offset_repositioning_steps}"
            f", offset_end={self.amod.config.offset_termination_steps}"
            f", steps={self.amod.config.time_steps}"
        )
