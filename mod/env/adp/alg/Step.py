import time

import mod.util.log_util as la
from mod.env.fleet.Car import Car
from mod.env.matching import (
    Matching,
    play_decisions,
    mpc,
)


class Step:
    def __init__(self, iteration, amod, step, trip_list):
        self.amod = amod
        self.iteration = iteration
        self.step = step
        self.revenue = []
        self.serviced = []
        self.rejected = []

        self.outstanding = []
        self.trips = trip_list

    def mpc_optimal_play_decisions(self, it_decisions):
        # print(
        #     f"it={step:04} - Playing decisions {len(it_decisions[step])}"
        # )
        self.revenue, self.serviced, self.rejected = play_decisions(
            self.amod, self.trips, self.step + 1, it_decisions[self.step]
        )

    def mpc_method(self, it_step_trip_list):
        # Predicted trips for next steps (exclusive)
        predicted_trips = it_step_trip_list[
                          self.step + 1: self.step + self.amod.config.mpc_forecasting_horizon
                          ]

        # Trips within the same region are invalid
        decisions = mpc(
            # Amod environment with configuration file
            self.amod,
            # Trips to be matched
            self.trips,
            # Predicted trips within the forecasting horizon
            predicted_trips,
            # Service step (+1 trip placement step)
            step=self.step + 1,
            log_mip=self.amod.config.log_config_dict[la.LOG_MIP],
        )
        self.revenue, self.serviced, self.rejected = play_decisions(
            self.amod, self.trips, self.step + 1, decisions
        )

    def adp_separate_fleet(self):

        matching = Matching(
            # Amod environment with configuration file
            self.amod,
            # Trips to be matched
            self.rejected,
            # Service step (+1 trip placement step)
            self.step + 1,
            # Save mip .lp and .log of iteration n
            iteration=self.iteration.n,
            car_type_hide=Car.TYPE_FLEET,
            log_times=self.amod.config.log_config_dict[la.LOG_TIMES],
            log_mip=self.amod.config.log_config_dict[la.LOG_MIP]
        )

        # Optimize
        revenue_fav, serviced_fav, rejected_fav = matching.service_trips()

        self.revenue += (revenue_fav,)
        self.serviced += (serviced_fav,)
        self.rejected = rejected_fav

    def adp_method(self):
        matching = Matching(
            # Amod environment with configuration file
            self.amod,
            # Trips to be matched
            self.trips,
            # Service step (+1 trip placement step)
            self.step + 1,
            # Save mip .lp and .log of iteration n
            iteration=self.iteration.n,
            log_mip=self.amod.config.log_config_dict[la.LOG_MIP],
            log_times=self.amod.config.log_config_dict[la.LOG_TIMES],
            car_type_hide=Car.TYPE_FLEET
        )

        self.revenue, self.serviced, self.rejected = matching.service_trips()

    def backlog_users(self):
        expired = []
        # print("outstanding:", len(self.outstanding))
        for r in self.rejected:

            # Add time increment to backlog delay
            r.backlog_delay += self.amod.config.time_increment
            r.times_backlogged += 1

            # print(r.backlog_delay, ">", self.amod.config.max_user_backlogging_delay)
            # Max. backlog reached -> discard trip
            if (
                    r.backlog_delay
                    > self.amod.config.max_user_backlogging_delay
                    or self.step + 1 == self.amod.config.time_steps
            ):
                expired.append(r)
            else:
                self.outstanding.append(r)

        # print("outstanding:", len(self.outstanding))
        # print(len(expired), [r.times_backlogged for r in expired])

        self.rejected = expired

    def rebalance_to_not_serviced(self, iteration):

        # If reactive rebalance, send vehicles to rejected
        # user's origins
        iteration.logger.debug(
            "####################"
            f"[{iteration.n:04}]-[{self.step:04}] REACTIVE REBALANCE "
            "####################"
        )
        iteration.logger.debug("Rejected requests (rebalancing targets):")
        for r in self.rejected:
            iteration.logger.debug(f"{r}")

        # print(step, amod.available_fleet_size, len(rejected))
        # Update fleet headings to isolate Idle vehicles.
        # Only empty cars are considered for rebalancing.
        t1 = time.time()
        self.amod.update_fleet_status(self.step + 1)
        iteration.execution_time_dict["t_update"] += time.time() - t1

        t1 = time.time()

        iteration.execution_time_dict["t_log"] += time.time() - t1

        # Service idle vehicles
        matching = Matching(
            # Amod environment with configuration file
            self.amod,
            # Trips to be matched
            self.rejected + self.outstanding,
            # Service step (+1 trip placement step)
            self.step + 1,
            # # Save mip .lp and .log of iteration n
            iteration=iteration.n,
            log_mip=self.amod.config.log_config_dict[la.LOG_MIP],
            log_times=self.amod.config.log_config_dict[la.LOG_TIMES],
            car_type_hide=Car.TYPE_FLEET,
            reactive=True,
        )

        rebal_costs, _, _ = matching.service_trips()

        self.revenue -= rebal_costs
        iteration.logger.debug(f"\n# REB. COSTS: {rebal_costs:6.2f}")

    def apply_reactive_rebalancing(self):
        return self.amod.config.policy_reactive and (self.rejected or self.outstanding)
