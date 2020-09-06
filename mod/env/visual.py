import itertools as it
import os
from collections import defaultdict
from pprint import pprint

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import mod.env.config as conf
import mod.env.network as nw
from mod.env.car import Car
from mod.env.network import Point

sns.set(style="ticks")
sns.set_context("paper")
np.set_printoptions(precision=3)

FOLDER_TESTING = "/adp/test"
FOLDER_TRAINING = "/adp/train"
FOLDER_MYOPIC = "/myopic"
FOLDER_POLICY_RANDOM = "/random"

ADP_LOGS = "adp_logs/"
FOLDER_TIME = "time/"
FOLDER_FLEET = "fleet/"
FOLDER_SERVICE = "service/"
PROGRESS_FILENAME = "progress.npy"


class EpisodeLog:

    @property
    def output_path(self):
        return self.config.output_path
        # if self.config.short_path:
        #     label = self.config.label_md5
        # else:
        #     label = self.config.label

        # if self.config.myopic:
        #     return conf.FOLDER_OUTPUT + label + FOLDER_MYOPIC

        # if self.config.policy_random:
        #     return conf.FOLDER_OUTPUT + label + FOLDER_POLICY_RANDOM

        # if self.save_progress and self.save_progress > 0:
        #     return conf.FOLDER_OUTPUT + label + FOLDER_TRAINING
        # else:
        #     return conf.FOLDER_OUTPUT + label + FOLDER_TESTING

    @property
    def progress_path(self):
        if self.config.short_path:
            label = self.config.label_md5
        else:
            label = self.config.label

        return f"{conf.FOLDER_OUTPUT}{label}/{PROGRESS_FILENAME}"

    def create_folders(self):

        # If config is not None, then the experiments should be saved
        if self.config:
            self.output_folder_delay = self.config.output_path + FOLDER_TIME
            self.output_folder_fleet = self.config.output_path + FOLDER_FLEET
            self.output_folder_service = (
                    self.config.output_path + FOLDER_SERVICE
            )
            self.output_folder_adp_logs = self.config.output_path + ADP_LOGS
            self.folder_delay_data = self.output_folder_delay + "data/"
            self.folder_fleet_status_data = self.output_folder_fleet + "data/"
            self.folder_demand_status_data = (
                    self.output_folder_service + "data/"
            )
            # Creating folders to log MIP models
            self.config.folder_mip = self.config.output_path + "mip/"
            self.config.folder_mip_log = self.config.folder_mip + "log/"
            self.config.folder_mip_lp = self.config.folder_mip + "lp/"
            self.config.folder_adp_log = self.config.output_path + "logs/"

            folders = [
                self.output_folder_delay,
                self.folder_delay_data,
                self.output_folder_fleet,
                self.folder_fleet_status_data,
                self.output_folder_service,
                self.folder_demand_status_data,
                self.config.folder_mip_log,
                self.config.folder_mip_lp,
                self.config.folder_adp_log,
            ]

            # Creating folders
            for f in folders:
                if not os.path.exists(f):
                    os.makedirs(f)

            print(
                f"### Saving episodes at:"
                f"\n###  - {self.config.output_path}"
                f"\n### Saving plots at:"
                f"\n###  - {self.output_folder_fleet}"
                f"\n###  - {self.output_folder_service}"
            )

    def __init__(
            self,
            amod,
    ):

        self.amod = amod
        self.config = amod.config
        self.adp = amod.adp
        self.save_progress = amod.config.save_progress
        self.create_folders()

    @property
    def n(self):
        return self.adp.n

    @property
    def reward(self):
        return self.adp.reward

    @property
    def service_rate(self):
        return self.adp.service_rate

    @property
    def weights(self):
        return self.adp.weights

    def last_episode_stats(self):
        try:
            a = dict()
            for k, v in self.adp.weights.items():
                a[k] = v[-1]

            stats_str = []
            for sq, stats in self.adp.pk_delay[-1].items():
                label_values = ", ".join(
                    [
                        f"{label}={v:2.2f}"
                        if isinstance(v, float)
                        else f"{label}={v:>5}"
                        for label, v in stats.items()
                    ]
                )
                stats_str.append(f"{sq}[{label_values}]")
            sq_delay_stats = ", ".join(stats_str)
            delay_info = f"delays=({sq_delay_stats})"

            stats_str_cars = []
            for car_type, stats in self.adp.car_time[-1].items():
                label_values = ", ".join(
                    [
                        f"{label}={v:>6.2f}"
                        if isinstance(v, float)
                        else f"{label}={v:>5}"
                        for label, v in stats.items()
                    ]
                )
                stats_str_cars.append(f"{car_type}[{label_values}]")
            car_type_stats = ", ".join(stats_str_cars)

            car_time_info = f"car time status=({car_type_stats})"

            return (
                f"({self.adp.reward[-1]:15,.2f},"
                f" {self.adp.service_rate[-1]:6.2%},"
                f" {delay_info},"
                f" {car_time_info}), "
                f"Agg. level weights = {a}"
            )
        except:
            return f"(0.00, 00.00%) Agg. level weights = []"

    def compute_learning(self):

        # Reward over the course of the whole experiment
        self.plot_reward(
            file_path=self.output_path + f"r_{self.adp.n:04}",
            file_format="png",
            dpi=150,
            scale="linear",
        )

        # Service rate over the course of the whole experiment
        self.plot_service_rate(
            file_path=self.output_path + f"sl_{self.adp.n:04}",
            file_format="png",
            dpi=150,
        )

        # Service rate over the course of the whole experiment
        self.plot_weights(
            file_path=self.output_path + f"w_{self.adp.n:04}",
            file_format="png",
            dpi=150,
        )

    def plot_trip_delays(
            self, rejected, serviced, file_path=None, file_format="png", dpi=150,
    ):

        sns.set_context("talk", font_scale=1.4)

        total_trips = 0
        for sq, delays in serviced.items():
            n_serviced = len(delays)
            n_rejected = len(rejected[sq])
            total = n_rejected + n_serviced
            total_trips += total

            plt.hist(
                delays,
                label=f"{sq}(S={n_serviced:>5}, R={n_rejected:>5}) {n_serviced / total:6.2%}",
            )

        plt.title(f"{total_trips}")
        plt.xlabel("Delay (min)")

        # Configure y axis
        plt.ylabel("#Trips")
        plt.legend(
            loc="center left",
            frameon=False,
            bbox_to_anchor=(1, 0, 0.5, 1),  # (0.5, -0.15),
            ncol=1,
        )

        if file_path:
            plt.savefig(
                f"{file_path}.{file_format}", bbox_inches="tight", dpi=dpi
            )
        else:
            plt.show()
        plt.close()

    def compute_episode(
        self,
        step_log,
        it_step_trip_list,
        processing_time,
        fleet_size=None,
        plots=True,
        save_df=True,
        save_learning=True,
        save_after_iteration=1,
        save_overall_stats=True,
    ):

        # # Process trip data ######################################## #
        # Class pickup delays of SERVICED users
        trip_delays = defaultdict(list)
        # Class in-vehicle distances (km) of SERVICED users
        trip_distances = defaultdict(list)
        # Class in-vehicle distances (km) of REJECTED users
        trip_rejections = defaultdict(list)
        # Class trip count
        total_trips = defaultdict(int)
        # Origins of rejected trips (new car starting points)
        rejected_trip_origins = set()
        last_trip_origins = set()

        # Loop all trips from all steps
        for t in it.chain(*it_step_trip_list):
            total_trips[t.sq_class] += 1

            last_trip_origins.add(t.o.id)

            # If None -> Trip was rejected
            if t.pk_delay is not None:
                trip_delays[t.sq_class].append(t.pk_delay)
                trip_distances[t.sq_class].append(
                    nw.get_distance(t.o.id, t.d.id)
                )
            else:
                # Append travel distance of rejected trip
                trip_rejections[t.sq_class].append(
                    nw.get_distance(t.o.id, t.d.id)
                )
                rejected_trip_origins.add(t.o.id)

        delays_stats = dict()

        step_log.env.rejected_trip_origins = list(rejected_trip_origins)
        step_log.env.last_trip_origins = list(last_trip_origins)

        # TODO change to "for sq in user_bases"
        for sq, delays in trip_delays.items():
            delays_stats[sq] = dict(
                delay_mean=np.mean(delays),
                delay_median=np.median(delays),
                delay_total=np.sum(delays),
                serviced=len(delays),
                serviced_dist_mean=np.mean(trip_distances.get(sq, [0])),
                serviced_dist_median=np.median(trip_distances.get(sq, [0])),
                serviced_dist_total=np.sum(trip_distances.get(sq, [0])),
                rejected=len(trip_rejections.get(sq, [0])),
                rejected_dist_mean=np.mean(trip_rejections.get(sq, [0])),
                rejected_dist_median=np.median(trip_rejections.get(sq, [0])),
                rejected_dist_total=np.sum(trip_rejections.get(sq, [0])),
                sl=len(delays) / total_trips.get(sq, 0),
            )

        # TODO comment this section
        car_type_status_durations = defaultdict(lambda: defaultdict(list))
        # How much time each car have spent in each status (in minutes)?
        # dict(dict(dict()))
        # CAR TYPE -> STATUS -> TOTAL DURATION
        for c in it.chain(step_log.env.cars, step_log.env.overall_hired):
            for status, duration in c.time_status.items():
                car_type_status_durations[c.type][status].append(
                    np.sum(duration)
                )

        # Remove status level (insert it as label, such as "STATUS_total")
        # dict(dict())
        # E.g.: {"CARTYPE1":{"STATUS1_total":total_duration}}
        car_type_status_dict = dict()
        for car_type, status_durations in car_type_status_durations.items():
            car_type_status_dict[car_type] = dict()
            overall_duration = 0
            for status, durations in status_durations.items():
                # Create status
                status_label = (
                    conf.status_label_dict[status].lower().replace(" ", "_")
                )
                total = np.sum(durations)
                overall_duration += total
                # mean = np.mean(durations)
                d = {
                    f"{status_label}_total": total,
                    # f"{status_label}_mean": mean
                }
                car_type_status_dict[car_type].update(d)
            car_type_status_dict[car_type].update(
                {"total_duration": overall_duration}
            )

        self.adp.pk_delay.append(delays_stats)
        self.adp.car_time.append(car_type_status_dict)

        # Increment number of episodes
        self.adp.n += 1

        # Update reward and service rate tracks
        self.adp.reward.append(step_log.total_reward)
        self.adp.service_rate.append(step_log.service_rate)

        if self.adp.weight_track is not None:
            for car_type in Car.car_types:
                self.adp.weights[car_type].append(
                    self.adp.weight_track[car_type]
                )

        # Save intermediate plots
        if plots:

            # Fleet status (idle, recharging, rebalancing, servicing)
            # step_log.plot_fleet_status(
            #     step_log.car_statuses,
            #     file_path=self.output_folder_fleet + f"{self.adp.n:04}_total",
            #     file_format="png",
            #     dpi=150,
            # )

            if total_trips:
                self.plot_trip_delays(
                    trip_rejections,
                    trip_delays,
                    file_path=self.output_folder_delay + f"{self.adp.n:04}",
                    file_format="pdf",
                    dpi=150,
                )

            if step_log.env.config.fleet_size > 0:
                step_log.plot_fleet_status(
                    step_log.pav_statuses,
                    file_path=self.output_folder_fleet
                              + f"{self.adp.n:04}_pav",
                    **step_log.env.config.fleet_plot_config,
                )
            # step_log.plot_fleet_status_all(
            #     step_log.car_statuses, step_log.pav_statuses, step_log.fav_statuses,
            #     file_path=self.output_folder_fleet + f"{self.adp.n:04}_fav",
            #     file_format="png",
            #     dpi=150,
            # )
            if step_log.env.config.fav_fleet_size > 0:
                step_log.plot_fleet_status(
                    step_log.fav_statuses,
                    file_path=self.output_folder_fleet
                              + f"{self.adp.n:04}_fav",
                    **step_log.env.config.fleet_plot_config,
                )

            # Service status (battery level, demand, serviced demand)
            step_log.plot_service_status(
                file_path=self.output_folder_service + f"{self.adp.n:04}",
                **step_log.env.config.demand_plot_config,
            )

        if save_df:
            df_fleet = step_log.get_step_status_count()
            df_fleet.to_csv(
                self.folder_fleet_status_data
                + f"e_fleet_status_{self.adp.n:04}.csv"
            )

            df_demand = step_log.get_step_demand_status()
            df_demand.to_csv(
                self.folder_demand_status_data
                + f"e_demand_status_{self.adp.n:04}.csv"
            )

        if save_overall_stats:
            cols, df_stats = step_log.get_step_stats()

            # Add user stats
            for sq, stats in sorted(
                    self.adp.pk_delay[-1].items(), key=lambda sq_stats: sq_stats[0]
            ):
                for label, v in sorted(
                        stats.items(), key=lambda label_v: label_v[0]
                ):
                    col = f"{sq}_{label}"
                    cols.append(col)
                    df_stats[col] = v

            # Add car stats
            for car_type, stats in sorted(
                    self.adp.car_time[-1].items(),
                    key=lambda car_type_stats: car_type_stats[0],
            ):
                for label, v in sorted(
                        stats.items(), key=lambda label_v: label_v[0]
                ):
                    col = f"{car_type}_{label}"
                    cols.append(col)
                    df_stats[col] = v

            cols.append("time")
            df_stats["time"] = pd.Series([processing_time])

            # MPC optimal defines fleet sizes for each iteration based
            # on trip data
            if fleet_size is not None:
                cols.append("fleet_size")
                df_stats["fleet_size"] = pd.Series([fleet_size])

            stats_file = self.output_path + "overall_stats.csv"
            df_stats.to_csv(
                stats_file,
                mode="a",
                index=False,
                columns=cols,
                header=not os.path.exists(stats_file),
            )

        # Save what was learned so far
        if save_learning and self.adp and self.adp.n % save_learning == 0:
            # t1 = time.time()
            # adp_data = self.adp.current_data
            # np.save("dic.npy", adp_data)
            # print(time.time() - t1)

            # t2 = time.time()
            # adp_data_np = self.adp.current_data_np
            # np.save("tuple.npy", adp_data_np)
            # print(time.time() - t2)

            # t3 = time.time()
            # adp_data_np = self.adp.data
            # np.save("tuple_np.npy", dict(adp_data_np))
            # print(time.time() - t3)

            # t3 = time.time()
            # adp_data_np = self.adp.current_data_np2
            # np.save("tuple2.npy", adp_data_np)
            # print(time.time() - t3)

            # For each:
            # - Time step t,
            # - Aggregation level g,
            # - Attribute a
            # - Save (value, count) tuple
            # print("BEFORE TRANSFORMATION")
            # pprint(self.adp.values)
            # print("CURRENT DATA")
            # pprint(self.adp.current_data)
            np.save(
                self.progress_path,
                {
                    "episodes": self.adp.n,
                    "reward": self.adp.reward,
                    "service_rate": self.adp.service_rate,
                    "pk_delay": self.adp.pk_delay,
                    "car_time": self.adp.car_time,
                    "progress": self.adp.current_data,
                    "weights": self.adp.weights,
                },
            )

    def load_progress(self):
        """Load episodes learned so farD

        Returns:
            values, counts -- Value functions and count per aggregation
                level.
        """

        (
            self.adp.n,
            self.adp.reward,
            self.adp.service_rate,
            self.adp.weights,
        ) = self.adp.read_progress(self.progress_path)

        # print("After reading")
        # pprint(self.adp.values)

    def plot_weights(
            self, file_path=None, file_format="png", dpi=150, scale="linear"
    ):

        sns.set_context("paper")

        def plot_series(weights, car_type="AV"):
            series_list = [list(a) for a in zip(*weights)]

            for series in series_list:
                plt.plot(np.arange(self.adp.n), series)

            plt.xlabel("Episodes")
            plt.xscale(scale)
            plt.ylabel("Weights")
            plt.legend([f"Level {g}" for g in range(len(series_list))])

            # Ticks
            # plt.yticks(np.arange(1, step=0.05))
            # plt.xticks(np.arange(self.adp.n))

            if file_path:
                plt.savefig(
                    f"{file_path}_{car_type}.{file_format}",
                    bbox_inches="tight",
                    dpi=dpi,
                )
            else:
                plt.show()

        # print("# Weights")
        # pprint(self.adp.weights)

        for car_type, weights in self.adp.weights.items():
            plot_series(weights, car_type=car_type)

        plt.close()

    def plot_reward(
            self, file_path=None, file_format="png", dpi=150, scale="linear"
    ):
        sns.set_context("paper")
        plt.plot(np.arange(self.adp.n), self.adp.reward, color="r")
        plt.xlabel("Episodes")
        plt.xscale(scale)
        plt.ylabel("Reward")

        if file_path:
            plt.savefig(
                f"{file_path}.{file_format}", bbox_inches="tight", dpi=dpi
            )
        else:
            plt.show()

        plt.close()

    def plot_service_rate(self, file_path=None, file_format="png", dpi=150):
        sns.set_context("paper")
        plt.plot(np.arange(self.adp.n), self.adp.service_rate, color="b")
        plt.xlabel("Episodes")
        plt.ylabel("Service rate (%)")

        if file_path:
            plt.savefig(
                f"{file_path}.{file_format}", bbox_inches="tight", dpi=dpi
            )
        else:
            plt.show()

        plt.close()


class StepLog:
    def __init__(self, env):
        self.env = env
        self.reward_list = list()
        self.sq_class_count = list()
        self.serviced_list = list()
        self.outstanding_list = list()
        self.rejected_list = list()
        self.total_list = list()
        self.all_trips = set()
        self.car_statuses = defaultdict(list)
        self.pav_statuses = defaultdict(list)
        self.fav_statuses = defaultdict(list)
        self.total_battery = list()
        self.n = 0

    def compute_fleet_status(self):
        """Save the status of each car in step (fleet snapshot)"""

        # Get number of cars per status in a time step
        # and aggregate battery level
        (
            dict_status,
            pav_status,
            fav_status,
            battery_level,
        ) = self.env.get_fleet_status()

        # Fleet aggregate battery level
        self.total_battery.append(battery_level)

        # Number of vehicles per status
        for k in Car.status_list:
            self.car_statuses[k].append(dict_status.get(k, 0))
            self.pav_statuses[k].append(pav_status.get(k, 0))
            self.fav_statuses[k].append(fav_status.get(k, 0))

    def add_record(self, step):

        # Fleet step happens after trip step
        self.n += 1
        self.reward_list.append(step.revenue)
        self.serviced_list.append(len(step.serviced))
        self.outstanding_list.append(len(step.outstanding))
        self.rejected_list.append(len(step.rejected))
        total = len(step.serviced) + len(step.rejected)
        self.total_list.append(total)
        self.sq_class_count.append(
            {
                sq_class: count
                for sq_class, count in zip(
                *np.unique([t.sq_class for t in step.trips], return_counts=True)
            )
            }
        )

    @property
    def total_reward(self):
        return sum(self.reward_list)

    @property
    def serviced(self):
        return sum(self.serviced_list)

    @property
    def rejected(self):
        return sum(self.rejected_list)

    @property
    def outstanding_last(self):
        try:
            return self.outstanding_list[-1]
        except:
            return 0

    @property
    def total(self):
        return sum(self.total_list)

    @property
    def service_rate(self):
        s = self.serviced
        t = self.total
        try:
            return s / t
        except ZeroDivisionError:
            return 0

    def info(self):
        """Print last time step statistics
        """

        try:
            sr = self.serviced_list[-1] / self.total_list[-1]
            reward = self.reward_list[-1]
            total = self.total_list[-1]
            sq_class_count = self.sq_class_count[-1]
        except:
            sr = 0
            reward = 0
            total = 0
            sq_class_count = {}

        status, status_pav, status_fav, battery = self.env.get_fleet_status()
        statuses = ", ".join(
            [
                f"{Car.status_label_dict[status_code]}: {status_count:>4}"
                for status_code, status_count in status.items()
            ]
        )

        pav_statuses = (
            f" | PAV: "
            + (
                ", ".join(
                    [
                        f"{Car.status_label_dict[status_code]}: {status_count:>4}"
                        for status_code, status_count in status_pav.items()
                    ]
                )
            )
            if self.env.hired_cars
            else ""
        )

        fav_statuses = (
            f" | FAV: "
            + (
                ", ".join(
                    [
                        f"{Car.status_label_dict[status_code]}: {status_count:>4}"
                        for status_code, status_count in status_fav.items()
                    ]
                )
            )
            if self.env.hired_cars
            else ""
        )

        all_statuses = (

            ", ".join(
                [
                    (
                        f"{Car.status_label_dict[status_code]}= "
                        f"{status_pav[status_code]:>4}/{status_fav[status_code]:>4}/{status[status_code]:>4}"
                    )
                    for status_code in status.keys()
                ]
            )
        )

        try:
            sq_classes, sq_counts = list(zip(*sq_class_count.items()))

        except:
            sq_classes, sq_counts = [], []

        sq_classes = ",".join(c for c in sq_classes)
        sq_counts = ",".join(f"{c / sum(sq_counts):3.2f}" for c in sq_counts)

        # Car neighborhood info
        n_neigh_cars_list, avg_reb_delay_list = self.env.car_neigh_stats()
        s_mean, s_max, s_min = (
            np.mean(n_neigh_cars_list),
            np.max(n_neigh_cars_list),
            np.min(n_neigh_cars_list),
        )
        reb_delay_mean, reb_delay_max, reb_delay_min = (
            np.mean(avg_reb_delay_list),
            np.max(avg_reb_delay_list),
            np.min(avg_reb_delay_list),
        )

        sq_info = f"({sq_classes})=({sq_counts})"
        summary = (
            f"#{self.n:>4}"
            f"  ###  cost= {self.total_reward:>10.2f}"
            f"  ###  trips={total:<4}"
            f" ({sr:>7.2%})"
            f" {sq_info:^19}"
            f" - [s]{self.serviced:>4} + [r]{self.rejected:>4} + [o]{self.outstanding_last:>4} = {self.total:>4} - "
            # f"  ###  {statuses}{pav_statuses}{fav_statuses}"
            f"  ### PAV/FAV/TOTAL: {all_statuses}"
            f"  ### Car neighbors (mean, max, min): ({s_mean:>6.2f}, {s_max}, {s_min})"
            f"  ### Reb. delay (mean, max, min): ({reb_delay_mean:<6.2f}, {reb_delay_max:<6.2f}, {reb_delay_min:<6.2f})"
        )
        return summary

    def overall_log(self, label="Operational"):
        """Show service rate, recharge count, and total profit.
        
        Keyword Arguments:
            label {str} -- Experiment lable (default: {"Operational"})
        """

        # Get number of times recharging for each vehicle
        recharge_list = []
        for c in self.env.cars:
            recharge_list.append(c.recharge_count)

        s = sum(self.serviced_list)
        t = sum(self.total_list)

        print(
            f"### {label} performance ##########################\n"
            f"Service rate: {s}/{t} ({self.service_rate:.2%})\n"
            f"Fleet recharge count: {sum(recharge_list)}\n"
            f"        Total profit: {self.total_reward:.2f}"
        )

    def plot_fleet_status(
            self,
            car_statuses,
            file_path=None,
            file_format="png",
            dpi=150,
            earliest_hour=0,
            omit_cruising=True,
            show_legend=True,
            linewidth=2,
            lenght_tick=6,
            xticks_labels=[
                "",
                "5AM",
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
            ],
            x_min=0,
            x_max=330,
            x_num=12,
            sns_context="talk",
            sns_font_scale=1.4,
            fig_x_inches=10,
            fig_y_inches=10,
    ):

        sns.set_context(sns_context, font_scale=sns_font_scale)

        if omit_cruising:
            car_statuses[Car.SERVICING] = np.array(
                car_statuses[Car.CRUISING]
            ) + np.array(car_statuses[Car.ASSIGN])
            del car_statuses[Car.CRUISING]
            del car_statuses[Car.ASSIGN]

        xticks = np.linspace(x_min, x_max, x_num)

        for status_code, status_count_step in car_statuses.items():
            status_label = Car.status_label_dict[status_code]
            plt.plot(
                status_count_step,
                linewidth=linewidth,
                label=status_label,
                color=self.env.config.color_fleet_status[status_code],
            )

        matrix_status_count = np.array(list(car_statuses.values()))
        total_count = np.sum(matrix_status_count, axis=0)
        plt.plot(
            total_count, linewidth=linewidth, color="magenta", label="Total",
        )

        termination = self.n - self.env.config.offset_termination_steps
        repositioning = self.env.config.offset_repositioning_steps

        # Demarcate offsets
        plt.axvline(
            repositioning, color="k", linewidth=linewidth, linestyle="--"
        )
        plt.axvline(
            termination, color="k", linewidth=linewidth, linestyle="--"
        )

        # TODO automatic xticks
        fig = plt.gcf()
        fig.set_size_inches(fig_x_inches, fig_y_inches)

        plt.xticks(xticks, xticks_labels)
        plt.tick_params(labelsize=25)

        plt.tick_params(
            direction="out",
            # length=lenght_tick,
            width=linewidth,
            colors="k",
            grid_color="k",
            grid_alpha=0.5,
        )

        # x_ticks = 6
        # x_stride = 60*3
        # max_x = np.math.ceil(self.n / x_stride) * x_stride
        # xticks = np.arange(0, max_x + x_stride, x_stride)
        # plt.xticks(xticks, [f'{tick//60}h' for tick in xticks])

        # Configure x axis
        plt.xlim([0, self.env.config.time_steps])
        plt.ylim(
            [
                0,
                max(self.env.config.fleet_size, self.env.config.fav_fleet_size)
                + 10,
            ]
        )

        # plt.ylim([0, self.env.config.fleet_size])
        # plt.xlabel(f"Steps ({self.env.config.time_increment} min)")

        plt.xlabel("Time")

        # Configure y axis
        plt.ylabel("# Vehicles")
        ax = plt.gca()
        box = ax.get_position()
        ax.set_position(
            [box.x0, box.y0 + box.height * 0.1, box.width, box.height * 0.9]
        )

        plt.legend(
            loc="center left",
            frameon=False,
            bbox_to_anchor=(1, 0, 0.5, 1),  # (0.5, -0.15),
            ncol=1,
        )

        if not show_legend:
            ax.get_legend().remove()

        if file_path:
            plt.savefig(
                f"{file_path}.{file_format}", bbox_inches="tight", dpi=dpi
            )
        else:
            plt.show()
        plt.close()

    def get_step_status_count(self):

        self.step_df = pd.DataFrame()
        for status_code, status_count_step in self.car_statuses.items():
            self.step_df[Car.status_label_dict[status_code]] = pd.Series(
                status_count_step
            )

        self.step_df.index.name = "step"

        return self.step_df

    def get_step_demand_status(self):

        self.step_demand_status = pd.DataFrame()
        self.step_demand_status["Total demand"] = pd.Series(self.total_list)
        self.step_demand_status["Serviced demand"] = pd.Series(
            self.serviced_list
        )
        self.step_demand_status["Battery level"] = pd.Series(
            self.total_battery
        )
        self.step_demand_status["Reward"] = pd.Series(self.total_reward)
        self.step_demand_status.index.name = "step"

        return self.step_demand_status

    def get_step_stats(self):
        columns = ["Episode", "Service rate", "Total reward"]
        self.step_stats = pd.DataFrame()
        self.step_stats["Episode"] = pd.Series([self.env.adp.n])
        self.step_stats["Service rate"] = pd.Series(
            [self.env.adp.service_rate[-1]]
        )
        self.step_stats["Total reward"] = pd.Series([self.env.adp.reward[-1]])

        for car_type, weights in sorted(
                self.env.adp.weights.items(),
                key=lambda car_type_weights: car_type_weights[0],
        ):
            for i, w in enumerate(weights[-1]):
                col = f"{car_type}_{i:02}"
                columns.append(col)
                self.step_stats[col] = pd.Series([w])

        return columns, self.step_stats

    def plot_service_status(
            self,
            file_path=None,
            file_format="png",
            dpi=150,
            show_legend=True,
            linewidth=2,
            lenght_tick=6,
            xticks_labels=[
                "",
                "5AM",
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
            ],
            x_min=0,
            x_max=330,
            y_min=0,
            y_num=500,
            y_max=4000,
            x_num=12,
            sns_context="talk",
            sns_font_scale=1.4,
            fig_x_inches=10,
            fig_y_inches=10,
    ):

        sns.set_context("talk", font_scale=1.4)

        termination = self.n - self.env.config.offset_termination_steps
        repositioning = self.env.config.offset_repositioning_steps

        xticks = np.linspace(x_min, x_max, x_num)

        fig, ax1 = plt.subplots()
        ax1.set_xlabel("Time")
        ax1.set_ylabel("Cumulative n. of requests")
        # Sum zero to account for the starting step where there are no
        # customers.
        ax1.plot(
            np.cumsum([0] + self.total_list),
            linewidth=linewidth,
            label="Total",
            color="magenta",
        )
        ax1.plot(
            np.cumsum([0] + self.serviced_list),
            linewidth=linewidth,
            label="Serviced",
            color="k",
        )
        ax1.legend()

        # Plot battery
        try:
            max_battery_level = len(self.env.cars) * (
                    self.env.cars[0].battery_level_miles_max
                    * self.env.config.battery_size_kwh_distance
            )

            # Closest power of 10
            max_battery_level_10 = 10 ** round(
                np.math.log10(max_battery_level)
            )

            list_battery_level_kwh = (
                    np.array(self.total_battery)
                    * self.env.config.battery_size_kwh_distance
            )

            ax2 = ax1.twinx()
            ax2.plot(list_battery_level_kwh, label="Battery Level", color="r")
            ax2.set_ylabel("Total Battery Level (KWh)")

            # Configure ticks y axis (battery level)
            y_ticks = 5  # apart from 0
            y_stride = max_battery_level_10 / y_ticks
            yticks = np.arange(0, max_battery_level_10 + y_stride, y_stride)
            plt.yticks(yticks)

            plt.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))

            # Put a legend below current axis
            ax2.legend(
                loc="upper center",
                frameon=False,
                bbox_to_anchor=(0.8, -0.15),
                ncol=1,
            )
        except:
            pass

        # Demarcate offsets
        plt.axvline(
            repositioning, color="k", linewidth=linewidth, linestyle="--"
        )
        plt.axvline(
            termination, color="k", linewidth=linewidth, linestyle="--"
        )

        fig = plt.gcf()
        fig.set_size_inches(fig_x_inches, fig_y_inches)

        plt.xticks(xticks, xticks_labels)

        plt.tick_params(labelsize=25)

        plt.tick_params(
            direction="out",
            # length=lenght_tick,
            width=linewidth,
            colors="k",
            grid_color="k",
            grid_alpha=0.5,
        )

        # x_ticks = 6
        # x_stride = 60*3
        # max_x = np.math.ceil(self.n / x_stride) * x_stride
        # xticks = np.arange(0, max_x + x_stride, x_stride)
        # plt.xticks(xticks, [f'{tick//60}h' for tick in xticks])

        # Configure x axis
        plt.xlim([x_min, x_max])
        # Configure ticks y axis (battery level)
        yticks = np.linspace(y_min, y_max, y_num)
        yticks_labels = [f"{int(y):,}" for y in yticks]
        # print("YTICKS:", yticks, yticks_labels)
        plt.yticks(yticks, yticks_labels)
        plt.ylim([y_min, y_max])

        ax = plt.gca()
        box = ax.get_position()
        ax.set_position(
            [box.x0, box.y0 + box.height * 0.1, box.width, box.height * 0.9]
        )

        # Shrink current axis's height by 10% on the bottom
        box = ax1.get_position()
        ax1.set_position(
            [box.x0, box.y0 + box.height * 0.1, box.width, box.height * 0.9]
        )

        # Put a legend below current axis
        ax1.legend(
            loc="upper center",
            frameon=False,
            bbox_to_anchor=(0.3, -0.15),
            ncol=2,
        )

        if not show_legend:
            ax.get_legend().remove()

        if file_path:
            plt.savefig(
                f"{file_path}.{file_format}", bbox_inches="tight", dpi=dpi
            )
        else:
            plt.show()
        plt.close()


# BOKEH ################################################################
def compute_trips(trips):
    """Return dictionary of trip coordinates indexed by origin and
    destination point ids.
    
    Parameters
    ----------
    trips : List of Trip objects
        Trips placed in a certain time step
    
    Returns
    -------
    dict of coordinates
        Dictionary of trip od points and correspondent coordinates.
    
    Example
    -------
    >>> o = Point(40.4, 72.4, 1) # origin point
    >>> d = Point(40.5, 71.3, 2) # destination point
    >>> t = 5 # current time
    >>> trips = [Trip(o, d, 5)] # list of trips
    >>> compute_trips(trips)
    >>> {"o":{"x":[40.4], "y":[72.4]}, "d":{"x":[40.5], "y":[71.3]}}
    """

    xy_trips = defaultdict(lambda: defaultdict(list))
    xy_trips["o"]["x"] = []
    xy_trips["o"]["y"] = []

    # Origin, destination coordinates
    for t in trips:
        xy_trips["o"]["x"].append(t.o.x)
        xy_trips["o"]["y"].append(t.o.y)

        xy_trips["d"]["x"].append(t.d.x)
        xy_trips["d"]["y"].append(t.d.y)

    return xy_trips


def plot_centers(
        list_center_level,
        levels,
        level_demand,
        level_fleet,
        show_sp_lines=True,
        show_lines=True,
):
    print("-------- Plotting region centers --------")

    list_center_level = defaultdict(lambda: defaultdict(list))
    set_center_ids_level = defaultdict(set)

    # Lines connecting region centers to nodes
    center_lines_xy = defaultdict(
        lambda: {"xs": [], "ys": [], "o": [], "d": [], "level": []}
    )
    center_lines_sp_xy = defaultdict(lambda: {"xs": [], "ys": []})

    for p in list_center_level:
        # print("Ids dic:", p.level_ids_dic)
        list_center_level[0]["x"] += [p.x]
        list_center_level[0]["y"] += [p.y]

        # print(f"{i_p} - ids: {p.level_ids_dic}")

        # Ids and levels (i.e., 30, 60, 90)
        for i, level in enumerate(levels):
            # print("\n\n\n###################### LEVEL", i, "-", level, "##################")
            if i == 0:
                continue

            center_point = list_center_level[p.id_level(i)]

            if show_sp_lines and center_point.id != p.id:
                xy_sp = nw.query_sp(center_point, p, projection="MERCATOR")

                # Get list of x and y coordinates (e.g., [x1,x2,x3] and
                # [y1,y2,y3])
                sp_x, sp_y = zip(*xy_sp)

                sp_x = list(sp_x)
                sp_x_pair = [list(pairx) for pairx in zip(sp_x[:-1], sp_x[1:])]

                sp_y = list(sp_y)
                sp_y_pair = [list(pairy) for pairy in zip(sp_y[:-1], sp_y[1:])]

                center_lines_sp_xy[level]["xs"].extend(sp_x_pair)
                center_lines_sp_xy[level]["ys"].extend(sp_y_pair)

            if show_lines and center_point.id != p.id:
                # print(
                #     center_point, p, nw.get_distance(center_point, p)
                # )
                center_lines_xy[level]["xs"].append([center_point.x, p.x])
                center_lines_xy[level]["ys"].append([center_point.y, p.y])

                # ("index", "$index"),
                # ("(x,y)", "($x, $y)"),
                # ("Origin", "@o"),
                # ("Destination", "@d"),
                # ("Level", "@level"),

            if center_point.id not in set_center_ids_level[level]:
                list_center_level[level]["x"].append(center_point.x)
                list_center_level[level]["y"].append(center_point.y)
                set_center_ids_level[level].add(center_point.id)

    return list_center_level, center_lines_xy, center_lines_sp_xy


STRAIGHT_LINE = "OD"
SP_LINE = "SP"


def get_center_elements(points, levels, direct_lines=True, sp_lines=False):
    print("-------- Plotting region centers --------")

    # X,Y coordinates of points at each level
    centers = defaultdict(lambda: defaultdict(list))

    # Unique center ids per level
    center_ids = defaultdict(set)

    # Lines connecting region centers to nodes
    lines = defaultdict(lambda: defaultdict(lambda: {"xs": [], "ys": []}))

    for p in points:

        # Ids and levels (i.e., 30, 60, 90)
        for i, level in enumerate(levels):

            # Get center considering level
            center_point = points[p.id_level(i)]

            # Add points (level 0 contains all nodes)
            if center_point.id not in center_ids[level]:
                centers[level]["x"].append(center_point.x)
                centers[level]["y"].append(center_point.y)

                # Guarantees centers are not included twice
                center_ids[level].add(center_point.id)

            # ######################################################## #
            # Adding outbound lines from centers ##################### #
            # ######################################################## #

            # Filter o == d
            if center_point.id != p.id:

                # If shortest paths from centers
                if sp_lines:
                    xy_sp = nw.query_sp(center_point, p, projection="MERCATOR")

                    # Get list of x and y coordinates (e.g., [x1,x2,x3] and
                    # [y1,y2,y3])
                    sp_x, sp_y = zip(*xy_sp)

                    sp_x = list(sp_x)
                    sp_x_pair = [
                        list(pairx) for pairx in zip(sp_x[:-1], sp_x[1:])
                    ]

                    sp_y = list(sp_y)
                    sp_y_pair = [
                        list(pairy) for pairy in zip(sp_y[:-1], sp_y[1:])
                    ]

                    lines[SP_LINE][level]["xs"].extend(sp_x_pair)
                    lines[SP_LINE][level]["ys"].extend(sp_y_pair)

                # If straight lines from centers
                if direct_lines:
                    lines[STRAIGHT_LINE][level]["xs"].append(
                        [center_point.x, p.x]
                    )
                    lines[STRAIGHT_LINE][level]["ys"].append(
                        [center_point.y, p.y]
                    )

    return centers, lines


def compute_movements(step, cars, step_car_path, n_points=30):
    # Get car paths
    for c in cars:

        # Car path was stored in previous step since its route covers
        # more than one time step
        if c.id not in step_car_path[step]:

            # segmented_sp = nw.query_segmented_sp(
            #     c.previous,
            #     c.point,
            #     n_points,
            #     step_duration,
            #     projection="MERCATOR",
            #     waypoint=c.waypoint,
            # )

            dif = c.arrival_time - c.previous_arrival

            # If vehice is parked
            if dif == 0:
                segmented_sp = [[[c.point.x, c.point.y]]]
            # Vehicle is moving
            else:
                segmented_sp = nw.query_sp_sliced(
                    c.previous,
                    c.point,
                    n_points * dif,
                    dif,
                    projection="MERCATOR",
                    waypoint=c.waypoint,
                )

            # if segmented_sp[0]:
            #     print(f'{c.id} - INSERT TW: ({c.previous_arrival},{c.arrival_time}) Segmented SP: {len(segmented_sp)}')
            #     print("S:", c.previous, c.point)
            # Update car movement in step
            for i, s in enumerate(segmented_sp):
                step_car_path[step + i][c.id] = (c.status, s)

        # else:
        #     print(
        #         # f"Segmented: {[len(s) for status, s in movement_step_fleet_dict[step]]}."
        #         f"\n################ {c} ##############################"
        #         f"\n-           Status: {c.status} "
        #         f"\n- Previous arrival: {c.previous_arrival} "
        #         f"\n-          Arrival: {c.arrival_time} "
        #         f"\n-             Step: {c.step}/{step} "
        #     )


def get_next_frame(step_car_path, step):
    if step in step_car_path and step_car_path[step]:

        xy_status = defaultdict(lambda: dict(x=[], y=[]))
        count_finished = 0

        for status, path_car in step_car_path[step].values():

            if len(path_car) > 1:
                x, y = path_car.pop(0)
                xy_status[status]["x"].append(x)
                xy_status[status]["y"].append(y)

            elif len(path_car) == 1:
                x, y = path_car[0]
                count_finished += 1
                xy_status[status]["x"].append(x)
                xy_status[status]["y"].append(y)

            else:

                print("WHAT", status, path_car)
                # pass

                pprint(step_car_path)

        if count_finished == len(step_car_path[step].keys()):
            return xy_status, step + 1
        else:
            return xy_status, step
