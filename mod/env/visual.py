import os
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
from pprint import pprint
import pandas as pd
import itertools as it

import mod.env.config as conf
from mod.env.car import Car
from mod.env.network import Point
import mod.env.network as nw
import seaborn as sns

sns.set(style="ticks")
sns.set_context("paper")
np.set_printoptions(precision=3)


class EpisodeLog:


    def get_od_lists(self, amod):

        # ---------------------------------------------------------------- #
        # Trips ########################################################## #
        # ---------------------------------------------------------------- #

        try:
            o_ids, d_ids = self.load_ods()
            origins = [amod.points[o] for o in o_ids]
            destinations = [amod.points[d] for d in d_ids]
            print(
                f"Loading {len(origins)} origins and "
                f"{len(destinations)} destinations."
            )

        except Exception as e:

            print(f"Error!{e}")

            # Create random centers from where trips come from
            # TODO choose level to query origins
            origins = nw.query_centers(
                amod.points,
                amod.config.origin_centers,
                amod.config.demand_center_level,
            )

            destinations = nw.query_centers(
                amod.points,
                amod.config.destination_centers,
                amod.config.demand_center_level,
            )

            print(
                f"\nSaving {len(origins)} origins and "
                f"{len(destinations)} destinations."
            )
            self.save_ods(
                [o.id for o in origins], [d.id for d in destinations]
            )
        return origins, destinations

    def create_folders(self):
        # If config is not None, then the experiments should be saved
        if self.config:
            self.output_path = conf.FOLDER_OUTPUT + self.config.label
            self.output_folder_fleet = self.output_path + "/fleet/"
            self.output_folder_service = self.output_path + "/service/"
            self.output_folder_adp_logs = self.output_path + "/adp_logs/"
            self.folder_fleet_status_data = self.output_folder_fleet + "data/"
            self.folder_demand_status_data = self.output_folder_service + "data/"

            # Creating folders to log episodes
            if not os.path.exists(self.output_folder_fleet):
                os.makedirs(self.output_folder_fleet)
                os.makedirs(self.folder_fleet_status_data)

            if not os.path.exists(self.output_folder_service):
                os.makedirs(self.output_folder_service)
                os.makedirs(self.folder_demand_status_data)

            if not os.path.exists(self.config.folder_mip):
                os.makedirs(self.config.folder_mip_log)
                os.makedirs(self.config.folder_mip_lp)

            if not os.path.exists(self.config.folder_adp_log):
                os.makedirs(self.config.folder_adp_log)

                print(
                    f"\n### Saving episodes at:"
                    f"\n - {self.output_path}"
                    f"\n### Saving plots at:"
                    f"\n - {self.output_folder_fleet}"
                    f"\n - {self.output_folder_service}"
                )

    def __init__(self, config=None, n=0, reward=list(), service_rate=list(), weights=list(), adp=None):
        self.config = config
        self.adp = adp
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

    def save_ods(self, origin_ids, destination_ids):
        """Save trip ods in .npy file. When method is restarted, same
        ods can be used to guarantee consistency.

        Parameters
        ----------
        origin_ids : list of ints
            Origin ids
        destination_ids : list of ints
            Destination ids
        """
        ods = {"origin": origin_ids, "destination": destination_ids}
        np.save(self.output_path + "/trip_od_ids.npy", ods)

    def load_ods(self):
        try:
            path_od_ids = self.output_path + "/trip_od_ids.npy"
            ods = np.load(path_od_ids, allow_pickle=True).item()
            return ods["origin"], ods["destination"]

        except Exception as e:
            print(f'Origins at "{path_od_ids}" could not be find {e}.')
            raise Exception

    def last_episode_stats(self):
        try:
            a = dict()
            for k, v in self.adp.weights.items():
                a[k] = v[-1]
            return (
                f"({self.adp.reward[-1]:15,.2f},"
                f" {self.adp.service_rate[-1]:6.2%}) "
                f"Agg. level weights = {a}"
            )
        except:
            return f"(0.00, 00.00%) Agg. level weights = []"

    def compute_learning(self):

        # Reward over the course of the whole experiment
        self.plot_reward(
            file_path=self.output_path + f"/reward_{self.adp.n:04}_episodes",
            file_format="png",
            dpi=150,
            scale="linear",
        )

        # Service rate over the course of the whole experiment
        self.plot_service_rate(
            file_path=self.output_path + f"/service_rate_{self.adp.n:04}_episodes",
            file_format="png",
            dpi=150,
        )

        # Service rate over the course of the whole experiment
        self.plot_weights(
            file_path=self.output_path + f"/weights_{self.adp.n:04}_episodes",
            file_format="png",
            dpi=150,
        )

    def compute_episode(
        self,
        step_log,
        processing_time,
        plots=True,
        save_df=True,
        save_learning=True,
        save_overall_stats=True
    ):

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
            step_log.plot_fleet_status(
                file_path=self.output_folder_fleet + f"{self.adp.n:04}",
                file_format="png",
                dpi=150,
            )

            # Service status (battery level, demand, serviced demand)
            step_log.plot_service_status(
                file_path=self.output_folder_service + f"{self.adp.n:04}",
                file_format="png",
                dpi=150,
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
            df_stats = step_log.get_step_stats()
            df_stats["time"] = pd.Series([processing_time])
            stats_file = self.output_path + "/overall_stats.csv"
            df_stats.to_csv(
                stats_file,
                mode="a",
                index=False,
                header=not os.path.exists(stats_file)
            )

        # Save what was learned so far
        if save_learning and self.adp:

            path = self.output_path + "/progress.npy"

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
                path,
                {
                    "episodes": self.adp.n,
                    "reward": self.adp.reward,
                    "service_rate": self.adp.service_rate,
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

        path = self.output_path + "/progress.npy"

        (
            self.adp.n,
            self.adp.reward,
            self.adp.service_rate,
            self.adp.weights
        ) = self.adp.read_progress(path)
        
        # print("After reading")
        # pprint(self.adp.values)

    def plot_weights(
        self, file_path=None, file_format="png", dpi=150, scale="linear"
    ):
        def plot_series(weights, car_type="AV"):
            series_list = [list(a) for a in zip(*weights)]

            for series in series_list:
                plt.plot(np.arange(self.adp.n), series)

            plt.xlabel("Episodes")
            plt.xscale(scale)
            plt.ylabel("Weights")
            plt.legend([f"Level {g}" for g in range(len(series_list))])

            # Ticks
            plt.yticks(np.arange(1, step=0.05))
            plt.xticks(np.arange(self.adp.n))

            if file_path:
                plt.savefig(f"{file_path}_{car_type}.{file_format}", dpi=dpi)
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

        plt.plot(np.arange(self.adp.n), self.adp.reward, color="r")
        plt.xlabel("Episodes")
        plt.xscale(scale)
        plt.ylabel("Reward")

        if file_path:
            plt.savefig(f"{file_path}.{file_format}", dpi=dpi)
        else:
            plt.show()

        plt.close()

    def plot_service_rate(self, file_path=None, file_format="png", dpi=150):

        plt.plot(np.arange(self.adp.n), self.adp.service_rate, color="b")
        plt.xlabel("Episodes")
        plt.ylabel("Service rate (%)")

        if file_path:
            plt.savefig(f"{file_path}.{file_format}", dpi=dpi)
        else:
            plt.show()

        plt.close()


class StepLog:
    def __init__(self, env):
        self.env = env
        self.reward_list = list()
        self.serviced_list = list()
        self.rejected_list = list()
        self.total_list = list()
        self.car_statuses = defaultdict(list)
        self.total_battery = list()
        self.n = 0

    def compute_fleet_status(self, step):
        """Save the status of each car in step (fleet snapshot)"""
        
        # Fleet step happens after trip step
        self.n = step

        # Get number of cars per status in a time step
        # and aggregate battery level
        dict_status, battery_level = self.env.get_fleet_status()

        # Fleet aggregate battery level
        self.total_battery.append(battery_level)

        # Number of vehicles per status
        for k in Car.status_list:
            self.car_statuses[k].append(dict_status.get(k, 0))

    def add_record(self, reward, serviced, rejected):
        self.reward_list.append(reward)
        self.serviced_list.append(len(serviced))
        self.rejected_list.append(len(rejected))
        total = len(serviced) + len(rejected)
        self.total_list.append(total)


    @property
    def total_reward(self):
        return sum(self.reward_list)

    @property
    def service_rate(self):
        s = sum(self.serviced_list)
        t = sum(self.total_list)
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
        except:
            sr = 0
            reward = 0
            total = 0

        status, battery = self.env.get_fleet_status()
        return (
            f"### Time step (trip={self.n:>4}, fleet={self.n+1:>4})"
            f" ### Profit: {self.total_reward:>10.2f}"
            f" ### Service level: {sr:>7.2%}"
            f" ### Trips: {total:>4}"
            f" ### Status: {dict(status)}"
        )

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

    def plot_fleet_status(self, file_path=None, file_format="png", dpi=150, earliest_hour=0):
        steps = np.arange(self.n+1)

        for status_label, status_count_step in self.car_statuses.items():
            plt.plot(
                steps,
                status_count_step,
                label=status_label,
                color=self.env.config.color_fleet_status[status_label]
            )

        matrix_status_count = np.array(list(self.car_statuses.values()))
        total_count = np.sum(matrix_status_count, axis=0)
        plt.plot(steps, total_count, color="#000000", label="Total")

        

        
        # x_ticks = 6
        # x_stride = 60*3
        # max_x = np.math.ceil(self.n / x_stride) * x_stride
        # xticks = np.arange(0, max_x + x_stride, x_stride)
        # plt.xticks(xticks, [f'{tick//60}h' for tick in xticks])

        # Configure x axis
        plt.xlim([0, self.env.config.time_steps])
        plt.ylim([0, self.env.config.fleet_size])
        plt.xlabel(f"Steps ({self.env.config.time_increment} min)")
        
        # Configure y axis
        plt.ylabel("# Cars")
        ax = plt.gca()
        box = ax.get_position()
        ax.set_position(
            [box.x0, box.y0 + box.height * 0.1, box.width, box.height * 0.9]
        )

        plt.legend(
            loc="upper center",
            frameon=False,
            bbox_to_anchor=(0.5, -0.15),
            ncol=5,
        )

        if file_path:
            plt.savefig(f"{file_path}.{file_format}", dpi=dpi)
        else:
            plt.show()
        plt.close()

    def get_step_status_count(self):

        self.step_df = pd.DataFrame()
        for status_label, status_count_step in self.car_statuses.items():
            self.step_df[status_label] = pd.Series(status_count_step)

        self.step_df.index.name = "step"

        return self.step_df

    def get_step_demand_status(self):

        self.step_demand_status = pd.DataFrame()
        self.step_demand_status["Total demand"] = pd.Series(self.total_list)
        self.step_demand_status["Serviced demand"] = pd.Series(self.serviced_list)
        self.step_demand_status["Battery level"] = pd.Series(self.total_battery)
        self.step_demand_status["Reward"] = pd.Series(self.total_reward)
        self.step_demand_status.index.name = "step"

        return self.step_demand_status


    def get_step_stats(self):

        self.step_stats = pd.DataFrame()
        self.step_stats["Episode"] = pd.Series([self.env.adp.n])
        self.step_stats["Service rate"] = pd.Series([self.env.adp.service_rate[-1]])
        self.step_stats["Total reward"] = pd.Series([self.env.adp.reward[-1]])
        for car_type, weights in self.env.adp.weights.items():
            for i, w in enumerate(weights[-1]):
                self.step_stats[f"{car_type}_{i:02}"] = pd.Series([w])

        return self.step_stats


    def plot_service_status(self, file_path=None, file_format="png", dpi=150):

        steps = np.arange(self.n+1)

        fig, ax1 = plt.subplots()
        ax1.set_xlabel("Time")
        ax1.set_ylabel("Trips")
        ax1.plot(steps, self.total_list, label="Total demand", color="b")
        ax1.plot(steps, self.serviced_list, label="Met demand", color="g")
        ax1.legend()

        try:
            max_battery_level = len(self.env.cars) * (
                self.env.cars[0].battery_level_miles_max
                * self.env.config.battery_size_kwh_distance
            )

            # Closest power of 10
            max_battery_level_10 = 10 ** round(np.math.log10(max_battery_level))

            list_battery_level_kwh = (
                np.array(self.total_battery)
                * self.env.config.battery_size_kwh_distance
            )

            ax2 = ax1.twinx()
            ax2.plot(
                steps, list_battery_level_kwh, label="Battery Level", color="r"
            )
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

        # Configure ticks x axis
        # x_ticks = 6
        # x_stride = 60*3
        # max_x = np.math.ceil(self.n / x_stride) * x_stride
        # xticks = np.arange(0, max_x + x_stride, x_stride)
        # plt.xticks(xticks, [f'{tick//60}h' for tick in xticks])

         # Configure x axis
        plt.xlim([0, self.env.config.time_steps])
        plt.ylim([0, self.env.config.fleet_size])
        plt.xlabel(f"Steps ({self.env.config.time_increment} min)")


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

        if file_path:
            plt.savefig(f"{file_path}.{file_format}", dpi=dpi)
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
    center_lines_xy = defaultdict(lambda: {"xs": [], "ys": [], "o": [], "d": [], "level": []})
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

STRAIGHT_LINE = 'OD'
SP_LINE = 'SP'


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
