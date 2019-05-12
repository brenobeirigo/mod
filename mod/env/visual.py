from collections import defaultdict
from mod.env.car import Car
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from mod.env.config import FOLDER_OUTPUT
import os

sns.set(style="ticks")
sns.set_context("paper")


class EpisodeLog:
    def __init__(self, config=None):
        self.n = 0
        self.reward = list()
        self.service_rate = list()

        # If config is not None, then the experiments should be saved
        if config:
            self.output_path = FOLDER_OUTPUT + config.label
            self.output_folder_fleet = self.output_path + "/fleet/"
            self.output_folder_service = self.output_path + "/service/"

            # Creating folders to log episodes
            if not os.path.exists(self.output_path):

                os.makedirs(self.output_folder_fleet)
                os.makedirs(self.output_folder_service)

                print(
                    f"\n### Saving episodes at:"
                    f"\n - {self.output_path}"
                    f"\n### Saving plots at:"
                    f"\n - {self.output_folder_fleet}"
                    f"\n - {self.output_folder_service}"
                )

    def save_origins(self, origin_ids):
        np.save(self.output_path + "/trip_origin_ids.npy", origin_ids)

    def load_origins(self):
        try:
            path_origin_ids = self.output_path + "/trip_origin_ids.npy"
            return np.load(path_origin_ids)
        except Exception as e:
            print(f'Origins at "{path_origin_ids}" could not be find {e}.')
            raise Exception

    def last_episode_stats(self):
        try:
            return f"({self.reward[-1]:15.2f}, {self.service_rate[-1]:6.2%})"
        except:
            return f"(0, 00.00%)"

    def compute_learning(self):

        # Reward over the course of the whole experiment
        self.plot_reward(
            file_path=self.output_path + f"/reward_{self.n:04}_episodes",
            file_format="png",
            dpi=150,
            scale="linear",
        )

        # Service rate over the course of the whole experiment
        self.plot_service_rate(
            file_path=self.output_path + f"/service_rate_{self.n:04}_episodes",
            file_format="png",
            dpi=150,
        )

    def compute_episode(self, step_log, plots=True, progress=False):

        # Increment number of episodes
        self.n += 1

        # Update reward and service rate tracks
        self.reward.append(step_log.total_reward)
        self.service_rate.append(step_log.service_rate)

        # Save intermediate plots
        if plots:

            # Fleet status (idle, recharging, rebalancing, servicing)
            step_log.plot_fleet_status(
                file_path=self.output_folder_fleet + f"{self.n:04}",
                file_format="png",
                dpi=150,
            )

            # Service status (battery level, demand, serviced demand)
            step_log.plot_service_status(
                file_path=self.output_folder_service + f"{self.n:04}",
                file_format="png",
                dpi=150,
            )

        # Save what was learned so far
        if progress:

            path = self.output_path + "/progress.npy"

            # For each:
            # - Time step t,
            # - Aggregation level g,
            #  -Attribute a
            #  Save (value, count) tuple

            value_count = {
                t: {
                    g: {
                        a: (value, step_log.env.count[t][g][a])
                        for a, value in a_value.items()
                    }
                    for g, a_value in g_a.items()
                }
                for t, g_a in step_log.env.values.items()
            }

            np.save(
                path,
                {
                    "episodes": self.n,
                    "reward": self.reward,
                    "service_rate": self.service_rate,
                    "progress": value_count,
                },
            )

    def load(self):
        """Load episodes learned so far

        Returns:
            values, counts -- Value functions and count per aggregation
                level.
        """

        path = self.output_path + "/progress.npy"

        progress = np.load(path).item()

        self.n = progress["episodes"]
        self.reward = progress["reward"]
        self.service_rate = progress["service_rate"]

        values = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
        counts = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))

        for t, g_a in progress["progress"].items():
            for g, a_value in g_a.items():
                for a, value_count in a_value.items():
                    v, c = value_count
                    values[t][g][a] = v
                    counts[t][g][a] = c

        return values, counts

    def plot_reward(
        self, file_path=None, file_format="png", dpi=150, scale="linear"
    ):

        plt.plot(np.arange(self.n), self.reward, color="r")
        plt.xlabel("Episodes")
        plt.xscale(scale)
        plt.ylabel("Reward")

        if file_path:
            plt.savefig(f"{file_path}.{file_format}", dpi=dpi)
        else:
            plt.show()

        plt.close()

    def plot_service_rate(self, file_path=None, file_format="png", dpi=150):

        plt.plot(np.arange(self.n), self.service_rate, color="b")
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

    def add_record(self, reward, serviced, rejected):
        self.n += 1
        self.reward_list.append(reward)
        self.serviced_list.append(len(serviced))
        self.rejected_list.append(len(rejected))
        total = len(serviced) + len(rejected)
        self.total_list.append(total)

        # Get number of cars per status in a time step
        # and aggregate battery level
        dict_status, battery_level = self.env.get_fleet_status()

        # Fleet aggregate battery level
        self.total_battery.append(battery_level)

        # Number of vehicles per status
        for k in Car.status_list:
            self.car_statuses[k].append(dict_status.get(k, 0))

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

    def show_info(self):
        """Print last time step statistics
        """

        try:
            sr = self.serviced_list[-1] / self.total_list[-1]
        except:
            sr = 0

        print(
            f"### Time step: {self.n+1:>3}"
            f" ### Profit: {self.reward_list[-1]:>10.2f}"
            f" ### Service level: {sr:>6.2%}"
            f" ### Trips: {self.total_list[-1]:>3}"
            " ###"
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

    def plot_fleet_status(self, file_path=None, file_format="png", dpi=150):
        steps = np.arange(self.n)

        for status_label, status_count_step in self.car_statuses.items():
            plt.plot(steps, status_count_step, label=status_label)

        matrix_status_count = np.array(list(self.car_statuses.values()))
        total_count = np.sum(matrix_status_count, axis=0)
        plt.plot(steps, total_count, color="#000000", label="Total")

        # Configure x axis
        x_ticks = 6
        x_stride = 20
        max_x = np.math.ceil(self.n / x_stride) * x_stride
        xticks = np.arange(0, max_x + x_stride, x_stride)
        plt.xticks(xticks)
        plt.xlabel("Time (15min)")

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

    def plot_service_status(self, file_path=None, file_format="png", dpi=150):

        max_battery_level = len(self.env.cars) * (
            self.env.cars[0].battery_level_miles_max
            * self.env.config.battery_size_kwh_mile
        )

        # Closest power of 10
        max_battery_level_10 = 10 ** round(np.math.log10(max_battery_level))

        list_battery_level_kwh = (
            np.array(self.total_battery)
            * self.env.config.battery_size_kwh_mile
        )

        steps = np.arange(self.n)

        fig, ax1 = plt.subplots()
        ax1.set_xlabel("Time (15min)")
        ax1.set_ylabel("Trips")
        ax1.plot(steps, self.total_list, label="Total demand", color="b")
        ax1.plot(steps, self.serviced_list, label="Met demand", color="g")
        ax1.legend()
        ax2 = ax1.twinx()
        ax2.plot(
            steps, list_battery_level_kwh, label="Battery Level", color="r"
        )
        ax2.set_ylabel("Total Battery Level (KWh)")

        # Configure ticks x axis
        x_ticks = 6
        x_stride = 20
        max_x = np.math.ceil(self.n / x_stride) * x_stride
        xticks = np.arange(0, max_x + x_stride, x_stride)
        plt.xticks(xticks)

        # Configure ticks y axis (battery level)
        y_ticks = 5  # apart from 0
        y_stride = max_battery_level_10 / y_ticks
        yticks = np.arange(0, max_battery_level_10 + y_stride, y_stride)
        plt.yticks(yticks)

        plt.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))

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

        # Put a legend below current axis
        ax2.legend(
            loc="upper center",
            frameon=False,
            bbox_to_anchor=(0.8, -0.15),
            ncol=1,
        )

        if file_path:
            plt.savefig(f"{file_path}.{file_format}", dpi=dpi)
        else:
            plt.show()
        plt.close()

