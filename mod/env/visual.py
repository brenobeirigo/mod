from collections import defaultdict
from mod.env.car import Car
import matplotlib.pyplot as plt
import numpy as np
import pickle


class EpisodeLog:
    def __init__(self):
        self.n = 0
        self.reward = list()
        self.service_rate = list()

    def last_episode_stats(self):
        try:
            return f"({self.reward[-1]:15.2f}, {self.service_rate[-1]:6.2%})"
        except:
            return f"(0, 00.00%)"

    def add_record(self, reward, service_rate):
        self.n += 1
        self.reward.append(reward)
        self.service_rate.append(service_rate)

    def plot_reward(self, scale="linear"):

        plt.plot(np.arange(self.n), self.reward, color="r", linewidth=0.5)
        plt.xlabel("Episodes")
        plt.xscale(scale)
        plt.ylabel("Reward")
        plt.legend()
        plt.show()

    def plot_service_rate(self):

        plt.plot(np.arange(self.n), self.service_rate, color="b")
        plt.xlabel("Episodes")
        plt.ylabel("Service rate (%)")
        plt.legend()
        plt.show()


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

    def overall_log(self, label="Operational"):

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

        plt.legend()

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
        ax1.plot(steps, self.total_list, label="Trips Requested", color="b")
        ax1.plot(steps, self.serviced_list, label="Trips Taken", color="g")
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

        ax2.legend()

        if file_path:
            plt.savefig(f"{file_path}.{file_format}", dpi=dpi)
        else:
            plt.show()
        plt.close()
