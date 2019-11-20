import os
import sys


# Adding project folder to import modules
root = os.getcwd().replace("\\", "/")
sys.path.append(root)


import mod.env.config as conf
from mod.env.config import ConfigNetwork
import pandas as pd
from copy import deepcopy
from collections import defaultdict
from pprint import pprint
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def movingaverage(data, w):
    new_data = np.zeros(len(data))
    for i in range(len(data)):
        if i + w < len(data):
            new_data[i] = sum(data[i : i + w]) / w
        else:
            new_data[i] = sum(data[i - w : i]) / w
    return new_data


if __name__ == "__main__":

    adhoc_compare = [
        # "TABU_LIN_cars=0300-0000(R)_t=1_levels[3]=(1-0, 3-300, 3-600)_rebal=([0-8, 1-8][tabu=00])[L(05)][P]_[05h,+30m+04h+60m]_0.10(S)_1.00_0.10",
        "ANN_LIN_cars=0300-0000(L)_t=1_levels[3]=(1-0, 3-300, 3-600)_rebal=([0-8, 1-8][tabu=00])[L(05)][P]_[05h,+30m+04h+60m]_match=15_0.10(S)_1.00_0.10",
        # "ANN_LIN_cars=0300-0000(L)_t=1_levels[3]=(1-0, 3-300, 3-600)_rebal=([0-8, 1-8][tabu=10])[L(05)][P]_[05h,+30m+04h+60m]_match=15_0.10(S)_1.00_0.10",
        # "ANN_P_LIN_cars=0300-0000(L)_t=1_levels[3]=(1-0, 3-300, 3-600)_rebal=([0-8, 1-8][tabu=10])[L(05)]_[05h,+30m+04h+60m]_match=15_0.10(S)_1.00_0.10",
        "annealing_thompsom_LIN_cars=0300-0000(L)_t=1_levels[3]=(1-0, 3-300, 3-600)_rebal=([0-8, 1-8][tabu=00])[L(05)]_[05h,+30m+04h+60m]_match=15_0.10(S)_1.00_0.10",
        "annealing_thompson_0.2_LIN_cars=0300-0000(L)_t=1_levels[3]=(1-0, 3-300, 3-600)_rebal=([0-8, 1-8][tabu=00])[L(05)]_[05h,+30m+04h+60m]_match=15_0.10(S)_1.00_0.10",
    ]

    adhoc_compare_labels = [
        "Annealing",
        "Annealing + Thompson (0.5)",
        "Annealing + Thompson (0.2)",
    ]

    ITERATIONS = 500
    colors = [
        "k",
        "g",
        "r",
        "b",
        "#fb9a99",
        "#e31a1c",
        "#fdbf6f",
        "#ff7f00",
        "#cab2d6",
    ]
    markers = [".", "o", "*", "|", "x"]
    linewidth = 2
    shadow = False
    try:
        d = dict()
        d_plot = defaultdict(list)
        window = 10
        for exp, sum_label in zip(adhoc_compare, adhoc_compare_labels):
            path_all_stats = conf.FOLDER_OUTPUT + exp + "/overall_stats.csv"
            config_exp = ConfigNetwork()
            config_exp.load(conf.FOLDER_OUTPUT + exp + "/exp_settings.json")

            df = pd.read_csv(path_all_stats)
            # spatiotemporal_levels = exp[2].get_levels()
            # neighbors = exp[2].get_reb_neighbors()
            id_label = exp  # spatiotemporal_levels + neighbors
            d["reward_" + id_label] = df["Total reward"][:ITERATIONS]
            d["service_rate_" + id_label] = df["Service rate"][:ITERATIONS]
            d["time_" + id_label] = df["time"][:ITERATIONS]

            d_plot["Profit($)"].append(
                (id_label, sum_label, df["Total reward"][:ITERATIONS])
            )
            d_plot["Service level"].append(
                (id_label, sum_label, df["Service rate"][:ITERATIONS])
            )
            # d_plot["Time(s)"].append((id_label, sum_label, df["time"][:ITERATIONS]))
            # print(f" - {id_label}")\

        yticks = dict()
        yticks["Profit($)"] = np.linspace(15000, 19000, 9)
        yticks["Service level"] = np.linspace(0.6, 1, 5)
        df_outcome = pd.DataFrame(d)
        df_outcome = df_outcome[sorted(df_outcome.columns.values)]
        df_outcome.to_csv("outcome_tuning.csv", index=False)

        sns.set(style="ticks")
        sns.set_context("talk")
        np.set_printoptions(precision=3)
        fig, axs = plt.subplots(1, 2, figsize=(18, 6))

        for i, cat_label_data in enumerate(d_plot.items()):

            cat, label_data = cat_label_data

            if shadow:
                for j, label_data in enumerate(label_data):
                    label, sum_label, data = label_data
                    axs[i].plot(
                        data,
                        color=colors[j],
                        linewidth=linewidth,
                        alpha=0.25,
                        label="",
                    )

            cat, label_data = cat_label_data

            for j, label_data in enumerate(label_data):
                label, sum_label, data = label_data
                mavg = movingaverage(data, window)
                axs[i].plot(
                    mavg,
                    color=colors[j],
                    linewidth=linewidth,
                    # marker=markers[j],
                    label=sum_label,
                )
                # axs[i].set_title(vst)
                axs[i].set_xlabel("Iteration")
                axs[i].set_ylabel(cat)
                axs[i].set_xlim(0, len(data))
                axs[i].set_yticks(yticks[cat])

        plt.legend(
            loc="lower right",
            frameon=False,
            # bbox_to_anchor=(1, 0, 0.5, 1), #(0.5, -0.15),
            ncol=1,
        )

        plt.show()

    except Exception as e:
        print(f"{e}")
