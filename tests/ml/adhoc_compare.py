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


def movingaverage(data, w, start=0, start_den=2):
    new_data = np.zeros(len(data))
    for i in range(len(data)):
        if i < start:
            new_data[i] = sum(data[i : i + int(w / start_den)]) / int(
                w / start_den
            )
            continue
        if i + w < len(data):
            new_data[i] = sum(data[i : i + w]) / w
        else:
            new_data[i] = sum(data[i - w : i]) / w
    return new_data


if __name__ == "__main__":

    adhoc_compare = dict()
    adhoc_compare_labels = dict()
    colors = dict()
    markers = dict()
    linewidth = dict()

    # test_label = "penalize"
    test_label = "rebalance"
    # test_label = "pavfav"
    # test_label = "exploration"
    # test_label = "flood"
    # test_label = "policy"
    # test_label = "penalty"

    adhoc_compare["penalty"] = [
        "baselineB10_disable_LIN_cars=0300-0000(R)_t=1_levels[3]=(1-0, 3-300, 3-600)_rebal=([1-4, 2-4][tabu=00])[L(05)][P]_[05h,+30m+04h+60m]_match=15_0.10(S)_1.00_0.10",
        "baselineB10_pen_LIN_cars=0300-0000(R)_t=1_levels[3]=(1-0, 3-300, 3-600)_rebal=([1-4, 2-4][tabu=00])[L(05)][P]_[05h,+30m+04h+60m]_match=15_0.10(S)_1.00_0.10",
        "baselineB10_pen_rej_pen_LIN_cars=0300-0000(R)_t=1_levels[3]=(1-0, 3-300, 3-600)_rebal=([1-4, 2-4][tabu=00])[L(05)][P]_[05h,+30m+04h+60m]_match=15_0.10(S)_1.00_0.10",
    ]

    adhoc_compare_labels["penalty"] = [
        "10 min. pickup delay",
        "10 min. pickup delay + 5 min. tolerance",
        "10 min. pickup delay + 5 min. tolerance + rejection penalty",
    ]

    adhoc_compare["penalize"] = [
        "baseline_R_LIN_cars=0300-0000(R)_t=1_levels[3]=(1-0, 3-300, 3-600)_rebal=([0-8][tabu=00])[L(05)][P]_[05h,+30m+04h+60m]_0.10(S)_1.00_0.10",
        "only1_LIN_cars=0300-0000(R)_t=1_levels[3]=(1-0, 3-300, 3-600)_rebal=([1-8][tabu=00])[L(05)]_[05h,+30m+04h+60m]_match=15_0.10(S)_1.00_0.10",
        "only1_LIN_cars=0300-0000(R)_t=1_levels[3]=(1-0, 3-300, 3-600)_rebal=([1-8][tabu=00])[L(05)][P]_[05h,+30m+04h+60m]_match=15_0.10(S)_1.00_0.10",
        "only1_LIN_cars=0300-0000(R)_t=1_levels[3]=(1-0, 3-300, 3-600)_rebal=([2-8][tabu=00])[L(05)]_[05h,+30m+04h+60m]_match=15_0.10(S)_1.00_0.10",
        "only1_LIN_cars=0300-0000(R)_t=1_levels[3]=(1-0, 3-300, 3-600)_rebal=([2-8][tabu=00])[L(05)][P]_[05h,+30m+04h+60m]_match=15_0.10(S)_1.00_0.10",
        "only1_LIN_cars=0300-0000(R)_t=1_levels[3]=(1-0, 3-300, 3-600)_rebal=([3-8][tabu=00])[L(05)]_[05h,+30m+04h+60m]_match=15_0.10(S)_1.00_0.10",
        "only1_LIN_cars=0300-0000(R)_t=1_levels[3]=(1-0, 3-300, 3-600)_rebal=([3-8][tabu=00])[L(05)][P]_[05h,+30m+04h+60m]_match=15_0.10(S)_1.00_0.10",
    ]

    adhoc_compare_labels["penalize"] = [
        "Adjacent neighbors (30s)",
        "8 x RC1",
        "8 x RC1 [P]",
        "8 x RC5",
        "8 x RC5 [P]",
        "8 x RC10",
        "8 x RC10 [P]",
    ]

    # Rebalance
    adhoc_compare["rebalance"] = [
        "only1_LIN_cars=0300-0000(R)_t=1_levels[3]=(1-0, 3-300, 3-600)_rebal=([1-8][tabu=00])[L(05)][P]_[05h,+30m+04h+60m]_match=15_0.10(S)_1.00_0.10",
        "only1_LIN_cars=0300-0000(R)_t=1_levels[3]=(1-0, 3-300, 3-600)_rebal=([1-8, 2-4][tabu=00])[L(05)][P]_[05h,+30m+04h+60m]_match=15_0.10(S)_1.00_0.10",
        "only1_LIN_cars=0300-0000(R)_t=1_levels[3]=(1-0, 3-300, 3-600)_rebal=([1-8, 3-4][tabu=00])[L(05)][P]_[05h,+30m+04h+60m]_match=15_0.10(S)_1.00_0.10",
        "far_LIN_cars=0300-0000(R)_t=1_levels[3]=(1-0, 3-300, 3-600)_rebal=([1-8, 2-4, 3-2][tabu=00])[L(05)][P]_[05h,+30m+04h+60m]_match=15_0.10(S)_1.00_0.10",
    ]

    adhoc_compare_labels["rebalance"] = [
        "8 x RC1",
        "8 x RC1 + 4 x RC5",
        "8 x RC1 + 4 x RC10",
        "8 x RC1 + 4 x RC5 + 2 x RC10",
    ]

    adhoc_compare["policy"] = [
        "myopic_[MY]_LIN_cars=0300-0000(R)_t=1_levels[3]=(1-0, 3-300, 3-600)_rebal=([1-8, 2-4][tabu=00])[L(05)][P]_[05h,+30m+04h+60m]_match=15_0.10(S)_1.00_0.10",
        # "myopic_[RA]_LIN_cars=0300-0000(R)_t=1_levels[3]=(1-0, 3-300, 3-600)_rebal=([1-8, 2-4][tabu=00])[L(05)][P]_[05h,+30m+04h+60m]_match=15_0.10(S)_1.00_0.10",
        "only1_LIN_cars=0300-0000(R)_t=1_levels[3]=(1-0, 3-300, 3-600)_rebal=([1-8, 2-4][tabu=00])[L(05)][P]_[05h,+30m+04h+60m]_match=15_0.10(S)_1.00_0.10",
        "annealing_hire_LIN_cars=0300-0200(R)_t=1_levels[3]=(1-0, 3-300, 3-600)_rebal=([1-8][tabu=00])[L(05)][P]_[05h,+30m+04h+60m]_match=15_0.10(S)_1.00_0.10",
    ]
    adhoc_compare_labels["policy"] = [
        "Myopic",
        # "Random rebalance",
        "VFA (300 PAVs)",
        "VFA (300 PAVs + 200 FAVs)",
    ]

    # Rebalance
    # adhoc_compare["flood"] = [
    #     "only1_LIN_cars=0300-0000(R)_t=1_levels[3]=(1-0, 3-300, 3-600)_rebal=([1-8][tabu=00])[L(05)][P]_[05h,+30m+04h+60m]_match=15_0.10(S)_1.00_0.10",
    #     "only1_LIN_cars=0300-0000(R)_t=1_levels[3]=(1-0, 3-300, 3-600)_rebal=([1-8][tabu=00])[P]_[05h,+30m+04h+60m]_match=15_0.10(S)_1.00_0.10",
    #     "only1_LIN_cars=0300-0000(R)_t=1_levels[3]=(1-0, 3-300, 3-600)_rebal=([1-8, 2-4][tabu=00])[L(05)][P]_[05h,+30m+04h+60m]_match=15_0.10(S)_1.00_0.10",
    #     "only1_LIN_cars=0300-0000(R)_t=1_levels[3]=(1-0, 3-300, 3-600)_rebal=([1-8, 2-4][tabu=00])[P]_[05h,+30m+04h+60m]_match=15_0.10(S)_1.00_0.10",
    # ]

    # adhoc_compare_labels["flood"] = [
    #     "8 x RC1",
    #     "8 x RC1 (unlimited)",
    #     "8 x RC1 + 4 x RC5",
    #     "8 x RC1 + 4 x RC5 (unlimited)",
    # ]

    adhoc_compare["flood"] = [
        # "only1_LIN_cars=0300-0000(R)_t=1_levels[3]=(1-0, 3-300, 3-600)_rebal=([1-8, 2-4][tabu=00])[P]_[05h,+30m+04h+60m]_match=15_0.10(S)_1.00_0.10",
        "far_LIN_cars=0300-0000(R)_t=1_levels[3]=(1-0, 3-300, 3-600)_rebal=([1-8, 2-4][tabu=00])[L(02)][P]_[05h,+30m+04h+60m]_match=15_0.10(S)_1.00_0.10",
        "only1_LIN_cars=0300-0000(R)_t=1_levels[3]=(1-0, 3-300, 3-600)_rebal=([1-8, 2-4][tabu=00])[L(05)][P]_[05h,+30m+04h+60m]_match=15_0.10(S)_1.00_0.10",
        "only1_LIN_cars=0300-0000(R)_t=1_levels[3]=(1-0, 3-300, 3-600)_rebal=([1-8, 2-4][tabu=00])[L(10)][P]_[05h,+30m+04h+60m]_match=15_0.10(S)_1.00_0.10",
    ]

    adhoc_compare_labels["flood"] = [
        # "unlimited",
        "2",
        "5",
        "10",
    ]

    # adhoc_compare_labels["avoidflood"] = [
    # ]

    adhoc_compare["exploration"] = [
        "only1_LIN_cars=0300-0000(R)_t=1_levels[3]=(1-0, 3-300, 3-600)_rebal=([1-8, 2-4][tabu=00])[L(05)][P]_[05h,+30m+04h+60m]_match=15_0.10(S)_1.00_0.10",
        "annealing_[X]LIN_cars=0300-0000(R)_t=1_levels[3]=(1-0, 3-300, 3-600)_rebal=([1-8][tabu=00])[L(05)][P]_[05h,+30m+04h+60m]_match=15_0.10(S)_1.00_0.10",
        "annealing0.25_[X]LIN_cars=0300-0000(R)_t=1_levels[3]=(1-0, 3-300, 3-600)_rebal=([1-8][tabu=00])[L(05)][P]_[05h,+30m+04h+60m]_match=15_0.10(S)_1.00_0.10",
        "far_LIN_cars=0300-0000(R)_t=1_levels[3]=(1-0, 3-300, 3-600)_rebal=([1-8, 2-4][thompson=06][tabu=00])[L(05)][P]_[05h,+30m+04h+60m]_match=15_0.10(S)_1.00_0.10",
        "annealing_[X]LIN_cars=0300-0000(R)_t=1_levels[3]=(1-0, 3-300, 3-600)_rebal=([1-16][thompson=08][tabu=00])[L(05)][P]_[05h,+30m+04h+60m]_match=15_0.10(S)_1.00_0.10",
    ]

    adhoc_compare_labels["exploration"] = [
        "8 x RC1 + 4 x RC5",
        # "16 x RC1",
        # "16 x RC1 (annealing)",
        # "16 x RC1 (annealing thompson 8)",
        "8 x RC1 (annealing)",
        "8 x RC1 (annealing 0.25)",
        "8 x RC1 + 4 x RC5 (thompson 6)",
        "16 x RC1 (thompson 6)",
    ]

    adhoc_compare["pavfav"] = [
        "only1_LIN_cars=0300-0000(R)_t=1_levels[3]=(1-0, 3-300, 3-600)_rebal=([1-8][tabu=00])[L(05)][P]_[05h,+30m+04h+60m]_match=15_0.10(S)_1.00_0.10",
        "annealing_hire_LIN_cars=0300-0200(R)_t=1_levels[3]=(1-0, 3-300, 3-600)_rebal=([1-8][tabu=00])[L(05)][P]_[05h,+30m+04h+60m]_match=15_0.10(S)_1.00_0.10",
    ]

    adhoc_compare_labels["pavfav"] = ["300 PAVs", "300 PAVs + 200 FAVs"]

    # adhoc_compare_labels = [
    #     "Rebalance to closest nodes",
    #     "Rebalance to closest nodes + 1min RCs",
    #     "Rebalance to 1min RCs",
    #     "Rebalance to 1min RCs [P]",
    #     "Rebalance to 1min (8), 5min(4), 10min(2) RCs [P]",
    #     "Rebalance to 1min (8), 5min(4), 10min(2) RCs [P] + annealing",
    #     "Rebalance to 1min RCs [P] + annealing",
    #     "Rebalance to 1min RCs [P] + annealing (0.1)",
    #     # "Annealing",
    #     # "Annealing + Thompson (0.5)",
    #     # "Annealing + Thompson (0.2)",
    # ]

    colors["rebalance"] = [
        "k",
        "g",
        "r",
        "b",
        "magenta",
        "gold",
        "gray",
        "pink",
        "#cab2d6",
    ]

    colors["policy"] = ["k", "r", "g"]
    markers["policy"] = [None, "x", "D"]

    colors["pavfav"] = ["k", "r"]

    colors["flood"] = ["r", "k", "g", "r"]

    colors["exploration"] = [
        "k",
        "g",
        "r",
        "b",
        "magenta",
        "gold",
        "gray",
        "pink",
        "#cab2d6",
    ]

    colors["penalize"] = ["k", "g", "g", "r", "r", "b", "b"]
    markers["penalize"] = [None, None, "o", None, "o", None, "o"]
    # linewidth["penalize"] = [2, 2, 1, 2, 1, 2, 1]

    linewidth["penalize"] = [1, 1, 1, 1, 1, 1, 1]
    # linewidth["rebalance"] = [1, 1, 1, 1, 1, 1, 1]
    linewidth["rebalance"] = [1, 1, 1, 1, 1, 1, 1]*2
    # linewidth["policy"] = [1, 1, 1, 1, 1, 1, 1]
    markers["rebalance"] = [None, "o", "x", "D"]

    linewidth["pavfav"] = [1, 1, 1, 1, 1, 1, 1]
    markers["pavfav"] = [None, "o", "x"]

    linewidth["flood"] = [1, 1, 1, 1, 1, 1, 1]
    markers["flood"] = ["x", None, "o"]

    linewidth["exploration"] = [1, 1, 1, 1, 1, 1, 1]
    # markers["exploration"] = [None, "o", "x"]

    colors_default = [
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
    markers_default = [None] * len(adhoc_compare[test_label])
    linewidth_default = [1] * len(adhoc_compare[test_label])

    legend_pos = dict()
    legend_pos["policy"] = "center right"

    SL = "Users serviced"
    OF = "Profit($)"
    XLABEL = "Iteration"
    window = 30
    ITERATIONS = 500

    # markers = [None, "o", "*", "x", "|", None]

    shadow = False
    dpi = 1200
    try:
        d = dict()
        d_plot = defaultdict(list)

        for exp, sum_label in zip(
            adhoc_compare[test_label], adhoc_compare_labels[test_label]
        ):  
            folder = "O:/phd/output_paper/"
            path_all_stats = folder + exp + "/overall_stats.csv"
            config_exp = ConfigNetwork()
            config_exp.load(folder + exp + "/exp_settings.json")

            df = pd.read_csv(path_all_stats)
            # spatiotemporal_levels = exp[2].get_levels()
            # neighbors = exp[2].get_reb_neighbors()
            id_label = exp  # spatiotemporal_levels + neighbors
            d["reward_" + id_label] = df["Total reward"][:ITERATIONS]
            d["service_rate_" + id_label] = df["Service rate"][:ITERATIONS]
            d["time_" + id_label] = df["time"][:ITERATIONS]

            d_plot[OF].append(
                (id_label, sum_label, df["Total reward"][:ITERATIONS].values)
            )
            d_plot[SL].append(
                (id_label, sum_label, df["Service rate"][:ITERATIONS].values)
            )

            # d_plot["Time(s)"].append(
            #     (id_label, sum_label, df["time"][:ITERATIONS])
            # )
            # print(f" - {id_label}")\

        yticks = dict()
        yticks_labels = dict()
        yticks[OF] = np.linspace(15000, 18500, 8)
        yticks[SL] = np.linspace(0.7, 0.9, 5)

        # Policy
        # yticks[OF] = np.linspace(13000, 19000, 13)
        # yticks[SL] = np.linspace(0.5, 0.95, 10)

        #yticks[OF] = np.linspace(10000, 20000, 9)
        #yticks[SL] = np.linspace(0.45, 1, 12)

        yticks_labels[SL] = [f"{s:3.0%}" for s in yticks[SL]]
        yticks_labels[OF] = [f"{p:,.0f}" for p in yticks[OF]]
        yticks["Time(s)"] = np.linspace(0, 300, 5)
        yticks_labels["Time(s)"] = np.linspace(0, 300, 5)
        df_outcome = pd.DataFrame(d)
        df_outcome = df_outcome[sorted(df_outcome.columns.values)]
        df_outcome.to_csv("outcome_tuning.csv", index=False)

        sns.set(style="ticks")
        sns.set_context("talk")
        # sns.set_context("paper")
        np.set_printoptions(precision=3)
        fig, axs = plt.subplots(1, len(d_plot), figsize=(8 * len(d_plot), 6))

        for i, cat_label_data in enumerate(d_plot.items()):

            cat, label_data = cat_label_data

            if shadow:
                for j, label_data in enumerate(label_data):
                    label, sum_label, data = label_data
                    axs[i].plot(
                        data,
                        color=colors.get(test_label, colors_default)[j],
                        linewidth=linewidth.get(test_label, linewidth_default)[
                            j
                        ],
                        marker=markers.get(test_label, markers_default)[j],
                        alpha=0.25,
                        label="",
                    )
            cat, label_data = cat_label_data

            for j, label_data in enumerate(label_data):
                label, sum_label, data = label_data
                mavg = movingaverage(data, window)
                axs[i].plot(
                    mavg,
                    color=colors.get(test_label, colors_default)[j],
                    linewidth=linewidth.get(test_label, [1] * len(label_data))[
                        j
                    ],
                    marker=markers.get(test_label, markers_default)[j],
                    fillstyle="none",
                    markevery=25,
                    # linestyle=':',
                    label=sum_label,
                )

                # And add a special annotation for the group we are interested in
                # axs[i].text(ITERATIONS+0.2, mavg[-1], sum_label, horizontalalignment='left', size='small', color='k')

                # axs[i].set_title(vst)
                axs[i].set_xlabel(XLABEL)
                axs[i].set_ylabel(cat)
                axs[i].set_xlim(0, len(data))
                axs[i].set_yticks(yticks[cat])
                axs[i].set_yticklabels(yticks_labels[cat])

        plt.legend(
            loc=legend_pos.get(test_label, "lower right"),
            frameon=False,
            bbox_to_anchor=(1, 0, 0, 1),  # (0.5, -0.15),
            ncol=1,
        )

        # plt.show()
        print(f'Saving "{test_label}.png"...')
        plt.savefig(f"{test_label}.pdf", bbox_inches="tight", dpi=dpi)

    except Exception as e:
        print(f"Exception: {e}")
