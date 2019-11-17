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

if __name__ == "__main__":

    adhoc_compare = [
        "concentric_LIN_cars=0300-0000(L)_t=1_levels[3]=(1-0, 3-300, 3-600)_rebal=([0-8, 1-8][tabu=00])[L(05)][P]_[05h,+30m+04h+60m]_0.10(S)_1.00_0.10",
        "concentric_LIN_cars=0300-0000(R)_t=1_levels[3]=(1-0, 3-300, 3-600)_rebal=([0-8, 1-8][tabu=00])[L(05)][P]_[05h,+30m+04h+60m]_0.10(S)_1.00_0.10",
        "concentric_LIN_cars=0300-0000(R)_t=1_levels[3]=(1-0, 3-300, 3-600)_rebal=([0-8][tabu=00])[L(05)][P]_[05h,+30m+04h+60m]_0.10(S)_1.00_0.10",
        "disjoint_LIN_cars=0300-0000(L)_t=1_levels[3]=(1-0, 3-300, 3-600)_rebal=([0-8, 1-8][tabu=00])[L(05)][P]_[05h,+30m+04h+60m]_0.10(S)_1.00_0.10",
        "disjoint_LIN_cars=0300-0000(R)_t=1_levels[3]=(1-0, 3-300, 3-600)_rebal=([0-8, 1-8][tabu=00])[L(05)][P]_[05h,+30m+04h+60m]_0.10(S)_1.00_0.10",
        "disjoint_LIN_cars=0300-0000(R)_t=1_levels[3]=(1-0, 3-300, 3-600)_rebal=([0-8][tabu=00])[L(05)][P]_[05h,+30m+04h+60m]_0.10(S)_1.00_0.10"
    ]

    ITERATIONS = 500

    try:
        d = dict()
        for exp in adhoc_compare:
            path_all_stats = conf.FOLDER_OUTPUT + exp + "/overall_stats.csv"
            config_exp = ConfigNetwork()
            config_exp.load( conf.FOLDER_OUTPUT + exp + "/exp_settings.json")

            df = pd.read_csv(path_all_stats)
            # spatiotemporal_levels = exp[2].get_levels()
            # neighbors = exp[2].get_reb_neighbors()
            id_label = exp # spatiotemporal_levels + neighbors
            d["reward_" + id_label] = df["Total reward"][:ITERATIONS]
            d["service_rate_" + id_label] = df["Service rate"][:ITERATIONS]
            d["time_" + id_label] = df["time"][:ITERATIONS]
            print(f" - {id_label}")

        df_outcome = pd.DataFrame(d)
        df_outcome = df_outcome[sorted(df_outcome.columns.values)]
        df_outcome.to_csv("outcome_tuning.csv", index=False)
    except Exception as e:
        print(f"{e}")
