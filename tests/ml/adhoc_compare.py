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
sns.set(style="ticks")

context = "paper"
fig_format = "pdf"


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

    test_label = "hire500"
    # test_label = "rebalance"
    # test_label = "pavfav"
    # test_label = "exploration"
    # test_label = "flood"
    # test_label = "unlimited"
    # test_label = "policy"
    # test_label = "b"

    # adhoc_compare["p"] = [
    #     "SL_LIN_cars=0300-0000(R)_t=1_levels[3]=(10-0-, 32-0-, 33-0-)_rebal=([1-8][tabu=00])[L(05)][P]_[05h,+30m+04h+60m]_match=15_0.10(S)_1.00_0.10_A_4.80_5.00_0.00_0.00_1.00_B_2.40_10.00_0.00_0.00_0.00",
    #     "SL_LIN_cars=0300-0000(R)_t=1_levels[3]=(10-0-, 32-0-, 33-0-)_rebal=([1-8][tabu=00])[L(05)][P]_[05h,+30m+04h+60m]_match=15_0.10(S)_1.00_0.10_A_4.80_5.00_0.00_4.80_1.00_B_2.40_10.00_0.00_2.40_0.00",
    #     "SL_LIN_cars=0300-0000(R)_t=1_levels[3]=(10-0-, 32-0-, 33-0-)_rebal=([1-8][tabu=00])[L(05)][P]_[05h,+30m+04h+60m]_match=15_0.10(S)_1.00_0.10_A_4.80_5.00_5.00_0.00_1.00_B_2.40_10.00_5.00_0.00_0.00",
    #     "SL_LIN_cars=0300-0000(R)_t=1_levels[3]=(10-0-, 32-0-, 33-0-)_rebal=([1-8][tabu=00])[L(05)][P]_[05h,+30m+04h+60m]_match=15_0.10(S)_1.00_0.10_A_4.80_5.00_5.00_4.80_1.00_B_2.40_10.00_5.00_2.40_0.00",
    #     "SL_LIN_cars=0300-0000(R)_t=1_levels[3]=(10-0-, 32-0-, 33-0-)_rebal=([1-8][tabu=00])[L(05)][P]_[05h,+30m+04h+60m]_match=15_0.10(S)_1.00_0.10_A_4.80_10.00_0.00_0.00_1.00_B_2.40_15.00_0.00_0.00_0.00",
    # ]
    d = "0.01"
    adhoc_compare["hire500"] = [
        f"SH_LIN_V=0000-0500[S{d}](R)_I=1_L[5]=(10102, 10203, 1030-, 32-0-, 33-0-)_R=([1-8][L(05)]_T=[05h,+30m+04h+60m]_0.10(S)_1.00_0.10_A_4.80_5.00_5.00_4.80_P_B_2.40_10.00_5.00_2.40_P",
        f"SH_LIN_V=0000-0500[S{d}](R)_I=1_L[3]=(10202, 32303, 33-0-)_R=([1-8][L(05)]_T=[05h,+30m+04h+60m]_0.10(S)_1.00_0.10_A_4.80_5.00_5.00_4.80_P_B_2.40_10.00_5.00_2.40_P",
        f"SH_LIN_V=0000-0500[S{d}](R)_I=1_L[3]=(10-02, 32-03, 33-0-)_R=([1-8][L(05)]_T=[05h,+30m+04h+60m]_0.10(S)_1.00_0.10_A_4.80_5.00_5.00_4.80_P_B_2.40_10.00_5.00_2.40_P",
        f"SH_LIN_V=0000-0500[S{d}](R)_I=1_L[3]=(10-0-, 32-0-, 33-0-)_R=([1-8][L(05)]_T=[05h,+30m+04h+60m]_0.10(S)_1.00_0.10_A_4.80_5.00_5.00_4.80_P_B_2.40_10.00_5.00_2.40_P",
        f"SH_LIN_V=0000-0500[S{d}](R)_I=1_L[4]=(10203, 1030-, 32-0-, 33-0-)_R=([1-8][L(05)]_T=[05h,+30m+04h+60m]_0.10(S)_1.00_0.10_A_4.80_5.00_5.00_4.80_P_B_2.40_10.00_5.00_2.40_P",
        f"SH_LIN_V=0000-0500[S{d}](R)_I=1_L[3]=(10303, 32-0-, 33-0-)_R=([1-8][L(05)]_T=[05h,+30m+04h+60m]_0.10(S)_1.00_0.10_A_4.80_5.00_5.00_4.80_P_B_2.40_10.00_5.00_2.40_P",
        f"SH_LIN_V=0000-0500[S{d}](R)_I=1_L[3]=(32202, 33303, 33-0-)_R=([1-8][L(05)]_T=[05h,+30m+04h+60m]_0.10(S)_1.00_0.10_A_4.80_5.00_5.00_4.80_P_B_2.40_10.00_5.00_2.40_P",

        # "SH_LIN_V=0000-0500[S0.10](R)_I=1_L[5]=(10102, 10203, 1030-, 32-0-, 33-0-)_R=([1-8][L(05)]_T=[05h,+30m+04h+60m]_0.10(S)_1.00_0.10_A_4.80_5.00_5.00_4.80_P_B_2.40_10.00_5.00_2.40_P",
        # "SH_LIN_V=0000-0500[S0.10](R)_I=1_L[3]=(10202, 32303, 33-0-)_R=([1-8][L(05)]_T=[05h,+30m+04h+60m]_0.10(S)_1.00_0.10_A_4.80_5.00_5.00_4.80_P_B_2.40_10.00_5.00_2.40_P",
        # "SH_LIN_V=0000-0500[S0.10](R)_I=1_L[3]=(10-02, 32-03, 33-0-)_R=([1-8][L(05)]_T=[05h,+30m+04h+60m]_0.10(S)_1.00_0.10_A_4.80_5.00_5.00_4.80_P_B_2.40_10.00_5.00_2.40_P",
        # "SH_LIN_V=0000-0500[S0.10](R)_I=1_L[3]=(10-0-, 32-0-, 33-0-)_R=([1-8][L(05)]_T=[05h,+30m+04h+60m]_0.10(S)_1.00_0.10_A_4.80_5.00_5.00_4.80_P_B_2.40_10.00_5.00_2.40_P",
        # "SH_LIN_V=0000-0500[S0.10](R)_I=1_L[4]=(10203, 1030-, 32-0-, 33-0-)_R=([1-8][L(05)]_T=[05h,+30m+04h+60m]_0.10(S)_1.00_0.10_A_4.80_5.00_5.00_4.80_P_B_2.40_10.00_5.00_2.40_P",
        # "SH_LIN_V=0000-0500[S0.10](R)_I=1_L[3]=(10303, 32-0-, 33-0-)_R=([1-8][L(05)]_T=[05h,+30m+04h+60m]_0.10(S)_1.00_0.10_A_4.80_5.00_5.00_4.80_P_B_2.40_10.00_5.00_2.40_P",
        # "SH_LIN_V=0000-0500[S0.10](R)_I=1_L[3]=(32202, 33303, 33-0-)_R=([1-8][L(05)]_T=[05h,+30m+04h+60m]_0.10(S)_1.00_0.10_A_4.80_5.00_5.00_4.80_P_B_2.40_10.00_5.00_2.40_P",

        # "SH_LIN_V=0000-0500[S0.01](R)_I=1_L[5]=(10102, 10203, 1030-, 32-0-, 33-0-)_R=([1-8][L(05)]_T=[05h,+30m+04h+60m]_0.10(S)_1.00_0.10_A_4.80_5.00_5.00_4.80_P_B_2.40_10.00_5.00_2.40_P",
        # "SH_LIN_V=0000-0500[S0.01](R)_I=1_L[3]=(10202, 32303, 33-0-)_R=([1-8][L(05)]_T=[05h,+30m+04h+60m]_0.10(S)_1.00_0.10_A_4.80_5.00_5.00_4.80_P_B_2.40_10.00_5.00_2.40_P",
        # "SH_LIN_V=0000-0500[S0.01](R)_I=1_L[3]=(10-02, 32-03, 33-0-)_R=([1-8][L(05)]_T=[05h,+30m+04h+60m]_0.10(S)_1.00_0.10_A_4.80_5.00_5.00_4.80_P_B_2.40_10.00_5.00_2.40_P",
        # "SH_LIN_V=0000-0500[S0.01](R)_I=1_L[3]=(10-0-, 32-0-, 33-0-)_R=([1-8][L(05)]_T=[05h,+30m+04h+60m]_0.10(S)_1.00_0.10_A_4.80_5.00_5.00_4.80_P_B_2.40_10.00_5.00_2.40_P",
        # "SH_LIN_V=0000-0500[S0.01](R)_I=1_L[4]=(10203, 1030-, 32-0-, 33-0-)_R=([1-8][L(05)]_T=[05h,+30m+04h+60m]_0.10(S)_1.00_0.10_A_4.80_5.00_5.00_4.80_P_B_2.40_10.00_5.00_2.40_P",
        # "SH_LIN_V=0000-0500[S0.01](R)_I=1_L[3]=(10303, 32-0-, 33-0-)_R=([1-8][L(05)]_T=[05h,+30m+04h+60m]_0.10(S)_1.00_0.10_A_4.80_5.00_5.00_4.80_P_B_2.40_10.00_5.00_2.40_P",
        # "SH_LIN_V=0000-0500[S0.01](R)_I=1_L[3]=(32202, 33303, 33-0-)_R=([1-8][L(05)]_T=[05h,+30m+04h+60m]_0.10(S)_1.00_0.10_A_4.80_5.00_5.00_4.80_P_B_2.40_10.00_5.00_2.40_P",

        # "SH_LIN_V=0000-0500[S0.01](R)_I=1_L[3]=(10202, 32303, 33-0-)_R=([1-8][L(05)]_T=[05h,+30m+04h+60m]_0.10(S)_1.00_0.10_A_4.80_5.00_5.00_4.80_P_B_2.40_10.00_5.00_2.40_P",
        # "SH_LIN_V=0000-0500[S0.01](R)_I=1_L[3]=(10-02, 32-03, 33-0-)_R=([1-8][L(05)]_T=[05h,+30m+04h+60m]_0.10(S)_1.00_0.10_A_4.80_5.00_5.00_4.80_P_B_2.40_10.00_5.00_2.40_P",
        # "SH_LIN_V=0000-0500[S0.01](R)_I=1_L[3]=(10-0-, 32-0-, 33-0-)_R=([1-8][L(05)]_T=[05h,+30m+04h+60m]_0.10(S)_1.00_0.10_A_4.80_5.00_5.00_4.80_P_B_2.40_10.00_5.00_2.40_P",
        # "SH_LIN_V=0000-0500[S0.01](R)_I=1_L[4]=(10203, 1030-, 32-0-, 33-0-)_R=([1-8][L(05)]_T=[05h,+30m+04h+60m]_0.10(S)_1.00_0.10_A_4.80_5.00_5.00_4.80_P_B_2.40_10.00_5.00_2.40_P",
        # "SH_LIN_V=0000-0500[S0.01](R)_I=1_L[3]=(10303, 32-0-, 33-0-)_R=([1-8][L(05)]_T=[05h,+30m+04h+60m]_0.10(S)_1.00_0.10_A_4.80_5.00_5.00_4.80_P_B_2.40_10.00_5.00_2.40_P",
    ]

    colors["hire500"] = ["k", "g", "r", "b", "k", "g", "r", "b", "k", "g", "r", "b"]
    markers["hire500"] = [None, None, None, None, "o", "o", "o","o",  "x","x","x","x"]

    adhoc_compare_labels["hire500"] = [

        f"{d} - (10102, 10203, 1030-, 32-0-, 33-0-)",
        f"{d} - (10202, 32302, 33-0-)",
        f"{d} - (10-02, 32-02, 33-0-)",
        f"{d} - (10-0-, 32-0-, 33-0-)",
        f"{d} - (10203, 1030-, 32-0-, 33-0-)",
        f"{d} - (10303, 32-0-, 33-0-)",
        f"{d} - (32202, 33303, 33-0-)",

        "0.10 - (10102, 10203, 1030-, 32-0-, 33-0-)",
        "0.10 - (10202, 32302, 33-0-)",
        "0.10 - (10-02, 32-02, 33-0-)",
        "0.10 - (10-0-, 32-0-, 33-0-)",
        "0.10 - (10203, 1030-, 32-0-, 33-0-)",
        "0.10 - (10303, 32-0-, 33-0-)",
        "0.10 - (32202, 33303, 33-0-)",

        "0.10 - (10102, 10203, 1030-, 32-0-, 33-0-)",
        "0.10 - (10202, 32302, 33-0-)",
        "0.10 - (10-02, 32-02, 33-0-)",
        "0.10 - (10-0-, 32-0-, 33-0-)",
        "0.10 - (10203, 1030-, 32-0-, 33-0-)",
        "0.10 - (10303, 32-0-, 33-0-)",
        "0.10 - (32202, 33303, 33-0-)",
    ]

    # adhoc_compare["hire500m"] = [
    #     "HI_LIN_V=0000-0500[S1.00][M](R)_I=1_L[3]=(10-01, 32-02, 33-03)_R=([1-8][L(05)]T=[05h,+30m+04h+60m]_0.10(S)_1.00_0.10_A_4.80_5.00_5.00_4.80_0.00_B_2.40_10.00_5.00_2.40_1.00",
    #     "HI_LIN_V=0000-0500[S1.00][M](R)_I=1_L[3]=(10-01, 32-02, 33-03)_R=([1-8][L(05)]T=[05h,+30m+04h+60m]_0.10(S)_1.00_0.10_A_4.80_5.00_5.00_4.80_1.00_B_2.40_10.00_5.00_2.40_0.00",
    #     "HI_LIN_V=0000-0500[S1.00][M](R)_I=1_L[3]=(10-02, 32-0-, 33-0-)_R=([1-8][L(05)]T=[05h,+30m+04h+60m]_0.10(S)_1.00_0.10_A_4.80_5.00_5.00_4.80_0.00_B_2.40_10.00_5.00_2.40_1.00",
    #     "HI_LIN_V=0000-0500[S1.00][M](R)_I=1_L[3]=(10-02, 32-0-, 33-0-)_R=([1-8][L(05)]T=[05h,+30m+04h+60m]_0.10(S)_1.00_0.10_A_4.80_5.00_5.00_4.80_1.00_B_2.40_10.00_5.00_2.40_0.00",
    #     "HI_LIN_V=0000-0500[S1.00][M](R)_I=1_L[3]=(10-02, 32-03, 33-0-)_R=([1-8][L(05)]T=[05h,+30m+04h+60m]_0.10(S)_1.00_0.10_A_4.80_5.00_5.00_4.80_0.00_B_2.40_10.00_5.00_2.40_1.00",
    #     "HI_LIN_V=0000-0500[S1.00][M](R)_I=1_L[3]=(10-02, 32-03, 33-0-)_R=([1-8][L(05)]T=[05h,+30m+04h+60m]_0.10(S)_1.00_0.10_A_4.80_5.00_5.00_4.80_1.00_B_2.40_10.00_5.00_2.40_0.00",
    #     "HI_LIN_V=0000-0500[S1.00][M](R)_I=1_L[4]=(10-02, 10-03, 32-0-, 33-0-)_R=([1-8][L(05)]T=[05h,+30m+04h+60m]_0.10(S)_1.00_0.10_A_4.80_5.00_5.00_4.80_0.00_B_2.40_10.00_5.00_2.40_1.00",
    #     "HI_LIN_V=0000-0500[S1.00][M](R)_I=1_L[4]=(10-02, 10-03, 32-0-, 33-0-)_R=([1-8][L(05)]T=[05h,+30m+04h+60m]_0.10(S)_1.00_0.10_A_4.80_5.00_5.00_4.80_1.00_B_2.40_10.00_5.00_2.40_0.00",
    # ]

    # adhoc_compare_labels["hire500m"] = [
    #     "500[M] - (10-01, 32-02, 33-03) - 1",
    #     "500[M] - (10-01, 32-02, 33-03) - 2",
    #     "500[M] - (10-02, 32-0-, 33-0-) - 1",
    #     "500[M] - (10-02, 32-0-, 33-0-) - 2",
    #     "500[M] - (10-02, 32-03, 33-0-) - 1",
    #     "500[M] - (10-02, 32-03, 33-0-) - 2",
    #     "500[M] - (10-02, 10-03, 32-0-, 33-0-) - 1",
    #     "500[M] - (10-02, 10-03, 32-0-, 33-0-) - 2",
    # ]

    adhoc_compare["b"] = [
        "SL_LIN_cars=0300-0000(R)_t=1_levels[3]=(10-0-, 32-0-, 33-0-)_rebal=([1-8][tabu=00])[L(05)][P]_[05h,+30m+04h+60m]_match=15_0.10(S)_1.00_0.10_A_4.80_5.00_0.00_0.00_0.00_B_2.40_10.00_0.00_0.00_1.00",
        "SL_LIN_cars=0300-0000(R)_t=1_levels[3]=(10-0-, 32-0-, 33-0-)_rebal=([1-8][tabu=00])[L(05)][P]_[05h,+30m+04h+60m]_match=15_0.10(S)_1.00_0.10_A_4.80_5.00_0.00_4.80_0.00_B_2.40_10.00_0.00_2.40_1.00",
        "SL_LIN_cars=0300-0000(R)_t=1_levels[3]=(10-0-, 32-0-, 33-0-)_rebal=([1-8][tabu=00])[L(05)][P]_[05h,+30m+04h+60m]_match=15_0.10(S)_1.00_0.10_A_4.80_5.00_5.00_0.00_0.00_B_2.40_10.00_5.00_0.00_1.00",
        "SL_LIN_cars=0300-0000(R)_t=1_levels[3]=(10-0-, 32-0-, 33-0-)_rebal=([1-8][tabu=00])[L(05)][P]_[05h,+30m+04h+60m]_match=15_0.10(S)_1.00_0.10_A_4.80_5.00_5.00_4.80_0.00_B_2.40_10.00_5.00_2.40_1.00",
        "SL_LIN_cars=0300-0000(R)_t=1_levels[3]=(10-0-, 32-0-, 33-0-)_rebal=([1-8][tabu=00])[L(05)][P]_[05h,+30m+04h+60m]_match=15_0.10(S)_1.00_0.10_A_4.80_5.00_5.00_7.20_0.00_B_2.40_10.00_5.00_4.80_1.00",
        "SL_LIN_cars=0300-0000(R)_t=1_levels[3]=(10-0-, 32-0-, 33-0-)_rebal=([1-8][tabu=00])[L(05)][P]_[05h,+30m+04h+60m]_match=15_0.10(S)_1.00_0.10_A_4.80_5.00_5.00_9.60_0.00_B_2.40_10.00_5.00_7.20_1.00",
        "SL_LIN_cars=0300-0000(R)_t=1_levels[3]=(10-0-, 32-0-, 33-0-)_rebal=([1-8][tabu=00])[L(05)][P]_[05h,+30m+04h+60m]_match=15_0.10(S)_1.00_0.10_A_4.80_10.00_0.00_0.00_0.00_B_2.40_15.00_0.00_0.00_1.00",
    ]

    adhoc_compare_labels["b"] = [
        r"10min (max. pk. delay)",
        r"10min (max. pk. delay) + 1 $\times$ RP",
        r"10min (max. pk. delay) + 5min (pen. tolerance)",
        r"10min (max. pk. delay) + 5min (pen. tolerance) + 1 $\times$ RP",
        r"10min (max. pk. delay) + 5min (pen. tolerance) + 2 $\times$ RP",
        r"10min (max. pk. delay) + 5min (pen. tolerance) + 3 $\times$ RP",
        r"15min (max. pk. delay) + 5min (pen. tolerance)",
    ]

    adhoc_compare_labels["sensitivity_analysis"] = [
        "SEN_LIN_cars=0300-0000(R)_t=1_levels[3]=(10-0-, 32-0-, 33-0-)_rebal=([1-8][tabu=00])[L(05)][P]_[05h,+30m+04h+60m]_match=15_0.10(S)_1.00_0.10_A_9.60_10.00_0.00_0.00_1.00_B_7.20_15.00_0.00_0.0,0_0.00",
        "SEN_LIN_cars=0300-0000(R)_t=1_levels[3]=(10-0-, 32-0-, 33-0-)_rebal=([1-8][tabu=00])[L(05)][P]_[05h,+30m+04h+60m]_match=15_0.10(S)_1.00_0.10_A_9.60_10.00_0.00_0.00_0.00_B_7.20_15.00_0.00_0.00_1.00",
        "SEN_LIN_cars=0300-0000(R)_t=1_levels[3]=(10-0-, 32-0-, 33-0-)_rebal=([1-8][tabu=00])[L(05)][P]_[05h,+30m+04h+60m]_match=15_0.10(S)_1.00_0.10_A_9.60_5.00_0.00_0.00_1.00_B_7.20_10.00_0.00_0.00_0.00",
        "SEN_LIN_cars=0300-0000(R)_t=1_levels[3]=(10-0-, 32-0-, 33-0-)_rebal=([1-8][tabu=00])[L(05)][P]_[05h,+30m+04h+60m]_match=15_0.10(S)_1.00_0.10_A_9.60_5.00_0.00_0.00_0.00_B_7.20_10.00_0.00_0.00_1.00",
        "SEN_LIN_cars=0300-0000(R)_t=1_levels[3]=(10-0-, 32-0-, 33-0-)_rebal=([1-8][tabu=00])[L(05)][P]_[05h,+30m+04h+60m]_match=15_0.10(S)_1.00_0.10_A_7.20_10.00_0.00_0.00_1.00_B_4.80_15.00_0.00_0.00_0.00",
        "SEN_LIN_cars=0300-0000(R)_t=1_levels[3]=(10-0-, 32-0-, 33-0-)_rebal=([1-8][tabu=00])[L(05)][P]_[05h,+30m+04h+60m]_match=15_0.10(S)_1.00_0.10_A_7.20_10.00_0.00_0.00_0.00_B_4.80_15.00_0.00_0.00_1.00",
        "SEN_LIN_cars=0300-0000(R)_t=1_levels[3]=(10-0-, 32-0-, 33-0-)_rebal=([1-8][tabu=00])[L(05)][P]_[05h,+30m+04h+60m]_match=15_0.10(S)_1.00_0.10_A_7.20_5.00_0.00_0.00_1.00_B_4.80_10.00_0.00_0.00_0.00",
        "SEN_LIN_cars=0300-0000(R)_t=1_levels[3]=(10-0-, 32-0-, 33-0-)_rebal=([1-8][tabu=00])[L(05)][P]_[05h,+30m+04h+60m]_match=15_0.10(S)_1.00_0.10_A_7.20_5.00_0.00_0.00_0.00_B_4.80_10.00_0.00_0.00_1.00",
        "SEN_LIN_cars=0300-0000(R)_t=1_levels[3]=(10-0-, 32-0-, 33-0-)_rebal=([1-8][tabu=00])[L(05)][P]_[05h,+30m+04h+60m]_match=15_0.10(S)_1.00_0.10_A_4.80_10.00_0.00_0.00_1.00_B_2.40_15.00_0.00_0.00_0.00",
        "SEN_LIN_cars=0300-0000(R)_t=1_levels[3]=(10-0-, 32-0-, 33-0-)_rebal=([1-8][tabu=00])[L(05)][P]_[05h,+30m+04h+60m]_match=15_0.10(S)_1.00_0.10_A_4.80_10.00_0.00_0.00_0.00_B_2.40_15.00_0.00_0.00_1.00",
        "SEN_LIN_cars=0300-0000(R)_t=1_levels[3]=(10-0-, 32-0-, 33-0-)_rebal=([1-8][tabu=00])[L(05)][P]_[05h,+30m+04h+60m]_match=15_0.10(S)_1.00_0.10_A_4.80_5.00_0.00_0.00_1.00_B_2.40_10.00_0.00_0.00_0.00",
        "SEN_LIN_cars=0300-0000(R)_t=1_levels[3]=(10-0-, 32-0-, 33-0-)_rebal=([1-8][tabu=00])[L(05)][P]_[05h,+30m+04h+60m]_match=15_0.10(S)_1.00_0.10_A_4.80_5.00_0.00_0.00_0.00_B_2.40_10.00_0.00_0.00_1.00",
    ]

    adhoc_compare["a"] = [
        "SL_LIN_cars=0300-0000(R)_t=1_levels[3]=(10-0-, 32-0-, 33-0-)_rebal=([1-8][tabu=00])[L(05)][P]_[05h,+30m+04h+60m]_match=15_0.10(S)_1.00_0.10_A_4.80_5.00_0.00_0.00_1.00_B_2.40_10.00_0.00_0.00_0.00",
        "SL_LIN_cars=0300-0000(R)_t=1_levels[3]=(10-0-, 32-0-, 33-0-)_rebal=([1-8][tabu=00])[L(05)][P]_[05h,+30m+04h+60m]_match=15_0.10(S)_1.00_0.10_A_4.80_5.00_0.00_4.80_1.00_B_2.40_10.00_0.00_2.40_0.00",
        "SL_LIN_cars=0300-0000(R)_t=1_levels[3]=(10-0-, 32-0-, 33-0-)_rebal=([1-8][tabu=00])[L(05)][P]_[05h,+30m+04h+60m]_match=15_0.10(S)_1.00_0.10_A_4.80_5.00_5.00_0.00_1.00_B_2.40_10.00_5.00_0.00_0.00",
        "SL_LIN_cars=0300-0000(R)_t=1_levels[3]=(10-0-, 32-0-, 33-0-)_rebal=([1-8][tabu=00])[L(05)][P]_[05h,+30m+04h+60m]_match=15_0.10(S)_1.00_0.10_A_4.80_5.00_5.00_4.80_1.00_B_2.40_10.00_5.00_2.40_0.00",
        "SL_LIN_cars=0300-0000(R)_t=1_levels[3]=(10-0-, 32-0-, 33-0-)_rebal=([1-8][tabu=00])[L(05)][P]_[05h,+30m+04h+60m]_match=15_0.10(S)_1.00_0.10_A_4.80_5.00_5.00_7.20_1.00_B_2.40_10.00_5.00_4.80_0.00",
        "SL_LIN_cars=0300-0000(R)_t=1_levels[3]=(10-0-, 32-0-, 33-0-)_rebal=([1-8][tabu=00])[L(05)][P]_[05h,+30m+04h+60m]_match=15_0.10(S)_1.00_0.10_A_4.80_5.00_5.00_9.60_1.00_B_2.40_10.00_5.00_7.20_0.00",
        "SL_LIN_cars=0300-0000(R)_t=1_levels[3]=(10-0-, 32-0-, 33-0-)_rebal=([1-8][tabu=00])[L(05)][P]_[05h,+30m+04h+60m]_match=15_0.10(S)_1.00_0.10_A_4.80_10.00_0.00_0.00_1.00_B_2.40_15.00_0.00_0.00_0.00",
    ]

    adhoc_compare_labels["a"] = [
        "5",
        "5+P",
        "5+5",
        "5+5+2P",
        "5+5+3P",
        "5+5+4P",
        "10",
    ]

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

    # ################################################################ #
    # Discount function ############################################## #
    # ################################################################ #

    adhoc_compare["penalize"] = [
        "base_LIN_V=0300-0000(R)_I=1_L[3]=(10-0-, 32-0-, 33-0-)_R=([0-8][L(05)]_T=[05h,+30m+04h+60m]_0.10(S)_1.00_0.10_A_4.80_5.00_0.00_0.00_0.00_B_2.40_10.00_0.00_0.00_1.00",
        "only1_LIN_cars=0300-0000(R)_t=1_levels[3]=(1-0, 3-300, 3-600)_rebal=([1-8][tabu=00])[L(05)]_[05h,+30m+04h+60m]_match=15_0.10(S)_1.00_0.10",
        "only1_LIN_cars=0300-0000(R)_t=1_levels[3]=(1-0, 3-300, 3-600)_rebal=([1-8][tabu=00])[L(05)][P]_[05h,+30m+04h+60m]_match=15_0.10(S)_1.00_0.10",
        "only1_LIN_cars=0300-0000(R)_t=1_levels[3]=(1-0, 3-300, 3-600)_rebal=([2-8][tabu=00])[L(05)]_[05h,+30m+04h+60m]_match=15_0.10(S)_1.00_0.10",
        "only1_LIN_cars=0300-0000(R)_t=1_levels[3]=(1-0, 3-300, 3-600)_rebal=([2-8][tabu=00])[L(05)][P]_[05h,+30m+04h+60m]_match=15_0.10(S)_1.00_0.10",
        "only1_LIN_cars=0300-0000(R)_t=1_levels[3]=(1-0, 3-300, 3-600)_rebal=([3-8][tabu=00])[L(05)]_[05h,+30m+04h+60m]_match=15_0.10(S)_1.00_0.10",
        "only1_LIN_cars=0300-0000(R)_t=1_levels[3]=(1-0, 3-300, 3-600)_rebal=([3-8][tabu=00])[L(05)][P]_[05h,+30m+04h+60m]_match=15_0.10(S)_1.00_0.10",
    ]

    adhoc_compare_labels["penalize"] = [
        "Adjacent neighbors (30s)",
        r"8 $\times$ RC1",
        r"8 $\times$ RC1 [P]",
        r"8 $\times$ RC5",
        r"8 $\times$ RC5 [P]",
        r"8 $\times$ RC10",
        r"8 $\times$ RC10 [P]",
    ]

    colors["penalize"] = ["k", "g", "g", "r", "r", "b", "b"]
    markers["penalize"] = [None, None, "o", None, "o", None, "o"]
    linewidth["penalize"] = [1, 1, 1, 1, 1, 1, 1]

    # ################################################################ #
    # Rebalance ###################################################### #
    # ################################################################ #
    adhoc_compare["rebalance"] = [
        "only1_LIN_cars=0300-0000(R)_t=1_levels[3]=(1-0, 3-300, 3-600)_rebal=([1-8][tabu=00])[L(05)][P]_[05h,+30m+04h+60m]_match=15_0.10(S)_1.00_0.10",
        "only1_LIN_cars=0300-0000(R)_t=1_levels[3]=(1-0, 3-300, 3-600)_rebal=([1-8, 2-4][tabu=00])[L(05)][P]_[05h,+30m+04h+60m]_match=15_0.10(S)_1.00_0.10",
        "only1_LIN_cars=0300-0000(R)_t=1_levels[3]=(1-0, 3-300, 3-600)_rebal=([1-8, 3-4][tabu=00])[L(05)][P]_[05h,+30m+04h+60m]_match=15_0.10(S)_1.00_0.10",
        "far_LIN_cars=0300-0000(R)_t=1_levels[3]=(1-0, 3-300, 3-600)_rebal=([1-8, 2-4, 3-2][tabu=00])[L(05)][P]_[05h,+30m+04h+60m]_match=15_0.10(S)_1.00_0.10",
    ]

    adhoc_compare_labels["rebalance"] = [
        r"8 $\times$ RC1",
        r"8 $\times$ RC1 + 4 $\times$ RC5",
        r"8 $\times$ RC1 + 4 $\times$ RC10",
        r"8 $\times$ RC1 + 4 $\times$ RC5 + 2 $\times$ RC10",
    ]

    linewidth["rebalance"] = [1, 1, 1, 1, 1, 1, 1]
    markers["rebalance"] = [None, "o", "x", "D"]

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

    # ################################################################ #
    # Max. number of cars ############################################ #
    # ################################################################ #

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
    colors["flood"] = ["r", "k", "g", "r"]
    linewidth["flood"] = [1, 1, 1, 1, 1, 1, 1]
    markers["flood"] = ["x", None, "o"]

    # ################################################################ #
    # Max. number of cars (unlimited)################################# #
    # ################################################################ #

    adhoc_compare["unlimited"] = [
        "only1_LIN_cars=0300-0000(R)_t=1_levels[3]=(1-0, 3-300, 3-600)_rebal=([1-8][tabu=00])[L(05)][P]_[05h,+30m+04h+60m]_match=15_0.10(S)_1.00_0.10",
        "only1_LIN_cars=0300-0000(R)_t=1_levels[3]=(1-0, 3-300, 3-600)_rebal=([1-8][tabu=00])[P]_[05h,+30m+04h+60m]_match=15_0.10(S)_1.00_0.10",
        "only1_LIN_cars=0300-0000(R)_t=1_levels[3]=(1-0, 3-300, 3-600)_rebal=([1-8, 2-4][tabu=00])[L(05)][P]_[05h,+30m+04h+60m]_match=15_0.10(S)_1.00_0.10",
        "only1_LIN_cars=0300-0000(R)_t=1_levels[3]=(1-0, 3-300, 3-600)_rebal=([1-8, 2-4][tabu=00])[P]_[05h,+30m+04h+60m]_match=15_0.10(S)_1.00_0.10",
    ]

    adhoc_compare_labels["unlimited"] = [
        r"8 $\times$ RC1",
        r"8 $\times$ RC1 (unlimited)",
        r"8 $\times$ RC1 + 4 $\times$ RC5",
        r"8 $\times$ RC1 + 4 $\times$ RC5 (unlimited)",
    ]

    colors["unlimited"] = ["k", "k", "r", "r"]
    linewidth["unlimited"] = [1, 1, 1, 1, 1, 1, 1]
    markers["unlimited"] = [None, "o", None, "o"]

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

    colors["policy"] = ["k", "r", "g"]
    markers["policy"] = [None, "x", "D"]

    colors["pavfav"] = ["k", "r"]

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

    # linewidth["penalize"] = [2, 2, 1, 2, 1, 2, 1]

    # linewidth["policy"] = [1, 1, 1, 1, 1, 1, 1]

    linewidth["pavfav"] = [1, 1, 1, 1, 1, 1, 1]
    markers["pavfav"] = [None, "o", "x"]

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

    legend_pos = dict()
    legend_pos["policy"] = "center right"

    SL = "Requests serviced"
    OF = "Objective function"
    TIME = "Time(s)"
    XLABEL = "Iteration"
    window = 50
    ITERATIONS = 1000

    markers_default = [None] * len(adhoc_compare[test_label])
    # markers = [None, "o", "*", "x", "|", None]

    shadow = False
    dpi = 1200
    d = dict()
    d_plot = defaultdict(list)

    for exp, sum_label in zip(
        adhoc_compare[test_label], adhoc_compare_labels[test_label]
    ):
        # folder = "O:/phd/output_paper/"
        # folder = conf.FOLDER_OUTPUT
        # path_all_stats = folder + exp + "/overall_stats.csv"
        # config_exp = ConfigNetwork()

        # Comparison is drawn from training
        path_all_stats = conf.FOLDER_OUTPUT + exp + "/adp/train/overall_stats.csv"
        print(sum_label, path_all_stats)
        config_exp = ConfigNetwork()
        

        try:
            # config_exp.load(folder + exp + "/exp_settings.json")
            config_exp.load(conf.FOLDER_OUTPUT + exp + "/exp_settings.json")
            df = pd.read_csv(path_all_stats)
        except Exception as e:
            print(f"Cannot load file!Exception: \"{e}\"")
            continue
        print(sum_label)
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
    yticks[OF] = np.linspace(0, 600, 8)
    yticks[SL] = np.linspace(0.5, 0.7, 5)

    # Policy
    # yticks[OF] = np.linspace(13000, 19000, 13)
    # yticks[SL] = np.linspace(0.5, 0.95, 10)

    # yticks[OF] = np.linspace(13000, 19000, 7)
    # yticks[SL] = np.linspace(0.5, 0.95, 8)

    yticks_labels[SL] = [f"{s:3.0%}" for s in yticks[SL]]
    yticks_labels[OF] = [f"{p:,.0f}" for p in yticks[OF]]
    yticks[TIME] = np.linspace(0, 300, 5)
    yticks_labels["Time(s)"] = np.linspace(0, 300, 5)
    df_outcome = pd.DataFrame(d)
    df_outcome = df_outcome[sorted(df_outcome.columns.values)]
    df_outcome.to_csv("outcome_tuning.csv", index=False)

    
    sns.set_context(context)
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
                    linewidth=linewidth.get(test_label, [2] * len(label_data))[
                        j
                    ],
                    marker=markers.get(test_label, markers_default)[j],
                    alpha=0.25,
                    label="",
                )
        cat, label_data = cat_label_data

        for j, ld in enumerate(label_data):
            label, sum_label, data = ld
            mavg = movingaverage(data, window)
            axs[i].plot(
                mavg,
                color=colors.get(test_label, colors_default)[j],
                linewidth=linewidth.get(test_label, [1] * len(label_data))[j],
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
        # title="Max. #cars/location"
    )

    # plt.show()
    print(f'Saving "{test_label}.{fig_format}"...')
    plt.savefig(f"{test_label}.{fig_format}", bbox_inches="tight", dpi=dpi)
