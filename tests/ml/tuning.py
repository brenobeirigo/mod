import os
import sys


# Adding project folder to import modules
root = os.getcwd().replace("\\", "/")
sys.path.append(root)

from mod.env.simulator import PlotTrack
import mod.ml.adp_network_server as alg
import mod.env.adp.adp as adp
import mod.env.config as conf
from mod.env.config import Config
import pandas as pd
from copy import deepcopy
import multiprocessing
from collections import defaultdict
import mod.util.log_util as la
from pprint import pprint
import itertools as it


def get_power_set(elements, keep_first=1, keep_last=2, n=None, max_size=None):
    if not n:
        n = len(elements)
    last = []
    first = []

    for k in range(1, keep_first + 1):
        first = first + list(it.combinations(elements[:keep_first], k))

    for k in range(1, keep_last + 1):
        last = last + list(it.combinations(elements[-keep_last:], k))

    power_set = set()
    for i in range(n + 1):
        a = [
            tuple(sorted(f) + sorted(x) + sorted(l))
            for x in list(it.combinations(elements[keep_first:-keep_last], i))
            for f in first
            for l in last
        ]
        power_set.update(a)
    if max_size:
        power_set = [s for s in power_set if len(s) <= max_size]

    # Sorted by length
    return sorted(power_set, key=lambda x: (len(x), x))


# Reward data for experiment
reward_data = defaultdict(dict)

ITERATIONS = 200


log_config = {
    la.LOG_DUALS: True,
    la.LOG_FLEET_ACTIVITY: True,
    la.LOG_VALUE_UPDATE: True,
    la.LOG_COSTS: True,
    la.LOG_SOLUTIONS: True,
    la.LOG_WEIGHTS: False,
    la.LOG_ALL: False,
    la.LOG_LEVEL: la.INFO,
    la.LEVEL_FILE: la.DEBUG,
    la.LEVEL_CONSOLE: la.INFO,
    la.FORMATTER_FILE: la.FORMATTER_TERSE,
}

config_adp = {
    "episodes": ITERATIONS,
    "classed_trips": True,
    # enable_hiring=True,
    # contract_duration_h=2,
    # sq_guarantee=True,
    # universal_service=True,
    "log_config_dict": log_config,
    "log_mip": False,
    "save_plots": False,
    "save_progress": True,
    "linearize_integer_model": False,
    "use_artificial_duals": False,
    "use_duals": True,
    "save_df": False,
    "is_myopic": False,
}


def test_all(
    tuning_labels, tuning_params, update_dict, all_settings, exp_list
):

    try:

        tuning_labels = deepcopy(tuning_labels)

        param = tuning_labels.pop(0)

        for e in tuning_params[param]:

            # Parameters work in tandem
            if isinstance(e, dict):
                update_dict = {**update_dict, **e}

            # Single update
            else:
                update_dict = {**update_dict, **{param: e}}

            test_all(
                tuning_labels,
                tuning_params,
                update_dict,
                all_settings,
                exp_list,
            )

    except:

        updated = deepcopy(all_settings)
        updated.update(update_dict)
        exp_list.append((all_settings.test_label, updated.label, updated))


def run_adp(exp):

    exp_name, label, exp_setup = exp

    reward_list = alg.alg_adp(None, exp_setup, **config_adp)

    return (exp_name, label, reward_list)


def multi_proc_exp(exp_list, processes=4, iterations=300):

    global reward_data

    pool = multiprocessing.Pool(processes=processes)

    results = pool.map(run_adp, exp_list)  # , chunksize=1)

    for exp_name, label, reward_list in results:

        reward_data[exp_name][label] = reward_list[:iterations]

        df = pd.DataFrame.from_dict(dict(reward_data[exp_name]))

        print(f"###################### Saving {(exp_name, label)}...")

        df.to_csv(f"tuning_{exp_name}.csv")


if __name__ == "__main__":

    try:
        test_label = sys.argv[1]
    except:
        test_label = "TUNE"

    try:
        N_PROCESSES = int(sys.argv[2])
    except:
        N_PROCESSES = 2

    n = 7
    spatiotemporal_levels = [(0, i, 0, 0, 0, 0) for i in range(n)]
    # print("Levels:")
    # pprint(levels)
    power_set = get_power_set(
        spatiotemporal_levels, keep_first=1, n=2, keep_last=2, max_size=4
    )

    tuning_params = {
        Config.STEPSIZE_RULE: [adp.STEPSIZE_MCCLAIN],
        Config.DISCOUNT_FACTOR: [1],
        Config.STEPSIZE_CONSTANT: [0.1],
        Config.HARMONIC_STEPSIZE: [1],
        Config.FLEET_SIZE: [300],
        Config.FLEET_START: [
            # conf.FLEET_START_LAST,
            # conf.FLEET_START_SAME,
            conf.FLEET_START_RANDOM,
        ],
        # -------------------------------------------------------- #
        # DEMAND ################################################# #
        # -------------------------------------------------------- #
        "DEMAND_TW": [
            {Config.DEMAND_TOTAL_HOURS: 4, Config.DEMAND_EARLIEST_HOUR: 5},
            # {Config.DEMAND_TOTAL_HOURS: 4, Config.DEMAND_EARLIEST_HOUR: 9},
        ],
        Config.DEMAND_SAMPLING: [
            True,
            # False
        ],
        Config.DEMAND_RESIZE_FACTOR: [0.1],
        # Cars rebalance to up to #region centers at each level
        Config.N_CLOSEST_NEIGHBORS: [
            ((0, 8),),
            # ((0, 4),),
            # ((0, 8),(4, 4)),
            # ((0, 8),(4, 4), (5, 1))
        ],
        Config.MAX_CARS_LINK: [5],
        Config.CAR_SIZE_TABU: [20, 30],
        # Config.MAX_CARS_LINK: [None, 5, 10],
        Config.AGGREGATION_LEVELS: [
            # [(2, 0, 0, 0, 0, 0), (2, 4, 0, 0, 0, 0), (2, 5, 0, 0, 0, 0)],
            #[(3, 0, 0, 0, 0, 0), (3, 2, 0, 0, 0, 0), (3, 3, 0, 0, 0, 0)],
            [(1, 0, 0, 0, 0, 0), (3, 2, 0, 0, 0, 0), (3, 3, 0, 0, 0, 0)],
            #[(1, 0, 0, 0, 0, 0), (1, 2, 0, 0, 0, 0), (1, 3, 0, 0, 0, 0)],
            # [(1, 0, 0, 0, 0, 0), (1, 1, 0, 0, 0, 0), (1, 2, 0, 0, 0, 0), (1, 3, 0, 0, 0, 0)],
            # [
            #     (3, 0, 0, 0, 0, 0),
            #     (3, 1, 0, 0, 0, 0),
            #     (3, 2, 0, 0, 0, 0),
            #     (3, 3, 0, 0, 0, 0),
            # ]
            # [(5, 0, 0, 0, 0, 0), (5, 4, 0, 0, 0, 0), (5, 5, 0, 0, 0, 0)],
            # [(6, 0, 0, 0, 0, 0), (6, 4, 0, 0, 0, 0), (6, 5, 0, 0, 0, 0)],
            # [(7, 0, 0, 0, 0, 0), (7, 4, 0, 0, 0, 0), (7, 5, 0, 0, 0, 0)],
            # [(0, 0, 0, 0, 0, 0), (4, 4, 0, 0, 0, 0), (4, 5, 0, 0, 0, 0)],
            # [(1, 0, 0, 0, 0, 0), (1, 4, 0, 0, 0, 0), (1, 5, 0, 0, 0, 0)],
            # [(3, 0, 0, 0, 0, 0), (3, 4, 0, 0, 0, 0), (3, 5, 0, 0, 0, 0)],
            # [(4, 0, 0, 0, 0, 0), (4, 4, 0, 0, 0, 0), (4, 5, 0, 0, 0, 0)],
            # [(0, 0, 0, 0, 0, 0), (2, 4, 0, 0, 0, 0), (4, 5, 0, 0, 0, 0)],
            # [(0, 0, 0, 0, 0, 0), (0, 4, 0, 0, 0, 0), (0, 5, 0, 0, 0, 0)],
            # [(0, 0, 0, 0, 0, 0), (0, 4, 0, 0, 0, 0), (0, 5, 0, 0, 0, 0)],
            # [
            #     (0, 0, 0, 0, 0, 0),
            #     (0, 1, 0, 0, 0, 0),
            #     (0, 4, 0, 0, 0, 0),
            #     (0, 6, 0, 0, 0, 0),
            # ],
            # [
            #     (0, 0, 0, 0, 0, 0),
            #     (0, 4, 0, 0, 0, 0),
            #     (0, 5, 0, 0, 0, 0),
            #     (0, 6, 0, 0, 0, 0),
            # ],
            # ############# 0.5 minutes
            # [(0, 0, 0, 0, 0, 0), (0, 4, 0, 0, 0, 0), (0, 5, 0, 0, 0, 0)],
            # [(0, 0, 0, 0, 0, 0), (0, 4, 0, 0, 0, 0), (0, 5, 0, 0, 0, 0)],
            # [
            #     (0, 0, 0, 0, 0, 0),
            #     (0, 1, 0, 0, 0, 0),
            #     (0, 4, 0, 0, 0, 0),
            #     (0, 6, 0, 0, 0, 0),
            # ],
            # [
            #     (0, 0, 0, 0, 0, 0),
            #     (0, 4, 0, 0, 0, 0),
            #     (0, 5, 0, 0, 0, 0),
            #     (0, 6, 0, 0, 0, 0),
            # ],
            # # ############# 1 all minutes
            # [(1, 0, 0, 0, 0, 0), (1, 4, 0, 0, 0, 0), (1, 5, 0, 0, 0, 0)],
            # [(1, 0, 0, 0, 0, 0), (1, 4, 0, 0, 0, 0), (1, 5, 0, 0, 0, 0)],
            # [
            #     (1, 0, 0, 0, 0, 0),
            #     (1, 1, 0, 0, 0, 0),
            #     (1, 4, 0, 0, 0, 0),
            #     (1, 6, 0, 0, 0, 0),
            # ],
            # [
            #     (1, 0, 0, 0, 0, 0),
            #     (1, 4, 0, 0, 0, 0),
            #     (1, 5, 0, 0, 0, 0),
            #     (1, 6, 0, 0, 0, 0),
            # ],
            # # ############# 3 minutes
            # [(0, 0, 0, 0, 0, 0), (3, 4, 0, 0, 0, 0), (3, 5, 0, 0, 0, 0)],
            # [(0, 0, 0, 0, 0, 0), (3, 4, 0, 0, 0, 0), (3, 6, 0, 0, 0, 0)],
            # [
            #     (0, 0, 0, 0, 0, 0),
            #     (3, 1, 0, 0, 0, 0),
            #     (3, 4, 0, 0, 0, 0),
            #     (3, 6, 0, 0, 0, 0),
            # ],
            # [
            #     (0, 0, 0, 0, 0, 0),
            #     (3, 4, 0, 0, 0, 0),
            #     (3, 5, 0, 0, 0, 0),
            #     (3, 6, 0, 0, 0, 0),
            # ],
            # # ############# 5 minutes
            # [(0, 0, 0, 0, 0, 0), (4, 4, 0, 0, 0, 0), (4, 5, 0, 0, 0, 0)],
            # [(0, 0, 0, 0, 0, 0), (4, 4, 0, 0, 0, 0), (4, 6, 0, 0, 0, 0)],
            # [
            #     (0, 0, 0, 0, 0, 0),
            #     (4, 1, 0, 0, 0, 0),
            #     (4, 4, 0, 0, 0, 0),
            #     (4, 6, 0, 0, 0, 0),
            # ],
            # [
            #     (0, 0, 0, 0, 0, 0),
            #     (4, 4, 0, 0, 0, 0),
            #     (4, 5, 0, 0, 0, 0),
            #     (4, 6, 0, 0, 0, 0),
            # ],
        ]
        # list(power_set),
        #     [(0, 0, 0, 0, 0, 0), (0, 2, 0, 0, 0, 0), (0, 6, 0, 0, 0, 0), (0, 11, 0, 0, 0, 0)],
        #     [(0, 0, 0, 0, 0, 0), (0, 2, 0, 0, 0, 0), (0, 6, 0, 0, 0, 0),(0, 9, 0, 0, 0, 0), (0, 11, 0, 0, 0, 0)],
        #     [(0, 0, 0, 0, 0, 0), (0, 2, 0, 0, 0, 0), (0, 6, 0, 0, 0, 0),(0, 9, 0, 0, 0, 0)],
        #     [(0, 0, 0, 0, 0, 0), (0, 2, 0, 0, 0, 0), (0, 9, 0, 0, 0, 0)],
        #     [(0, 0, 0, 0, 0, 0), (0, 6, 0, 0, 0, 0), (0, 9, 0, 0, 0, 0)],
        #     [(0, 0, 0, 0, 0, 0), (0, 11, 0, 0, 0, 0)],
        #     [(0, 0, 0, 0, 0, 0)],
        #     [(0, 0, 0, 0, 0, 0), (0, 1, 0, 0, 0, 0)],
        #     [(0, 0, 0, 0, 0, 0), (0, 1, 0, 0, 0, 0), (0, 2, 0, 0, 0, 0)],
        #     [(0, 0, 0, 0, 0, 0), (0, 1, 0, 0, 0, 0), (0, 2, 0, 0, 0, 0), (0, 3, 0, 0, 0, 0)],
        #     [(0, 0, 0, 0, 0, 0), (0, 1, 0, 0, 0, 0), (0, 2, 0, 0, 0, 0), (0, 3, 0, 0, 0, 0), (0, 4, 0, 0, 0, 0)],
        #     [(0, 0, 0, 0, 0, 0), (0, 1, 0, 0, 0, 0), (0, 2, 0, 0, 0, 0), (0, 3, 0, 0, 0, 0), (0, 4, 0, 0, 0, 0), (0, 5, 0, 0, 0, 0)],
        #     [
        #         (0, 0, 0, 0, 0, 0),
        #         (0, 1, 0, 0, 0, 0),
        #         (0, 2, 0, 0, 0, 0),
        #         (0, 3, 0, 0, 0, 0),
        #         (0, 4, 0, 0, 0, 0),
        #         (0, 5, 0, 0, 0, 0),
        #         (0, 6, 0, 0, 0, 0),
        #     ],
        #     [(0, 0, 0, 0, 0, 0), (0, 1, 0, 0, 0, 0), (0, 2, 0, 0, 0, 0), (0, 3, 0, 0, 0, 0), (0, 4, 0, 0, 0, 0), (0, 5, 0, 0, 0, 0), (0, 6, 0, 0, 0, 0),  (0, 7, 0, 0, 0, 0)],
        #     [(0, 0, 0, 0, 0, 0), (0, 1, 0, 0, 0, 0), (0, 2, 0, 0, 0, 0), (0, 3, 0, 0, 0, 0), (0, 4, 0, 0, 0, 0), (0, 5, 0, 0, 0, 0), (0, 6, 0, 0, 0, 0),  (0, 7, 0, 0, 0, 0),  (0, 8, 0, 0, 0, 0)],
        #     [(0, 0, 0, 0, 0, 0), (0, 1, 0, 0, 0, 0), (0, 2, 0, 0, 0, 0), (0, 3, 0, 0, 0, 0), (0, 4, 0, 0, 0, 0), (0, 5, 0, 0, 0, 0), (0, 6, 0, 0, 0, 0),  (0, 7, 0, 0, 0, 0),  (0, 8, 0, 0, 0, 0),  (0, 9, 0, 0, 0, 0)],
        #     [(0, 0, 0, 0, 0, 0), (0, 1, 0, 0, 0, 0), (0, 2, 0, 0, 0, 0), (0, 3, 0, 0, 0, 0), (0, 4, 0, 0, 0, 0), (0, 5, 0, 0, 0, 0), (0, 6, 0, 0, 0, 0),  (0, 7, 0, 0, 0, 0),  (0, 8, 0, 0, 0, 0),  (0, 9, 0, 0, 0, 0), (0, 10, 0, 0, 0, 0)],
        #     [(0, 0, 0, 0, 0, 0), (0, 1, 0, 0, 0, 0), (0, 2, 0, 0, 0, 0), (0, 3, 0, 0, 0, 0), (0, 4, 0, 0, 0, 0), (0, 5, 0, 0, 0, 0), (0, 6, 0, 0, 0, 0),  (0, 7, 0, 0, 0, 0),  (0, 8, 0, 0, 0, 0),  (0, 9, 0, 0, 0, 0), (0, 10, 0, 0, 0, 0), (0, 11, 0, 0, 0, 0)],
        #     [(0, 0, 0, 0, 0, 0), (0, 1, 0, 0, 0, 0), (0, 2, 0, 0, 0, 0), (0, 3, 0, 0, 0, 0), (0, 4, 0, 0, 0, 0), (0, 5, 0, 0, 0, 0), (0, 6, 0, 0, 0, 0),  (0, 7, 0, 0, 0, 0),  (0, 8, 0, 0, 0, 0),  (0, 9, 0, 0, 0, 0)],
        # ]
    }

    shared_settings = {
        Config.TEST_LABEL: test_label,
        Config.DISCOUNT_FACTOR: 1,
        Config.PENALIZE_REBALANCE: True,
        Config.FLEET_SIZE: 300,
        # 10 steps = 5 min
        Config.TIME_MAX_CARS_LINK: 5,
        Config.DEMAND_RESIZE_FACTOR: 0.1,
        Config.DEMAND_TOTAL_HOURS: 5,
        Config.DEMAND_EARLIEST_HOUR: 5,
        Config.TIME_INCREMENT: 1,
        Config.OFFSET_TERMINATION_MIN: 60,
        Config.OFFSET_REPOSITIONING_MIN: 30,
        Config.DEMAND_SAMPLING: True,
        Config.LEVEL_TIME_LIST: [0.5, 1, 2.5, 3, 5, 10],
        Config.LEVEL_DIST_LIST: [0, 60, 300, 600],
        Config.LINEARIZE_INTEGER_MODEL: False,
        Config.SQ_GUARANTEE: False,
        Config.USE_ARTIFICIAL_DUALS: False,
        Config.MATCHING_DELAY: 15,
        Config.ALLOW_USER_BACKLOGGING: False,
        Config.REACHABLE_NEIGHBORS: False,
    }

    # Creating folders to log episodes
    if not os.path.exists(conf.FOLDER_TUNING):
        os.makedirs(conf.FOLDER_TUNING)

    conf.save_json(
        dict(tuning_settings=tuning_params, shared_settings=shared_settings),
        folder=conf.FOLDER_TUNING,
        file_name=test_label,
    )

    # Setup shared by all experiments
    setup = alg.get_sim_config(shared_settings)

    print("################ Initial setup")
    pprint(setup.config)

    tuning_labels = list(tuning_params.keys())

    # List with tuples (EXPERIMENT_NAME, FOLDER_NAME, SETTINGS)
    exp_list = []

    # Dictionary updated during recursion
    update_dict = {}

    # print("## Levels:")
    # for i, spatiotemporal_levels in enumerate(power_set):
    #     print(i, spatiotemporal_levels)

    test_all(tuning_labels, tuning_params, update_dict, setup, exp_list)

    exp_list = sorted(exp_list, key=lambda x: x[1])

    print("\n################ Experiment folders:")

    try:
        d = dict()
        for exp in exp_list:
            path_all_stats = conf.FOLDER_OUTPUT + exp[1] + "/overall_stats.csv"
            df = pd.read_csv(path_all_stats)
            spatiotemporal_levels = exp[2].get_levels()
            neighbors = exp[2].get_reb_neighbors()
            id_label = spatiotemporal_levels + neighbors
            d["reward_" + id_label] = df["Total reward"][:ITERATIONS]
            d["service_rate_" + id_label] = df["Service rate"][:ITERATIONS]
            d["time_" + id_label] = df["time"][:ITERATIONS]
            print(f" - {id_label}")

        df_outcome = pd.DataFrame(d)
        df_outcome = df_outcome[sorted(df_outcome.columns.values)]
        df_outcome.to_csv("outcome_tuning.csv", index=False)

    except Exception as e:
        print(f"Could not save aggregated data. Exception: {e}")

    multi_proc_exp(exp_list, processes=N_PROCESSES, iterations=ITERATIONS)
