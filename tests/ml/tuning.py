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

# Reward data for experiment
reward_data = defaultdict(dict)

ITERATIONS = 100

from pprint import pprint


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

    reward_list = alg.alg_adp(
        None,
        exp_setup,
        episodes=ITERATIONS,
        classed_trips=True,
        # enable_hiring=True,
        # contract_duration_h=2,
        # sq_guarantee=True,
        # universal_service=True,
    )

    return (exp_name, label, reward_list)


def multi_proc_exp(exp_list, processes=4):

    global reward_data

    pool = multiprocessing.Pool(processes=processes)

    results = pool.map(run_adp, exp_list)  # , chunksize=1)

    for exp_name, label, reward_list in results:

        reward_data[exp_name][label] = reward_list

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

    tuning_params = {
        Config.STEPSIZE_RULE: [adp.STEPSIZE_MCCLAIN],
        Config.DISCOUNT_FACTOR: [0.05],
        Config.STEPSIZE_CONSTANT: [0.1],
        Config.HARMONIC_STEPSIZE: [1],
        Config.FLEET_SIZE: [
            # 300,
            400
        ],
        Config.FLEET_START: [
            conf.FLEET_START_LAST,
            # conf.FLEET_START_SAME,
            # conf.FLEET_START_RANDOM,
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
        "REBAL_LEVELS_NEIGHBORS": [
            {
                Config.REBALANCE_LEVEL: (0, 1, 2, 3, 4, 5, 6),
                Config.N_CLOSEST_NEIGHBORS: (4, 4, 4, 2, 2, 1, 1),
            },
            # {
            #     Config.REBALANCE_LEVEL: (0, 1),
            #     Config.N_CLOSEST_NEIGHBORS: (8, 8),
            # },
        ],
        Config.AGGREGATION_LEVELS: [
            [
                (0, 0),
                (1, 0),
                (2, 0),
                (2, 1),
                (2, 2),
                (2, 3),
                (2, 4),
                (2, 5),
                (2, 6),
            ],
            # [(0, 0), (1, 0), (2, 0), (2, 1), (2, 2), (2, 3), (2, 4), (2, 5), (2, 6)],
            [(0, 0), (1, 0), (1, 1), (1, 2), (1, 3), (1, 4), (1, 5), (1, 6)],
        ],
    }

    shared_settings = {
        Config.TEST_LABEL: test_label,
        Config.OFFSET_REPOSIONING: 15,
        Config.OFFSET_TERMINATION: 30,
        Config.CONTRACT_DURATION_LEVEL: 10,
        Config.LEVEL_DIST_LIST: [0, 30, 60, 120, 150, 240, 600],
        Config.LEVEL_TIME_LIST: [1, 2, 4, 8],
    }

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

    test_all(tuning_labels, tuning_params, update_dict, setup, exp_list)

    exp_list = sorted(exp_list, key=lambda x: x[1])

    print("\n################ Experiment folders:")
    for exp in exp_list:
        print(f" - {exp[1]}")

    multi_proc_exp(exp_list, processes=N_PROCESSES)
