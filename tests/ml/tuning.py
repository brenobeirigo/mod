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

# Reward data for experiment
reward_data = dict()

N_PROCESSES = 2

ITERATIONS = 50

from pprint import pprint


def test_all(
    tuning_labels, tuning_params, update_dict, all_settings, exp_list
):

    try:

        param = tuning_labels.pop(0)

        for e in tuning_params[param]:

            # Parameters work in tandem
            if isinstance(e, dict):
                update_dict.update(e)
            # Single update
            else:
                update_dict[param] = e

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

    run_plot = PlotTrack(exp_setup)

    reward_list = alg.alg_adp(
        run_plot,
        exp_setup,
        False,
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

        df = pd.DataFrame.from_dict(reward_data[exp_name])

        print(f"###################### Saving {(exp_name, label)}...")

        df.to_csv(f"tuning_{exp_name}.csv")


if __name__ == "__main__":

    tuning_params = {
        Config.STEPSIZE_RULE: [adp.STEPSIZE_MCCLAIN],
        Config.DISCOUNT_FACTOR: [0.05],
        Config.STEPSIZE_CONSTANT: [0.1],
        Config.HARMONIC_STEPSIZE: [1],
        Config.FLEET_SIZE: [300],
        Config.DEMAND_SCENARIO: [conf.SCENARIO_NYC],
        Config.DEMAND_RESIZE_FACTOR: [0.1],
        "REBAL_NEIGHBORS": [
            {
                Config.REBALANCE_LEVEL: (0, 1),
                Config.N_CLOSEST_NEIGHBORS: (8, 8),
            },
            {Config.REBALANCE_LEVEL: (1,), Config.N_CLOSEST_NEIGHBORS: (8,)},
        ],
        Config.AGGREGATION_LEVELS: [
            [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (0, 5)],
            [(0, 0), (0, 1), (0, 2), (0, 3)],
        ],
    }

    # Setup shared by all experiments
    setup = alg.get_sim_config(
        {
            Config.TEST_LABEL: "TEST",
            Config.DEMAND_EARLIEST_HOUR: 5,
            Config.DEMAND_TOTAL_HOURS: 4,
            Config.OFFSET_REPOSIONING: 15,
            Config.OFFSET_TERMINATION: 30,
            Config.CONTRACT_DURATION_LEVEL: 10,
            Config.LEVEL_DIST_LIST: [0, 60, 90, 120, 180, 270],
            Config.LEVEL_TIME_LIST: [1, 2, 4],
        }
    )

    tuning_labels = list(tuning_params.keys())

    exp_list = []

    test_all(tuning_labels, tuning_params, {}, setup, exp_list)

    pprint(exp_list)

    multi_proc_exp(exp_list, processes=N_PROCESSES)
