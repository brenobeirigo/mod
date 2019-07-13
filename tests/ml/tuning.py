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

output = multiprocessing.Queue()

fleet_size = [300]
resize = [0.1]
discount_factor = [0.1]
rebalance_levels = [(1, 2)]
stepsize_constant = [0.1]
scenarios = [conf.SCENARIO_NYC]
stepsize_rules = [adp.STEPSIZE_MCCLAIN]
harmonic_stepsize = [1]

iterations = 30

exp_name = "TUNING_123"


def run_adp(exp):

    global iterations

    exp_name, label, exp_setup = exp

    run_plot = PlotTrack(exp_setup)

    reward_list = alg.alg_adp(run_plot, exp_setup, False, episodes=iterations)

    return (exp_name, label, reward_list)


def get_exp(setup):

    global reward_data

    exp_list = []

    for reb_level in rebalance_levels:
        for sc in scenarios:
            for f_size in fleet_size:

                reward_data[exp_name] = dict()

                # Tune stepsize rules
                for dist in discount_factor:
                    for rule in stepsize_rules:
                        for step in stepsize_constant:

                            update_dict = {
                                Config.TEST_LABEL: exp_name,
                                # Config.STEPSIZE_RULE: rule,
                                Config.DISCOUNT_FACTOR: dist,
                                Config.STEPSIZE_CONSTANT: step,
                                # Config.HARMONIC_STEPSIZE: harm,
                                Config.FLEET_SIZE: f_size,
                                Config.DEMAND_SCENARIO: sc,
                                Config.REBALANCE_LEVEL: reb_level,
                            }

                            updated = deepcopy(setup)
                            updated.update(update_dict)

                            exp_list.append((exp_name, updated.label, updated))
    return exp_list


def multi_proc_exp(exp_list):

    global reward_data

    pool = multiprocessing.Pool(processes=4)

    results = pool.map(run_adp, exp_list)  # , chunksize=1)

    for exp_name, label, reward_list in results:

        reward_data[exp_name][label] = reward_list

        df = pd.DataFrame.from_dict(reward_data[exp_name])

        print(f"###################### Saving {(exp_name, label)}...")

        df.to_csv(f"tuning_{exp_name}.csv")


if __name__ == "__main__":

    # Setup shared by all experiments
    setup = alg.get_sim_config(
        {
            Config.DEMAND_EARLIEST_HOUR: 5,
            Config.DEMAND_TOTAL_HOURS: 4,
            Config.OFFSET_REPOSIONING: 30,
            Config.OFFSET_TERMINATION: 60,
            # Config.AGGREGATION_LEVELS: 7,
            # Config.LEVEL_DIST_LIST: [0, 30, 60, 90, 120, 180, 270, 750, 1140],
        }
    )

    print(setup.label)
    exp_list = get_exp(setup)
    multi_proc_exp(exp_list)
