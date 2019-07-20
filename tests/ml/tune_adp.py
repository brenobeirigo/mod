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

fleet_size = [30]
resize = [0.1]
discount_factor = [0.01, 0.05, 0.03]
rebalance_levels = [(1,)]
stepsize_constant = [0.1, 0.05]
scenarios = [conf.SCENARIO_NYC]
stepsize_rules = [adp.STEPSIZE_MCCLAIN]
harmonic_stepsize = [1]

iterations = 2
exp_name = "TUNING"


def run_exp(exp):

    global iterations

    exp_name, label, exp_setup = exp

    run_plot = PlotTrack(exp_setup)

    reward_list = alg.alg_adp(run_plot, exp_setup, episodes=iterations)

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

                            # for harm in harmonic_stepsize:

                            # # Harmonic constant is discarded when rule
                            # # is not harmonic
                            # if rule != adp.STEPSIZE_HARMONIC:
                            #     harm = 1

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

                            label = "_".join(
                                [
                                    f"{k}={str(v)}"
                                    for k, v in update_dict.items()
                                ]
                            )

                            updated = deepcopy(setup)
                            updated.update(update_dict)

                            exp_list.append((exp_name, label, updated))
    return exp_list


def multi_proc_exp(exp_list):

    global reward_data

    # with multiprocessing.Pool() as pool:
    pool = multiprocessing.Pool(processes=4)
    # pool.processes = 2
    results = pool.map(run_exp, exp_list, chunksize=1)

    for exp_name, label, reward_list in results:

        reward_data[exp_name][label] = reward_list

        df = pd.DataFrame.from_dict(reward_data[exp_name])

        # processed.add((exp_name, label))
        print(f"###################### Saving {(exp_name, label)}...")

        df.to_csv(f"tuning_{exp_name}.csv")


def proc(exp_list):
    # Setup a list of processes that we want to run

    n_workers = 3

    for i1 in range(0, len(exp_list), n_workers):
        processes = [
            multiprocessing.Process(target=run_exp, args=(exp_list[i], output))
            for i in range(i1, min(i1 + n_workers, len(exp_list)))
        ]

        # Run processes
        for p in processes:
            p.start()

        # Exit the completed processes
        for p in processes:
            p.join()

        results = [output.get() for p in processes]

        for exp_name, label, reward_list in results:

            reward_data[exp_name][label] = reward_list

            df = pd.DataFrame.from_dict(reward_data[exp_name])

            # processed.add((exp_name, label))
            print(f"###################### Saving {(exp_name, label)}...")

            df.to_csv(f"tuning_{exp_name}.csv")
            # np.save("tune.npy", processed)


if __name__ == "__main__":
    # try:
    #     processed = set(np.load("tune.npy", allow_pickle=True).item())
    #     print("Previous tuning loaded.")
    #     pprint(processed)
    # except:
    #     print("Starting tuning...")

    # Setup shared by all experiments
    setup = alg.get_sim_config(
        {
            Config.DEMAND_EARLIEST_HOUR: 5,
            Config.DEMAND_TOTAL_HOURS: 1,
            Config.OFFSET_REPOSIONING: 0,
            Config.OFFSET_TERMINATION: 0,
        }
    )

    print(setup.label)

    exp_list = get_exp(setup)
    # print(exp_list)
    # proc(exp_list)
    multi_proc_exp(exp_list)
    # for exp in exp_list:
    #     run_exp(exp)
