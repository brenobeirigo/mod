import os
import sys

# Adding project folder to import modules
root = os.getcwd().replace("\\", "/")
sys.path.append(root)

from mod.env.simulator import PlotTrack
import mod.ml.adp_network_server as alg
import mod.env.adp.adp as adp
import mod.env.config as conf
from  mod.env.config import Config
import pandas as pd
from copy import deepcopy
from collections import defaultdict
import multiprocessing
import numpy as np

# Reward data for experiment
reward_data = dict()

output = multiprocessing.Queue()

setup = alg.get_sim_config()
setup.update(
    {
        Config.DEMAND_EARLIEST_HOUR: 7,
        Config.DEMAND_TOTAL_HOURS: 2,
    }
)

lock = multiprocessing.Lock()

fleet_size = [300, 500, 1000]
discount_factor = [0.1, 0.5, 0.7]
rebalance_levels = [(1,), (1,2), (1,2,3)]
stepsize_constant = [0.05, 0.1]
scenarios = [conf.SCENARIO_UNBALANCED, conf.SCENARIO_NYC]
stepsize_rules = adp.STEPSIZE_RULES
stepsize_rules.remove(adp.STEPSIZE_HARMONIC)
harmonic_stepsize = [1]

iterations = 30
overall_exp_label = "M"

processed = set()

def run_exp(exp, output):

    exp_name, label, update_dict = exp

    exp_setup = deepcopy(setup)
    exp_setup.update(update_dict)

    run_plot = PlotTrack(exp_setup)
    reward_list = alg.alg_adp(run_plot, exp_setup, episodes=iterations)

    # Setting become a column in the dataframe
    output.put((exp_name, label, reward_list))
   

def get_exp():
    global reward_data

    exp_list = []

    for reb_level in rebalance_levels:
        for sc in scenarios:
            for f_size in fleet_size:
                
                # File names
                exp_name = f"{overall_exp_label}_{sc}_{reb_level}_{f_size}"

                reward_data[exp_name] = dict()

                # Hyperparam index
                i = 0

                # Tune stepsize rules
                for dist in discount_factor:
                    for rule in stepsize_rules:
                        for step in stepsize_constant:
                            
                            
                            
                            # for harm in harmonic_stepsize:

                            # # Harmonic constant is discarded when rule
                            # # is not harmonic
                            # if rule != adp.STEPSIZE_HARMONIC:
                            #     harm = 1
                                
                            # Experiment id
                            i+=1

                            update_dict = {
                                Config.TEST_LABEL: f"{exp_name}_STEP_{i:04}",
                                Config.STEPSIZE_RULE: rule,
                                Config.DISCOUNT_FACTOR: dist,
                                Config.STEPSIZE_CONSTANT: step,
                                # Config.HARMONIC_STEPSIZE: harm,
                                Config.FLEET_SIZE: f_size,
                                Config.DEMAND_SCENARIO: sc,
                                Config.REBALANCE_LEVEL: reb_level,
                            }

                            label = "_".join(
                                [f"{k}={str(v)}" for k, v in update_dict.items()]
                            )

                            if (exp_name, label) in processed:
                                continue

                            exp_list.append((exp_name, label, update_dict))
    return exp_list

def multi_proc_exp(exp_list):
    #with multiprocessing.Pool() as pool:
    pool = multiprocessing.Pool(processes=4)
    # pool.processes = 2   
    pool.map(run_exp, exp_list, chunksize=1)

def proc(exp_list):
    # Setup a list of processes that we want to run
    
    n_workers = 4

    for i1 in range(0, len(exp_list), n_workers):
        processes = [
            multiprocessing.Process(
                target=run_exp,
                args=(exp_list[i], output)
            ) for i in range(i1, min(i1+n_workers, len(exp_list)))
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
            processed.add((exp_name, label))
            print(f"###################### Saving {(exp_name, label)}...")
            df.to_csv(f"tuning_{exp_name}.csv")
            np.save("tune.npy", processed)
        

if __name__ == "__main__":
    try:
        processed = set(np.load("tune.npy").item())
        print("Previous tuning loaded.")
        pprint(processed)
    except:
        print("Starting tuning...")


    exp_list = get_exp()
    proc(exp_list)
    # multi_proc_exp(exp_list)
    # for exp in exp_list:
    #     run_exp(exp)
