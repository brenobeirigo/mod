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


if __name__ == "__main__":

    exp_name = ""

    setup = alg.get_sim_config()
    setup.update(
        {
            Config.DEMAND_EARLIEST_HOUR: 5,
            Config.DEMAND_TOTAL_HOURS: 5,
        }
    )

    fleet_size = [50, 300, 500, 1000]

    discount_factor = [0.1, 0.5, 0.7, 1]
    stepsize_constant = [0.05, 0.1]
    harmonic_stepsize = [1]
    stepsize_rules = adp.STEPSIZE_RULES
    stepsize_rules.remove(adp.STEPSIZE_HARMONIC)
    n = 3

    i = 0

    # Tune stepsize rules
    for sc in [conf.SCENARIO_NYC, conf.SCENARIO_UNBALANCED]:
        for reb_level in [(1,), (1,2), (1,2,3)]:
            
            # File names
            exp_name = f"{sc}{reb_level}"

            # Reward data for experiment
            reward_data = dict()

            for f_size in fleet_size:
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
                                Config.TEST_LABEL: f"{exp_name}{i:04}",
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

                            if label in reward_data:
                                continue

                            setup.update(update_dict)
                            run_plot = PlotTrack(setup)

                            reward_list = alg.alg_adp(run_plot, setup, episodes=n)

                            # Setting become a column in the dataframe

                            reward_data[label] = reward_list

                            df = pd.DataFrame.from_dict(reward_data)

                            df.to_csv(f"tuning_{exp_name}.csv")

