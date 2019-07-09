import os
import sys

# Adding project folder to import modules
root = os.getcwd().replace("\\", "/")
sys.path.append(root)

from mod.env.simulator import PlotTrack
import mod.ml.adp_network_server as alg
import mod.env.adp.adp as adp
from mod.env.config import Config
import pandas as pd


if __name__ == "__main__":

    setup = alg.get_sim_config()
    setup.update(
        {
            Config.FLEET_SIZE: 50,
            Config.DEMAND_EARLIEST_HOUR: 9,
            Config.DEMAND_TOTAL_HOURS: 1,
        }
    )

    discount_factor = [0.1, 0.5, 0.7, 0.1]
    stepsize_constant = [0.05, 0.1]
    harmonic_stepsize = [2, 3]
    n = 3
    reward_data = dict()

    # Tune stepsize rules
    for dist in discount_factor:
        for rule in adp.STEPSIZE_RULES:
            for harm in harmonic_stepsize:

                # Discard harmonic
                if rule != adp.STEPSIZE_HARMONIC:
                    harm = -1

                for step in stepsize_constant:

                    update_dict = {
                        Config.STEPSIZE_RULE: rule,
                        Config.DISCOUNT_FACTOR: dist,
                        Config.STEPSIZE_CONSTANT: step,
                        Config.HARMONIC_STEPSIZE: harm,
                    }

                    label = "_".join(
                        [f"{k:>}={v}" for k, v in update_dict.items()]
                    )

                    if label in reward_data:
                        continue

                    setup.update(update_dict)
                    run_plot = PlotTrack(setup)

                    reward_list = alg.sim(run_plot, setup, episodes=n)

                    # Setting become a column in the dataframe

                    reward_data[label] = reward_list

                    df = pd.DataFrame.from_dict(reward_data)

                    df.to_csv("tuning.csv")

