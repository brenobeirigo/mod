import os
import sys

# Adding project folder to import modules
root = os.getcwd().replace("\\", "/")
sys.path.append(root)

from mod.env.simulator import PlotTrack
import mod.ml.adp_network_server as alg
import mod.env.config as config

start_config = alg.get_sim_config(
    {
        config.Config.TEST_LABEL: "SIM",
        config.Config.FLEET_SIZE: 1500,
        config.Config.DEMAND_RESIZE_FACTOR: 1,
    }
)
run_plot = PlotTrack(start_config)
run_plot.start_animation(alg.alg_adp)
