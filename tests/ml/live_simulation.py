import os
import sys

# Adding project folder to import modules
root = os.getcwd().replace("\\", "/")
sys.path.append(root)

from mod.env.simulator import PlotTrack
import mod.ml.adp_network_server as alg
import mod.env.config as conf

start_config = alg.get_sim_config(
    {
        conf.Config.TEST_LABEL: "S",
        conf.Config.FLEET_SIZE: 30,
        conf.Config.DEMAND_TOTAL_HOURS: 1,
        conf.Config.DEMAND_RESIZE_FACTOR: 0.1,
    }
)
run_plot = PlotTrack(start_config)
run_plot.start_animation(alg.alg_adp)
