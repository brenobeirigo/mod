import os
import sys

# Adding project folder to import modules
root = os.getcwd().replace("\\", "/")
sys.path.append(root)

from mod.env.simulator import PlotTrack
import mod.ml.adp_network_server as alg

start_config = alg.get_sim_config({})
run_plot = PlotTrack(start_config)
run_plot.start_animation(alg.alg_adp)
