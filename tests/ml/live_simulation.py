import os
import sys

# Adding project folder to import modules
root = os.getcwd().replace("\\", "/")
sys.path.append(root)

from mod.env.simulator import PlotTrack
import mod.ml.adp_network_server as alg
import mod.env.config as conf

# best_link_LIN_cars=0500(L)_t=0.5_levels[3]=(3-0, 3-300, 3-600)_rebal=(0-4)[P(10)]_[05h,+15m+04h+30m]_0.10(S)_1.00_0.10
start_config = alg.get_sim_config(
    {
        conf.Config.TEST_LABEL: "best_link",
        conf.Config.DISCOUNT_FACTOR: 1,
        conf.Config.PENALIZE_REBALANCE: True,
        conf.Config.FLEET_SIZE: 500,
        conf.Config.DEMAND_RESIZE_FACTOR: 0.1,
        conf.Config.DEMAND_TOTAL_HOURS: 4,
        conf.Config.DEMAND_EARLIEST_HOUR: 5,
        conf.Config.OFFSET_TERMINATION_MIN: 30,
        conf.Config.OFFSET_REPOSITIONING_MIN: 15,
        conf.Config.DEMAND_SAMPLING: True,
        conf.Config.LEVEL_TIME_LIST: [3],
        conf.Config.LEVEL_DIST_LIST: [0, 60, 180, 300, 600, 1200],
        conf.Config.LINEARIZE_INTEGER_MODEL: False,
        conf.Config.SQ_GUARANTEE: False,
        conf.Config.MAX_CARS_LINK: 10,
        conf.Config.USE_ARTIFICIAL_DUALS: False,
        conf.Config.N_CLOSEST_NEIGHBORS: ((0, 4),),
        conf.Config.AGGREGATION_LEVELS: [
            (0, 0, 0, 0, 0, 0),
            (0, 1, 0, 0, 0, 0),
            (0, 2, 0, 0, 0, 0),
        ],
    }
)
run_plot = PlotTrack(start_config)
run_plot.start_animation(alg.alg_adp)
