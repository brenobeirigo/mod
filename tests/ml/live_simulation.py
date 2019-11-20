import os
import sys

# Adding project folder to import modules
root = os.getcwd().replace("\\", "/")
sys.path.append(root)

from mod.env.simulator import PlotTrack
import mod.ml.adp_network_server as alg
from mod.env.config import ConfigNetwork
import mod.env.config as conf
from mod.env.trip import ClassedTrip

# best_link_LIN_cars=0500(L)_t=0.5_levels[3]=(3-0, 3-300, 3-600)_rebal=(0-4)[P(10)]_[05h,+15m+04h+30m]_0.10(S)_1.00_0.10
start_config = alg.get_sim_config(
    {
        ConfigNetwork.TEST_LABEL: "VAIKRAI",
        ConfigNetwork.DISCOUNT_FACTOR: 1,
        ConfigNetwork.PENALIZE_REBALANCE: True,
        ConfigNetwork.FLEET_SIZE: 300,
        ConfigNetwork.DEMAND_RESIZE_FACTOR: 0.1,
        ConfigNetwork.DEMAND_TOTAL_HOURS: 4,
        ConfigNetwork.DEMAND_EARLIEST_HOUR: 5,
        ConfigNetwork.OFFSET_TERMINATION_MIN: 60,
        ConfigNetwork.OFFSET_REPOSITIONING_MIN: 30,
        ConfigNetwork.TIME_INCREMENT: 1,
        ConfigNetwork.DEMAND_SAMPLING: True,
        ConfigNetwork.SQ_GUARANTEE: False,
        ConfigNetwork.MAX_CARS_LINK: 5,
        # 10 steps = 5 min
        ConfigNetwork.TIME_MAX_CARS_LINK: 5,
        ConfigNetwork.LINEARIZE_INTEGER_MODEL: False,
        ConfigNetwork.USE_ARTIFICIAL_DUALS: False,
        # Controlling user matching
        ConfigNetwork.MATCHING_DELAY: 15,
        ConfigNetwork.ALLOW_USER_BACKLOGGING: False,
        ConfigNetwork.MAX_IDLE_STEP_COUNT: None,
        # If zero, cars increasingly gain the right of stay
        # still. This obliges them to rebalance consistently.
        ConfigNetwork.IDLE_ANNEALING: None,
        # When cars start in the last visited point, the model takes
        # a long time to figure out the best time
        ConfigNetwork.FLEET_START: conf.FLEET_START_LAST,
        ConfigNetwork.CAR_SIZE_TABU: 10,
        ConfigNetwork.REACHABLE_NEIGHBORS: False,
        ConfigNetwork.ADP_IGNORE_ZEROS: True,
        ConfigNetwork.DEPOT_SHARE: None,
        ConfigNetwork.FAV_DEPOT_LEVEL: 2,
        ConfigNetwork.FAV_FLEET_SIZE: 0,
        ConfigNetwork.SEPARATE_FLEETS: False,
        ConfigNetwork.MYOPIC: False,
        # ConfigNetwork.PARKING_RATE_MIN = 1.50/60 # 1.50/h
        # ConfigNetwork.PARKING_RATE_MIN = 0.1*20/60,  # = rebalancing 1 min
        ConfigNetwork.PARKING_RATE_MIN: 0,  # = rebalancing 1 min
    }
)

ClassedTrip.q_classes = dict(A=1.0, B=0.9)
ClassedTrip.sq_level_class = dict(A=[0, 0], B=[0, 0])
# min_max_time_class = dict(A=dict(min=3, max=3), B=dict(min=3, max=6))
ClassedTrip.min_max_time_class = dict(
    A=dict(min=1, max=3), B=dict(min=4, max=9)
)
ClassedTrip.class_proportion = dict(A=0.0, B=1)

run_plot = PlotTrack(start_config)
run_plot.start_animation(alg.alg_adp)
