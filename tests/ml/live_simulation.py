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
import mod.util.log_util as la
from pprint import pprint

# ## LOAD CONFIGURATION ############################################## #
# Load data from tests previously executed
start_config = ConfigNetwork.load(
    "C:/Users/LocalAdmin/OneDrive/leap_forward/phd_project/reb/code/mod/data/output/"
    "DF2_LIN_V=0500-0000(R)_I=5_L[4]=(10-0-, 11-0-, 12-0-, 13-0-)_R=([1-6, 2-6, 3-3][L(05)]_T=[06h,+60m+06h+30m]_1.00(S)_1.00_0.10_A_4.80_5.00_5.00_4.80_0.00_B_2.40_15.00_0.00_0.00_1.00"
    "/exp_settings.json"
)

# After loading, add another iteration to play
start_config.config[ConfigNetwork.ITERATIONS] = (
    start_config.config[ConfigNetwork.ITERATIONS] + 1
)

# ## CREATE CONFIGURATION ############################################ #

n_iterations = 300
test_label = "SM"
fleet_size = 300
method = ConfigNetwork.METHOD_ADP_TRAIN
save_progress_interval = True
log_adp = False
log_level = la.INFO

start_config2 = alg.get_sim_config(
    {
        ConfigNetwork.ITERATIONS: n_iterations,
        ConfigNetwork.TEST_LABEL: test_label,
        ConfigNetwork.DISCOUNT_FACTOR: 1,
        ConfigNetwork.FLEET_SIZE: fleet_size,
        # DEMAND ############################################# #
        ConfigNetwork.DEMAND_RESIZE_FACTOR: 0.1,
        ConfigNetwork.DEMAND_TOTAL_HOURS: 6,
        ConfigNetwork.DEMAND_EARLIEST_HOUR: 6,
        ConfigNetwork.OFFSET_TERMINATION_MIN: 60,
        ConfigNetwork.OFFSET_REPOSITIONING_MIN: 30,
        ConfigNetwork.TIME_INCREMENT: 1,
        ConfigNetwork.DEMAND_SAMPLING: True,
        # Service quality
        ConfigNetwork.MATCHING_DELAY: 15,
        ConfigNetwork.ALLOW_USER_BACKLOGGING: False,
        ConfigNetwork.SQ_GUARANTEE: False,
        # ConfigNetwork.TRIP_REJECTION_PENALTY: {
        #     "A": 4.8,
        #     "B": 2.4,
        # },
        ConfigNetwork.TRIP_REJECTION_PENALTY: (("A", 4.8), ("B", 2.4)),
        ConfigNetwork.TRIP_BASE_FARE: (("A", 4.8), ("B", 2.4)),
        ConfigNetwork.TRIP_DISTANCE_RATE_KM: (("A", 1), ("B", 1)),
        ConfigNetwork.TRIP_TOLERANCE_DELAY_MIN: (("A", 5), ("B", 5)),
        ConfigNetwork.TRIP_MAX_PICKUP_DELAY: (("A", 5), ("B", 10)),
        ConfigNetwork.TRIP_CLASS_PROPORTION: (("A", 0.2), ("B", 0.8)),
        # ADP EXECUTION ###################################### #
        ConfigNetwork.METHOD: method,
        ConfigNetwork.SAVE_PROGRESS: save_progress_interval,
        ConfigNetwork.ADP_IGNORE_ZEROS: True,
        ConfigNetwork.LINEARIZE_INTEGER_MODEL: False,
        ConfigNetwork.USE_ARTIFICIAL_DUALS: False,
        # EXPLORATION ######################################## #
        # ANNEALING + THOMPSON
        # If zero, cars increasingly gain the right of stay
        # still. This obliges them to rebalance consistently.
        # If None, disabled.
        ConfigNetwork.IDLE_ANNEALING: None,
        ConfigNetwork.ACTIVATE_THOMPSON: False,
        ConfigNetwork.MAX_TARGETS: 6,
        # When cars start in the last visited point, the model takes
        # a long time to figure out the best time
        ConfigNetwork.FLEET_START: conf.FLEET_START_RANDOM,
        ConfigNetwork.CAR_SIZE_TABU: 0,
        # If REACHABLE_NEIGHBORS is True, then PENALIZE_REBALANCE
        # is False
        ConfigNetwork.PENALIZE_REBALANCE: False,
        ConfigNetwork.REACHABLE_NEIGHBORS: False,
        ConfigNetwork.N_CLOSEST_NEIGHBORS: (
            (1, 8),
            # (1, 4),
            # (2, 4),
            # (3, 4),
        ),
        # FLEET ############################################## #
        # Car operation
        ConfigNetwork.MAX_CARS_LINK: 5,
        ConfigNetwork.MAX_IDLE_STEP_COUNT: None,
        ConfigNetwork.TIME_MAX_CARS_LINK: 5,
        # FAV configuration
        ConfigNetwork.DEPOT_SHARE: 0.01,
        ConfigNetwork.FAV_DEPOT_LEVEL: None,
        ConfigNetwork.FAV_FLEET_SIZE: 0,
        ConfigNetwork.SEPARATE_FLEETS: False,
        ConfigNetwork.MAX_CONTRACT_DURATION: False,
        # ConfigNetwork.PARKING_RATE_MIN = 1.50/60 # 1.50/h
        # ConfigNetwork.PARKING_RATE_MIN = 0.1*20/60
        # ,  # = rebalancing 1 min
        ConfigNetwork.PARKING_RATE_MIN: 0,  # = rebalancing 1 min
        # Saving
        ConfigNetwork.USE_SHORT_PATH: False,
        ConfigNetwork.SAVE_TRIP_DATA: False,
        ConfigNetwork.SAVE_FLEET_DATA: False,
        # Load 1st class probabilities dictionary
        ConfigNetwork.USE_CLASS_PROB: False,
    }
)

run_plot = PlotTrack(start_config)
run_plot.start_animation(alg.alg_adp)
