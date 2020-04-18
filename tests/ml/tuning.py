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
from mod.env.config import ConfigNetwork
import pandas as pd
from copy import deepcopy
import multiprocessing
from collections import defaultdict
import mod.util.log_util as la
from pprint import pprint
import itertools as it


def get_power_set(elements, keep_first=1, keep_last=2, n=None, max_size=None):
    if not n:
        n = len(elements)
    last = []
    first = []

    for k in range(1, keep_first + 1):
        first = first + list(it.combinations(elements[:keep_first], k))

    for k in range(1, keep_last + 1):
        last = last + list(it.combinations(elements[-keep_last:], k))

    power_set = set()
    for i in range(n + 1):
        a = [
            tuple(sorted(f) + sorted(x) + sorted(l))
            for x in list(it.combinations(elements[keep_first:-keep_last], i))
            for f in first
            for l in last
        ]
        power_set.update(a)
    if max_size:
        power_set = [s for s in power_set if len(s) <= max_size]

    # Sorted by length
    return sorted(power_set, key=lambda x: (len(x), x))


# Reward data for experiment
reward_data = defaultdict(dict)

ITERATIONS = 2


log_config = {
    la.LOG_DUALS: True,
    la.LOG_FLEET_ACTIVITY: True,
    la.LOG_VALUE_UPDATE: True,
    la.LOG_COSTS: True,
    la.LOG_SOLUTIONS: True,
    la.LOG_WEIGHTS: False,
    la.LOG_ALL: False,
    la.LOG_LEVEL: la.INFO,
    la.LEVEL_FILE: la.DEBUG,
    la.LEVEL_CONSOLE: la.INFO,
    la.FORMATTER_FILE: la.FORMATTER_TERSE,
}

myopic = False
policy_random = True

config_adp = {
    "classed_trips": True,
    # sq_guarantee=True,
    # universal_service=True,
    "log_config_dict": log_config,
    "log_mip": False,
    "save_plots": True,
    "linearize_integer_model": False,
    "use_artificial_duals": False,
    "save_df": False,
}


def test_all(
    tuning_labels, tuning_params, update_dict, all_settings, exp_list
):

    try:

        tuning_labels = deepcopy(tuning_labels)

        param = tuning_labels.pop(0)

        for e in tuning_params[param]:

            # Parameters work in tandem
            if isinstance(e, dict):
                update_dict = {**update_dict, **e}

            # Single update
            else:
                update_dict = {**update_dict, **{param: e}}

            test_all(
                tuning_labels,
                tuning_params,
                deepcopy(update_dict),
                all_settings,
                exp_list,
            )

    except:
        updated = deepcopy(all_settings)
        updated.update(update_dict)
        exp_list.append((all_settings.test_label, updated.label, updated))


def run_adp(exp):

    exp_name, label, exp_setup = exp

    if (
        exp_setup.myopic
        or exp_setup.test
        or exp_setup.policy_random
        or exp_setup.policy_reactive
    ):

        rows = exp_setup.iterations

        try:
            df = pd.read_csv(
                exp_setup.output_path + "overall_stats.csv",
                index_col="Episode",
            )

            rows = exp_setup.iterations - len(df.index)
            print(f'{rows} tests left to perform for instance "{exp_name}".')

        except Exception as e:
            print(f"No stats for '{exp_setup.label}'. Exception {e}")

        exp_setup.config[ConfigNetwork.ITERATIONS] = max(0, rows)

    reward_list = alg.alg_adp(None, exp_setup, **config_adp)

    return (exp_name, label, reward_list)


def multi_proc_exp(exp_list, processes=4, iterations=300):

    global reward_data

    pool = multiprocessing.Pool(processes=processes)

    results = pool.map(run_adp, exp_list)  # , chunksize=1)

    for exp_name, label, reward_list in results:

        reward_data[exp_name][label] = reward_list[:iterations]

        df = pd.DataFrame.from_dict(dict(reward_data[exp_name]))

        print(f"###################### Saving {(exp_name, label)}...")

        df.to_csv(f"tuning_{exp_name}.csv")


def main(test_labels, focus, N_PROCESSES, method):

    tuning_focus = dict()

    n = 7
    spatiotemporal_levels = [(0, i, 0, 0, 0, 0) for i in range(n)]
    power_set = get_power_set(
        spatiotemporal_levels, keep_first=1, n=2, keep_last=2, max_size=4
    )

    # BASE FARE SENSITIVITY ANALYSIS
    # Goal - Does your penalty mechanism really works? Or the same results
    # can be achieved my manipulating the base predicted_trips    # 1 - PAV baseline
    # 2 - PAV baseline + 2 x Base fares
    # 3 - PAV baseline + 3 x Base fares
    tuning_focus["sensitivity"] = {
        ConfigNetwork.TRIP_REJECTION_PENALTY: [(("A", 0), ("B", 0))],
        ConfigNetwork.N_CLOSEST_NEIGHBORS: [((1, 8),)],  # , ((1, 4),(2,4))],
        ConfigNetwork.TRIP_BASE_FARE: [
            (("A", 0), ("B", 2.4)),
            (("A", 0), ("B", 4.8)),
            # (("A", 0), ("B", 7.2)),
            (("A", 0), ("B", 9.6)),
            # (("A", 0), ("B", 12)),
        ],
        ConfigNetwork.TRIP_TOLERANCE_DELAY_MIN: [(("A", 0), ("B", 0))],
        ConfigNetwork.TRIP_MAX_PICKUP_DELAY: [
            (("A", 0), ("B", 5)),
            (("A", 0), ("B", 10)),
            (("A", 0), ("B", 15)),
        ],
        ConfigNetwork.TRIP_CLASS_PROPORTION: [
            # (("A", 1), ("B", 0)),
            (("A", 0), ("B", 1)),
        ],
    }

    # Changed sensitivity analsys to show the range of pk and fares instead
    # of focusing on the user bases.
    tuning_focus["sensitivity2"] = {
        ConfigNetwork.TRIP_REJECTION_PENALTY: [(("A", 0), ("B", 0))],
        ConfigNetwork.TRIP_TOLERANCE_DELAY_MIN: [(("A", 0), ("B", 0))],
        ConfigNetwork.N_CLOSEST_NEIGHBORS: [((1, 8),)],  # , ((1, 4),(2,4))],
        "SQ": [
            # Fare=(2.4 -> 12) - pk=5
            {
                ConfigNetwork.TRIP_BASE_FARE: (("A", 0), ("B", 2.4)),
                ConfigNetwork.TRIP_MAX_PICKUP_DELAY: (("A", 0), ("B", 5)),
            },
            {
                ConfigNetwork.TRIP_BASE_FARE: (("A", 0), ("B", 4.8)),
                ConfigNetwork.TRIP_MAX_PICKUP_DELAY: (("A", 0), ("B", 5)),
            },
            {
                ConfigNetwork.TRIP_BASE_FARE: (("A", 0), ("B", 7.2)),
                ConfigNetwork.TRIP_MAX_PICKUP_DELAY: (("A", 0), ("B", 5)),
            },
            {
                ConfigNetwork.TRIP_BASE_FARE: (("A", 0), ("B", 9.6)),
                ConfigNetwork.TRIP_MAX_PICKUP_DELAY: (("A", 0), ("B", 5)),
            },
            {
                ConfigNetwork.TRIP_BASE_FARE: (("A", 0), ("B", 12)),
                ConfigNetwork.TRIP_MAX_PICKUP_DELAY: (("A", 0), ("B", 5)),
            },
            # Fare=12 - pk=10
            {
                ConfigNetwork.TRIP_BASE_FARE: (("A", 0), ("B", 12)),
                ConfigNetwork.TRIP_MAX_PICKUP_DELAY: (("A", 0), ("B", 10)),
            },
            # Fare=15 - pk=15
            {
                ConfigNetwork.TRIP_BASE_FARE: (("A", 0), ("B", 12)),
                ConfigNetwork.TRIP_MAX_PICKUP_DELAY: (("A", 0), ("B", 15)),
            },
        ],
        ConfigNetwork.TRIP_CLASS_PROPORTION: [(("A", 0), ("B", 1)),],
    }

    # TEST METHODS
    # Goal - Test all methods in the same baseline instance.
    tuning_focus["methods"] = {
        ConfigNetwork.METHOD: [
            ConfigNetwork.METHOD_ADP_TRAIN,
            ConfigNetwork.METHOD_ADP_TEST,
            ConfigNetwork.METHOD_MYOPIC,
            ConfigNetwork.METHOD_RANDOM,
        ],
    }

    # TEST HIRING
    # Goal - Depot shares X FAV fleet size X Aggregation levels
    tuning_focus["hiring"] = {
        ConfigNetwork.DEPOT_SHARE: [1, 0.1, 0.01],
        "FLEET": [
            {
                ConfigNetwork.FLEET_SIZE: 300,
                ConfigNetwork.FAV_FLEET_SIZE: 200,
            },
        ],
        ConfigNetwork.MAX_CONTRACT_DURATION: [True, False],
        ConfigNetwork.N_CLOSEST_NEIGHBORS: [((1, 8),)],  # , ((1, 4),(2,4))],
        "CLASS_PROP": [
            {
                ConfigNetwork.TRIP_CLASS_PROPORTION: (("A", 1), ("B", 0)),
                ConfigNetwork.USE_CLASS_PROB: False,
            },
            {
                ConfigNetwork.TRIP_CLASS_PROPORTION: (("A", 0), ("B", 1)),
                ConfigNetwork.USE_CLASS_PROB: False,
            },
            {
                ConfigNetwork.TRIP_CLASS_PROPORTION: (("A", 0), ("B", 0)),
                ConfigNetwork.USE_CLASS_PROB: True,
            },
        ],
        # ConfigNetwork.FAV_AVAILABILITY_FEATURES:[
        #     (8, 1, 5, 9), (8, 1, 5, 9),
        # ],
        ConfigNetwork.AGGREGATION_LEVELS: [
            # [
            #     (1, 0, 0, "-", 0, 2),
            #     (1, 0, 0, "-", 0, 3),
            #     (3, 2, 0, "-", 0, "-"),
            #     (3, 3, 0, "-", 0, "-"),
            # ],
            # [
            #     (1, 0, 0, "-", 0, 2),
            #     (3, 2, 0, "-", 0, "-"),
            #     (3, 3, 0, "-", 0, "-"),
            # ],
            [
                (1, 0, 0, "-", 0, 2),
                (3, 2, 0, "-", 0, 3),
                (3, 3, 0, "-", 0, "-"),
            ],
            # [
            #     (1, 0, 0, "-", 0, 1),
            #     (3, 2, 0, "-", 0, 2),
            #     (3, 3, 0, "-", 0, 3)
            # ],
        ],
    }

    # TEST HIRING LAYERS
    # Goal - Aggregation levels that consider contracts
    tuning_focus["hiring_layer"] = {
        ConfigNetwork.DEPOT_SHARE: [1, 0.1, 0.01],
        "FLEET": [
            {
                ConfigNetwork.FLEET_SIZE: 300,
                ConfigNetwork.FAV_FLEET_SIZE: 200,
            },
        ],
        ConfigNetwork.MAX_CONTRACT_DURATION: [True, False],
        ConfigNetwork.N_CLOSEST_NEIGHBORS: [((1, 8),)],  # , ((1, 4),(2,4))],
        "CLASS_PROP": [
            {
                ConfigNetwork.TRIP_CLASS_PROPORTION: (("A", 1), ("B", 0)),
                ConfigNetwork.USE_CLASS_PROB: False,
            },
            {
                ConfigNetwork.TRIP_CLASS_PROPORTION: (("A", 0), ("B", 1)),
                ConfigNetwork.USE_CLASS_PROB: False,
            },
            {
                ConfigNetwork.TRIP_CLASS_PROPORTION: (("A", 0), ("B", 0)),
                ConfigNetwork.USE_CLASS_PROB: True,
            },
        ],
        # ConfigNetwork.FAV_AVAILABILITY_FEATURES:[
        #     (8, 1, 5, 9), (8, 1, 5, 9),
        # ],
        ConfigNetwork.AGGREGATION_LEVELS: [
            # [
            #     (1, 0, 0, "-", 0, 2),
            #     (1, 0, 0, "-", 0, 3),
            #     (3, 2, 0, "-", 0, "-"),
            #     (3, 3, 0, "-", 0, "-"),
            # ],
            # [
            #     (1, 0, 0, "-", 0, 2),
            #     (3, 2, 0, "-", 0, "-"),
            #     (3, 3, 0, "-", 0, "-"),
            # ],
            # contract 2 = 15
            # contract 3 = 60
            [(1, 0, 0, 2, 0, 2), (3, 2, 0, 3, 0, 3), (3, 3, 0, "-", 0, "-"),],
            # [
            #     (1, 0, 0, "-", 0, 1),
            #     (3, 2, 0, "-", 0, 2),
            #     (3, 3, 0, "-", 0, 3)
            # ],
        ],
    }

    # TEST HIRING WITH ONLY FAV fleet
    # Goal - Aggregation levels that consider contracts
    tuning_focus["hiring500"] = {
        ConfigNetwork.DEPOT_SHARE: [1, 0.1, 0.01, 0.005, 0.001],
        "FLEET": [
            {ConfigNetwork.FLEET_SIZE: 0, ConfigNetwork.FAV_FLEET_SIZE: 500,},
        ],
        ConfigNetwork.MAX_CONTRACT_DURATION: [False],
        ConfigNetwork.N_CLOSEST_NEIGHBORS: [((1, 8),)],  # , ((1, 4),(2,4))],
        "CLASS_PROP": [
            # {
            #     ConfigNetwork.TRIP_CLASS_PROPORTION: (("A", 1), ("B", 0)),
            #     ConfigNetwork.USE_CLASS_PROB: False,
            # },
            # {
            #     ConfigNetwork.TRIP_CLASS_PROPORTION: (("A", 0), ("B", 1)),
            #     ConfigNetwork.USE_CLASS_PROB: False,
            # },
            {
                ConfigNetwork.TRIP_CLASS_PROPORTION: (("A", 0), ("B", 0)),
                ConfigNetwork.USE_CLASS_PROB: True,
            },
        ],
        # ConfigNetwork.FAV_AVAILABILITY_FEATURES:[
        #     (8, 1, 5, 9), (8, 1, 5, 9),
        # ],
        ConfigNetwork.AGGREGATION_LEVELS: [
            # [
            #     (1, 0, 0, "-", 0, 2),
            #     (1, 0, 0, "-", 0, 3),
            #     (3, 2, 0, "-", 0, "-"),
            #     (3, 3, 0, "-", 0, "-"),
            # ],
            [
                (1, 0, 0, "-", 0, "-"),
                (3, 2, 0, "-", 0, "-"),
                (3, 3, 0, "-", 0, "-"),
            ],
            # contract 2 = 15
            # contract 3 = 60
            [(1, 0, 0, 2, 0, 2), (3, 2, 0, 3, 0, 3), (3, 3, 0, "-", 0, "-"),],
            [
                (1, 0, 0, 2, 0, 3),
                (1, 0, 0, 3, 0, "-"),
                (3, 2, 0, "-", 0, "-"),
                (3, 3, 0, "-", 0, "-"),
            ],
            [(3, 2, 0, 2, 0, 2), (3, 3, 0, 3, 0, 3), (3, 3, 0, "-", 0, "-"),],
            [
                (1, 0, 0, 1, 0, 2),
                (1, 0, 0, 2, 0, 3),
                (1, 0, 0, 3, 0, "-"),
                (3, 2, 0, "-", 0, "-"),
                (3, 3, 0, "-", 0, "-"),
            ],
            [
                (1, 0, 0, "-", 0, 2),
                (3, 2, 0, "-", 0, 3),
                (3, 3, 0, "-", 0, "-"),
            ],
            [
                (1, 0, 0, 3, 0, 3),
                (3, 2, 0, "-", 0, "-"),
                (3, 3, 0, "-", 0, "-"),
            ],
            # [
            #     (1, 0, 0, "-", 0, 1),
            #     (3, 2, 0, "-", 0, 2),
            #     (3, 3, 0, "-", 0, 3)
            # ],
        ],
    }

    # ENFORCING SERVICE LEVELS
    # Goal: What is the impact of each penalty mechanism in A and B?
    # 1 - PAV baseline
    # 2 - PAV baseline + tolerance delay (=larger pk time)
    # 3 - PAV baseline + tolerance delay + delay penalty
    # 4 - PAV baseline + tolerance delay + Delay penalty + rejection penalty
    tuning_focus["sl"] = {
        ConfigNetwork.N_CLOSEST_NEIGHBORS: [
            ((1, 8),),
            # ((1, 4), (2, 4))
        ],
        "TOLERANCE_MAX_PICKUP": [
            {
                ConfigNetwork.TRIP_TOLERANCE_DELAY_MIN: (("A", 0), ("B", 0)),
                ConfigNetwork.TRIP_MAX_PICKUP_DELAY: (("A", 10), ("B", 15)),
                ConfigNetwork.TRIP_REJECTION_PENALTY: (("A", 0), ("B", 0)),
            },
            # {
            #     ConfigNetwork.TRIP_TOLERANCE_DELAY_MIN: (("A", 0), ("B", 0)),
            #     ConfigNetwork.TRIP_MAX_PICKUP_DELAY: (("A", 5), ("B", 10)),
            #     ConfigNetwork.TRIP_REJECTION_PENALTY: (("A", 0), ("B", 0)),
            # },
            # {
            #     ConfigNetwork.TRIP_TOLERANCE_DELAY_MIN: (("A", 0), ("B", 0)),
            #     ConfigNetwork.TRIP_MAX_PICKUP_DELAY: (("A", 5), ("B", 10)),
            #     ConfigNetwork.TRIP_REJECTION_PENALTY: (("A", 4.8), ("B", 2.4)),
            # },
            # {
            #     ConfigNetwork.TRIP_TOLERANCE_DELAY_MIN: (("A", 5), ("B", 5)),
            #     ConfigNetwork.TRIP_MAX_PICKUP_DELAY: (("A", 5), ("B", 10)),
            #     ConfigNetwork.TRIP_REJECTION_PENALTY: (("A", 0), ("B", 0)),
            # },
            # {
            #     ConfigNetwork.TRIP_TOLERANCE_DELAY_MIN: (("A", 5), ("B", 5)),
            #     ConfigNetwork.TRIP_MAX_PICKUP_DELAY: (("A", 5), ("B", 10)),
            #     ConfigNetwork.TRIP_REJECTION_PENALTY: (("A", 4.8), ("B", 2.4)),
            # },
        ],
        ConfigNetwork.TRIP_CLASS_PROPORTION: [
            (("A", 1), ("B", 0)),
            (("A", 0), ("B", 1)),
        ],
    }

    # ENFORCING SERVICE LEVELS
    # Goal: What is the impact of each penalty mechanism in A and B?
    # 1 - PAV baseline
    # 2 - PAV baseline + tolerance delay (=larger pk time)
    # 3 - PAV baseline + tolerance delay + delay penalty
    # 4 - PAV baseline + tolerance delay + Delay penalty + rejection penalty
    tuning_focus["sl_pen"] = {
        ConfigNetwork.N_CLOSEST_NEIGHBORS: [
            ((1, 8),),
            # ((1, 4), (2, 4))
        ],
        ConfigNetwork.TRIP_REJECTION_PENALTY: [
            (("A", 0), ("B", 0)),
            (("A", 4.8), ("B", 2.4)),
            # (("A", 7.2), ("B", 4.8)),
            (("A", 9.6), ("B", 4.8)),
        ],
        "TOLERANCE_MAX_PICKUP": [
            # {
            #     ConfigNetwork.TRIP_TOLERANCE_DELAY_MIN: (("A", 0), ("B", 0)),
            #     ConfigNetwork.TRIP_MAX_PICKUP_DELAY: (("A", 10), ("B", 15)),
            # },
            {
                ConfigNetwork.TRIP_TOLERANCE_DELAY_MIN: (("A", 0), ("B", 0)),
                ConfigNetwork.TRIP_MAX_PICKUP_DELAY: (("A", 5), ("B", 10)),
            },
            {
                ConfigNetwork.TRIP_TOLERANCE_DELAY_MIN: (("A", 5), ("B", 5)),
                ConfigNetwork.TRIP_MAX_PICKUP_DELAY: (("A", 5), ("B", 10)),
            },
        ],
        ConfigNetwork.TRIP_CLASS_PROPORTION: [
            (("A", 1), ("B", 0)),
            (("A", 0), ("B", 1)),
        ],
    }

    # TUNING ADP
    tuning_focus["adp"] = {
        Config.STEPSIZE_RULE: [adp.STEPSIZE_MCCLAIN],
        Config.DISCOUNT_FACTOR: [0.7, 0.5, 0.3],
        # Config.STEPSIZE_CONSTANT: [0.1],
        # Config.HARMONIC_STEPSIZE: [1],
    }

    # TUNING ADP
    tuning_focus["adp15"] = {
        Config.STEPSIZE_RULE: [adp.STEPSIZE_MCCLAIN],
        Config.DISCOUNT_FACTOR: [1, 0.7, 0.5, 0.3, 0.1],
        # Config.STEPSIZE_CONSTANT: [0.1],
        # Config.HARMONIC_STEPSIZE: [1],
        ConfigNetwork.N_CLOSEST_NEIGHBORS: [
            ((1, 6), (2, 6)),
            ((0, 6), (1, 6)),
        ],
        ConfigNetwork.TRIP_MAX_PICKUP_DELAY: [
            (("A", 5), ("B", 15)),
            (("A", 5), ("B", 10)),
        ],
    }

    # TUNING CLASS PROPORTION
    # Use class probability (file) X random class probability
    tuning_focus["prob"] = {
        ConfigNetwork.N_CLOSEST_NEIGHBORS: [((1, 8),)],  # , ((1, 4),(2,4))],
        "CLASS_PROP": [
            # {
            #     ConfigNetwork.TRIP_CLASS_PROPORTION: (("A", 0.2), ("B", 0.8)),
            #     ConfigNetwork.USE_CLASS_PROB: False,
            # },
            {
                ConfigNetwork.TRIP_CLASS_PROPORTION: (("A", 0), ("B", 0)),
                ConfigNetwork.USE_CLASS_PROB: True,
            },
        ],
        Config.DISCOUNT_FACTOR: [0.2, 0.5, 0.7],
    }

    # # Config.FLEET_SIZE: [300],
    # Config.FLEET_START: [
    #     # conf.FLEET_START_LAST,
    #     # conf.FLEET_START_SAME,
    #     conf.FLEET_START_RANDOM
    # ],
    # ConfigNetwork.IDLE_ANNEALING: [0],
    # # -------------------------------------------------------- #
    # # DEMAND ################################################# #
    # # -------------------------------------------------------- #
    # "DEMAND_TW": [
    #     {Config.DEMAND_TOTAL_HOURS: 4, Config.DEMAND_EARLIEST_HOUR: 5},
    #     # {Config.DEMAND_TOTAL_HOURS: 4, Config.DEMAND_EARLIEST_HOUR: 9},
    # ],
    # Config.DEMAND_SAMPLING: [
    #     True,
    #     # False
    # ],
    # # Config.DEMAND_RESIZE_FACTOR: [0.1],
    # # Cars rebalance to up to #region centers at each level
    # Config.N_CLOSEST_NEIGHBORS: [
    #     ((0, 8), (1, 8)),
    #     # ((0, 4),),
    #     # ((0, 8),(4, 4)),
    #     # ((0, 8),(4, 4), (5, 1))
    # ],
    # Config.MAX_CARS_LINK: [5],
    # # Config.MAX_CARS_LINK: [None, 5, 10],
    pprint(tuning_focus)
    tuning_params = tuning_focus[focus]

    if method == "-train":
        m = ConfigNetwork.METHOD_ADP_TRAIN
        ITERATIONS = 500
    elif method == "-reactive":
        m = ConfigNetwork.METHOD_REACTIVE
        ITERATIONS = 51
    elif method == "-test":
        m = ConfigNetwork.METHOD_ADP_TEST
        ITERATIONS = 51
    else:
        m = ConfigNetwork.METHOD_ADP_TRAIN
        ITERATIONS = 500

    print(f"ITERATIONS: {ITERATIONS:04} - METHOD: {m}")

    shared_settings = {
        ConfigNetwork.ITERATIONS: ITERATIONS,
        ConfigNetwork.TEST_LABEL: test_label,
        ConfigNetwork.DISCOUNT_FACTOR: 0.1,
        ConfigNetwork.FLEET_SIZE: 200,
        # DEMAND ############################################# #
        ConfigNetwork.DEMAND_RESIZE_FACTOR: 0.1,
        ConfigNetwork.DEMAND_TOTAL_HOURS: 6,
        ConfigNetwork.DEMAND_EARLIEST_HOUR: 6,
        ConfigNetwork.OFFSET_TERMINATION_MIN: 30,
        ConfigNetwork.OFFSET_REPOSITIONING_MIN: 30,
        ConfigNetwork.TIME_INCREMENT: 5,
        ConfigNetwork.DEMAND_SAMPLING: True,
        # Service quality
        ConfigNetwork.MATCHING_DELAY: 15,
        ConfigNetwork.MAX_USER_BACKLOGGING_DELAY: False,
        ConfigNetwork.SQ_GUARANTEE: False,
        # ConfigNetwork.TRIP_REJECTION_PENALTY: {
        #     "A": 4.8,
        #     "B": 2.4,
        # },
        ConfigNetwork.TRIP_REJECTION_PENALTY: (("A", 4.8), ("B", 0)),
        ConfigNetwork.TRIP_BASE_FARE: (("A", 4.8), ("B", 2.4)),
        ConfigNetwork.TRIP_DISTANCE_RATE_KM: (("A", 1), ("B", 1)),
        ConfigNetwork.TRIP_TOLERANCE_DELAY_MIN: (("A", 5), ("B", 0)),
        ConfigNetwork.TRIP_MAX_PICKUP_DELAY: (("A", 5), ("B", 15)),
        ConfigNetwork.TRIP_CLASS_PROPORTION: (("A", 0), ("B", 1)),
        # ADP EXECUTION ###################################### #
        ConfigNetwork.METHOD: m,
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
        ConfigNetwork.PENALIZE_REBALANCE: True,
        ConfigNetwork.REACHABLE_NEIGHBORS: False,
        ConfigNetwork.N_CLOSEST_NEIGHBORS: (
            (1, 6),
            (2, 6),
            # (3, 3),
            # (3, 4),
        ),
        ConfigNetwork.CENTROID_LEVEL: 1,
        # FLEET ############################################## #
        # Car operation
        ConfigNetwork.MAX_CARS_LINK: 5,
        ConfigNetwork.MAX_IDLE_STEP_COUNT: None,
        ConfigNetwork.TIME_MAX_CARS_LINK: 5,
        # FAV configuration
        ConfigNetwork.DEPOT_SHARE: None,
        ConfigNetwork.FAV_DEPOT_LEVEL: None,
        ConfigNetwork.FAV_FLEET_SIZE: 0,
        ConfigNetwork.SEPARATE_FLEETS: False,
        ConfigNetwork.MAX_CONTRACT_DURATION: True,
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

    # Creating folders to log episodes
    if not os.path.exists(conf.FOLDER_TUNING):
        os.makedirs(conf.FOLDER_TUNING)

    conf.save_json(
        dict(tuning_settings=tuning_params, shared_settings=shared_settings),
        folder=conf.FOLDER_TUNING,
        file_name=test_label,
    )

    # Setup shared by all experiments
    setup = alg.get_sim_config(shared_settings)

    print("################ Initial setup")
    pprint(setup.config)

    tuning_labels = list(tuning_params.keys())

    # List with tuples (EXPERIMENT_NAME, FOLDER_NAME, SETTINGS)
    exp_list = []

    # Dictionary updated during recursion
    update_dict = {}

    # print("## Levels:")
    # for i, spatiotemporal_levels in enumerate(power_set):
    #     print(i, spatiotemporal_levels)

    test_all(tuning_labels, tuning_params, update_dict, setup, exp_list)

    print(f"################ All tests ({len(exp_list)})")
    for short_label, label, config in exp_list:
        print(" - ", label)

    multi_proc_exp(exp_list, processes=N_PROCESSES, iterations=ITERATIONS)

    return exp_list


def save_outcome_tuning(test_label, exp_list):
    """Read all stats from all test cases
    
    Parameters
    ----------
    exp_list : tuple
        (test_label, entire label, config obj)
    """
    exp_list = sorted(exp_list, key=lambda x: x[1])

    print(f"\n################ Experiment folders ({len(exp_list)}):")

    d = defaultdict(list)

    # Get all columns from all CSVs
    columns = [
        list(
            pd.read_csv(
                config_exp.output_path + "overall_stats.csv",
                index_col="Episode",
            ).columns
        )
        for _, label, config_exp in exp_list
    ]
    # Join columns
    columns = set(it.chain.from_iterable(columns))

    # Sort to keep consistency
    columns = sorted(columns)
    headers = [
        "label",
        "method",
        "pav",
        "fav",
        "station",
        "contract",
        "start",
        "time_increment",
        "levels",
        "reb_neigh",
        "max_link",
        "penalize",
        "earliest_hour",
        "repositioning",
        "total_hours",
        "termination_min",
        "resize_factor",
        "sample",
        "discount_factor",
        "stepsize_constant",
    ]
    indexes = []
    cols = set()
    for short_label, label, config_exp in exp_list:
        path_all_stats = config_exp.output_path + "overall_stats.csv"
        print(f'Saving at "{path_all_stats}".')
        df = pd.read_csv(path_all_stats, index_col="Episode")
        indexes.append(config_exp.label)

        d["label"].append(config_exp.sl_label)
        d["method"].append(config_exp.method)
        d["pav"].append(config_exp.fleet_size)
        d["fav"].append(config_exp.fav_fleet_size)
        d["station"].append(config_exp.label_stations)
        d["contract"].append(config_exp.label_max_contract)
        d["start"].append(config_exp.label_start)
        d["time_increment"].append(config_exp.time_increment)
        d["levels"].append(config_exp.label_levels)
        d["reb_neigh"].append(config_exp.label_reb_neigh)
        d["max_link"].append(config_exp.label_max_link)
        d["penalize"].append(config_exp.label_penalize)
        d["earliest_hour"].append(config_exp.demand_earliest_hour)
        d["repositioning"].append(config_exp.offset_repositioning_min)
        d["total_hours"].append(config_exp.demand_total_hours)
        d["termination_min"].append(config_exp.offset_termination_min)
        d["resize_factor"].append(config_exp.demand_resize_factor)
        d["sample"].append(config_exp.label_sample)
        d["discount_factor"].append(config_exp.discount_factor)
        d["stepsize_constant"].append(config_exp.stepsize_constant)
        for k, v in sorted(
            config_exp.sl_config_dict.items(), key=lambda kv: kv[0]
        ):
            cols.add(k)
            d[k].append(v)

        # Get the mean values of all columns
        for c in columns:
            if c in df.columns:
                # TODO if training, take only last line
                d[c].append(df[c].mean())
            else:
                d[c].append(0)

    df_outcome = pd.DataFrame(dict(d), index=indexes)
    # df_outcome = df_outcome[sorted(df_outcome.columns.values)]
    label = "myopic" if myopic else ""

    sorted_columns = headers + sorted(list(cols)) + columns
    # TODO config_exp is out of scope
    df_outcome.to_csv(
        f"{test_label}_{config_exp.method.replace('/','_')}_outcome_tuning.csv",
        columns=sorted_columns,
        index=True,
    )
    # except Exception as e:
    #     print(
    #         f"Could not save aggregated data (result still needs to be processed). Exception: {e}"
    #     )


if __name__ == "__main__":

    print({k: v for k, v in enumerate(sys.argv)})

    try:
        test_label = sys.argv[1]
        print("TEST LABEL:", test_label)
    except:
        test_label = "TUNE"

    try:
        N_PROCESSES = int(sys.argv[2])
    except:
        N_PROCESSES = 2

    try:
        method = sys.argv[3]
        print("METHOD:", method)
    except:
        print("Include method [-train, -test, -myopic, -reactive]")

    try:
        focus = sys.argv[4:]
        print("FOCUS:", focus)
    except:
        print("Include tuning focus [sensitivity, sl, adp]")

    exp_list = []
    for f in focus:
        print(f"---> Focus: {f}")
        exp_list.extend(main(test_label, f, N_PROCESSES, method))
    save_outcome_tuning(test_label, exp_list)
