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

ITERATIONS = 500


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

config_adp = {
    "episodes": ITERATIONS,
    "classed_trips": True,
    # enable_hiring=True,
    # contract_duration_h=2,
    # sq_guarantee=True,
    # universal_service=True,
    "log_config_dict": log_config,
    "log_mip": False,
    "save_plots": True,
    "save_progress": 10,
    "linearize_integer_model": False,
    "use_artificial_duals": False,
    "save_df": False,
}


def test_all(tuning_labels, tuning_params, update_dict, all_settings, exp_list):

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

            test_all(tuning_labels, tuning_params, deepcopy(update_dict), all_settings, exp_list)

    except:

        updated = deepcopy(all_settings)
        updated.update(update_dict)
        exp_list.append((all_settings.test_label, updated.label, updated))


def run_adp(exp):

    exp_name, label, exp_setup = exp

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


def main():
    try:
        test_label = sys.argv[1]
    except:
        test_label = "TUNE"

    try:
        N_PROCESSES = int(sys.argv[2])
    except:
        N_PROCESSES = 2
    try:
        focus = sys.argv[3]
    except:
        print("Include tuning focus [sensitivity, sl, adp]")
        return

    tuning_focus = dict()

    n = 7
    spatiotemporal_levels = [(0, i, 0, 0, 0, 0) for i in range(n)]
    power_set = get_power_set(
        spatiotemporal_levels, keep_first=1, n=2, keep_last=2, max_size=4
    )
    # BASE FARE SENSITIVITY ANALYSIS
    # Goal - Does your penalty mechanism really works? Or the same results
    # can be achieved my manipulating the base fares?
    # 1 - PAV baseline
    # 2 - PAV baseline + 2 x Base fares
    # 3 - PAV baseline + 3 x Base fares
    tuning_focus["sensitivity"] = {
        ConfigNetwork.N_CLOSEST_NEIGHBORS: [((1, 8),)], #, ((1, 4),(2,4))],
        ConfigNetwork.TRIP_BASE_FARE: [(("A", 4.8), ("B", 2.4)), (("A", 7.2), ("B", 4.8)), (("A", 9.6), ("B", 7.2))],
        ConfigNetwork.TRIP_TOLERANCE_DELAY_MIN: [(("A", 0), ("B", 0)),],
        ConfigNetwork.TRIP_MAX_PICKUP_DELAY: [(("A", 5), ("B", 10)), (("A", 10), ("B", 15))],
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
    tuning_focus["sl"] = {
        ConfigNetwork.N_CLOSEST_NEIGHBORS: [((1, 8),), ((1, 4),(2,4))],
        ConfigNetwork.TRIP_REJECTION_PENALTY: [
            (("A", 0), ("B", 0)),
            (("A", 4.8), ("B", 2.4)),
        ],
        "TOLERANCE_MAX_PICKUP": [
            {
                ConfigNetwork.TRIP_TOLERANCE_DELAY_MIN: (("A", 0), ("B", 0)),
                ConfigNetwork.TRIP_MAX_PICKUP_DELAY: (("A", 10), ("B", 15)),
            },
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
        Config.DISCOUNT_FACTOR: [1],
        Config.STEPSIZE_CONSTANT: [0.1],
        Config.HARMONIC_STEPSIZE: [1],
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
    # Config.AGGREGATION_LEVELS:list(power_set),
    pprint(tuning_focus)
    tuning_params = tuning_focus[focus]

    shared_settings = {
        ConfigNetwork.TEST_LABEL: test_label,
        ConfigNetwork.DISCOUNT_FACTOR: 1,
        ConfigNetwork.FLEET_SIZE: 300,
        # DEMAND ############################################# #
        ConfigNetwork.DEMAND_RESIZE_FACTOR: 0.1,
        ConfigNetwork.DEMAND_TOTAL_HOURS: 4,
        ConfigNetwork.DEMAND_EARLIEST_HOUR: 5,
        ConfigNetwork.OFFSET_TERMINATION_MIN: 60,
        ConfigNetwork.OFFSET_REPOSITIONING_MIN: 30,
        ConfigNetwork.TIME_INCREMENT: 1,
        ConfigNetwork.DEMAND_SAMPLING: True,
        # Service quality
        ConfigNetwork.MATCHING_DELAY: 15,
        ConfigNetwork.ALLOW_USER_BACKLOGGING: False,
        ConfigNetwork.SQ_GUARANTEE: False,
        ConfigNetwork.TRIP_REJECTION_PENALTY: (("A", 0), ("B", 0)),
        ConfigNetwork.TRIP_BASE_FARE: (("A", 4.8), ("B", 2.4)),
        ConfigNetwork.TRIP_DISTANCE_RATE_KM: (("A", 1), ("B", 1)),
        ConfigNetwork.TRIP_TOLERANCE_DELAY_MIN: (("A", 0), ("B", 0)),
        ConfigNetwork.TRIP_MAX_PICKUP_DELAY: (("A", 5), ("B", 10)),
        ConfigNetwork.TRIP_CLASS_PROPORTION: (("A", 0), ("B", 1)),
        # ADP EXECUTION ###################################### #
        ConfigNetwork.MYOPIC: False,
        # Rebalance costs are ignored by MIP but included when
        # realizing the model
        ConfigNetwork.POLICY_RANDOM: False,
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

    exp_list = sorted(exp_list, key=lambda x: x[1])

    print(f"\n################ Experiment folders ({len(exp_list)}):")

    # After running all tuning instances, generates a file comparing them
    try:
        d = dict()
        for exp in exp_list:
            path_all_stats = conf.FOLDER_OUTPUT + exp[1] + "/overall_stats.csv"
            df = pd.read_csv(path_all_stats)
            spatiotemporal_levels = exp[2].get_levels()
            neighbors = exp[2].get_reb_neighbors()
            id_label = spatiotemporal_levels + neighbors
            d["reward_" + id_label] = df["Total reward"][:ITERATIONS]
            d["service_rate_" + id_label] = df["Service rate"][:ITERATIONS]
            d["time_" + id_label] = df["time"][:ITERATIONS]
            print(f" - {id_label}")

        df_outcome = pd.DataFrame(d)
        df_outcome = df_outcome[sorted(df_outcome.columns.values)]
        df_outcome.to_csv("outcome_tuning.csv", index=False)

    except Exception as e:
        print(f"Could not save aggregated data (result still needs to be processed). Exception: {e}")

    multi_proc_exp(exp_list, processes=N_PROCESSES, iterations=ITERATIONS)


if __name__ == "__main__":

   main()
