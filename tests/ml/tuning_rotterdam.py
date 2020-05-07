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
    la.LOG_DUALS: False,
    la.LOG_FLEET_ACTIVITY: False,
    la.LOG_VALUE_UPDATE: False,
    la.LOG_COSTS: False,
    la.LOG_SOLUTIONS: False,
    la.LOG_WEIGHTS: False,
    la.LOG_MIP: False,
    la.LOG_TIMES: False,
    la.LOG_ALL: False,
    la.LOG_DECISION_INFO: False,
    la.LOG_STEP_SUMMARY: False,
    la.LOG_LEVEL: False,
    la.LOG_ALL: False,
    la.LOG_LEVEL: la.INFO,
    la.LEVEL_FILE: la.DEBUG,
    la.LEVEL_CONSOLE: la.INFO,
    la.FORMATTER_FILE: la.FORMATTER_TERSE,
}

myopic = False
policy_random = True

config_adp = {
    "log_config_dict": log_config,
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

    tuning_focus["rotterdam"] = {
        ConfigNetwork.CASE_STUDY: [
            # "N08Z06CD02",
            # "N08Z06CD04",
            "N08Z06SD02",
            "N08Z06SD04",
            # "N08Z07CD02",
            # "N08Z07CD04",
            "N08Z07SD02",
            "N08Z07SD04",
            # "N08Z08CD02",
            # "N08Z08CD04",
            "N08Z08SD02",
            "N08Z08SD04",
            # "N08Z09CD02",
            # "N08Z09CD04",
            "N08Z09SD02",
            "N08Z09SD04",
            # "N08Z10CD02",
            # "N08Z10CD04",
            "N08Z10SD02",
            "N08Z10SD04",
        ],
        # ConfigNetwork.TRIP_TOLERANCE_DELAY_MIN: [(("A", 0), ("B", 0))],
        ConfigNetwork.RECHARGE_COST_DISTANCE: [0.1, 0.5],
        ConfigNetwork.MAX_USER_BACKLOGGING_DELAY: [0, 20],
        ConfigNetwork.TRIP_REJECTION_PENALTY: [
            (("A", 0), ("B", 0)),
            (("A", 2.5), ("B", 2.5)),
        ],
        ConfigNetwork.TRIP_OUTSTANDING_PENALTY: [(("A", 0.25), ("B", 0.25))],
        ConfigNetwork.REBALANCE_SUB_LEVEL: [None, 1],
        # ConfigNetwork.METHOD: [
        #     ConfigNetwork.METHOD_ADP_TRAIN,
        #     ConfigNetwork.METHOD_ADP_TEST,
        #     ConfigNetwork.METHOD_MYOPIC,
        #     ConfigNetwork.METHOD_RANDOM,
        # ],
        # Config.STEPSIZE_RULE: [adp.STEPSIZE_MCCLAIN],
        # Config.DISCOUNT_FACTOR: [0.1, 0.3, 0.5],
        # "CLASS_PROB": [
        #     {
        #         ConfigNetwork.TRIP_CLASS_PROPORTION: (("A", 1), ("B", 0)),
        #         ConfigNetwork.USE_CLASS_PROB: False,
        #     },
        #     {
        #         ConfigNetwork.TRIP_CLASS_PROPORTION: (("A", 0), ("B", 1)),
        #         ConfigNetwork.USE_CLASS_PROB: False,
        #     },
        #     {
        #         ConfigNetwork.TRIP_CLASS_PROPORTION: (("A", 0), ("B", 0)),
        #         ConfigNetwork.USE_CLASS_PROB: True,
        #     },
        # ],
    }

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
        # ConfigNetwork.CASE_STUDY: "N08Z07SD02",
        ConfigNetwork.PATH_CLASS_PROB: "distr/class_prob_distribution_p5min_6h.npy",
        ConfigNetwork.ITERATIONS: ITERATIONS,
        ConfigNetwork.TEST_LABEL: test_label,
        ConfigNetwork.DISCOUNT_FACTOR: 1,
        ConfigNetwork.FLEET_SIZE: 400,
        # DEMAND ############################################# #
        ConfigNetwork.UNIVERSAL_SERVICE: False,
        ConfigNetwork.DEMAND_RESIZE_FACTOR: 0.1,
        ConfigNetwork.DEMAND_TOTAL_HOURS: 6,
        ConfigNetwork.DEMAND_EARLIEST_HOUR: 6,
        ConfigNetwork.OFFSET_TERMINATION_MIN: 30,
        ConfigNetwork.OFFSET_REPOSITIONING_MIN: 30,
        ConfigNetwork.TIME_INCREMENT: 5,
        ConfigNetwork.DEMAND_SAMPLING: True,
        # Service quality
        ConfigNetwork.MATCHING_DELAY: 15,
        ConfigNetwork.SQ_GUARANTEE: False,
        ConfigNetwork.RECHARGE_COST_DISTANCE: 0.1,
        ConfigNetwork.TRIP_REJECTION_PENALTY: (("A", 0), ("B", 0)),
        ConfigNetwork.TRIP_BASE_FARE: (("A", 2.5), ("B", 2.5)),
        ConfigNetwork.TRIP_DISTANCE_RATE_KM: (("A", 1), ("B", 1)),
        ConfigNetwork.TRIP_TOLERANCE_DELAY_MIN: (("A", 0), ("B", 0)),
        ConfigNetwork.TRIP_MAX_PICKUP_DELAY: (("A", 15), ("B", 15)),
        ConfigNetwork.TRIP_CLASS_PROPORTION: (("A", 0), ("B", 1)),
        # ADP EXECUTION ###################################### #
        ConfigNetwork.METHOD: m,
        ConfigNetwork.SAVE_PROGRESS: 10,
        ConfigNetwork.ADP_IGNORE_ZEROS: True,
        ConfigNetwork.LINEARIZE_INTEGER_MODEL: False,
        ConfigNetwork.USE_ARTIFICIAL_DUALS: False,
        # MPC ################################################ #
        ConfigNetwork.MPC_FORECASTING_HORIZON: 15,
        # EXPLORATION ######################################## #
        # ANNEALING + THOMPSON
        # If zero, cars increasingly gain the right of stay
        # still. This obliges them to rebalance consistently.
        # If None, disabled.
        ConfigNetwork.IDLE_ANNEALING: None,
        ConfigNetwork.ACTIVATE_THOMPSON: False,
        ConfigNetwork.MAX_TARGETS: 12,
        # When cars start in the last visited point, the model takes
        # a long time to figure out the best time
        ConfigNetwork.FLEET_START: conf.FLEET_START_RANDOM,
        # ConfigNetwork.FLEET_START: conf.FLEET_START_REJECTED_TRIP_ORIGINS,
        # ConfigNetwork.FLEET_START: conf.FLEET_START_PARKING_LOTS,
        # ConfigNetwork.LEVEL_PARKING_LOTS: 2,
        # ConfigNetwork.FLEET_START: conf.FLEET_START_LAST_TRIP_ORIGINS,
        ConfigNetwork.CAR_SIZE_TABU: 0,
        # REBALANCING ##########################################
        # If REACHABLE_NEIGHBORS is True, then PENALIZE_REBALANCE
        # is False
        ConfigNetwork.PENALIZE_REBALANCE: False,
        # All rebalancing finishes within time increment
        ConfigNetwork.REBALANCING_TIME_RANGE_MIN: (0, 10),
        # Consider only rebalance targets from sublevel
        ConfigNetwork.REBALANCE_SUB_LEVEL: None,
        # Rebalance to at most max targets
        ConfigNetwork.REBALANCE_MAX_TARGETS: None,
        # Remove nodes that dont have at least min. neighbors
        ConfigNetwork.MIN_NEIGHBORS: 1,
        ConfigNetwork.REACHABLE_NEIGHBORS: False,
        ConfigNetwork.N_CLOSEST_NEIGHBORS: ((1, 6), (2, 6), (3, 6),),
        ConfigNetwork.CENTROID_LEVEL: 1,
        # FLEET ############################################## #
        # Car operation
        ConfigNetwork.MAX_CARS_LINK: 5,
        ConfigNetwork.MAX_IDLE_STEP_COUNT: None,
        # FAV configuration
        # Functions
        ConfigNetwork.DEPOT_SHARE: 0.01,
        ConfigNetwork.FAV_DEPOT_LEVEL: None,
        ConfigNetwork.FAV_FLEET_SIZE: 0,
        ConfigNetwork.SEPARATE_FLEETS: False,
        ConfigNetwork.MAX_CONTRACT_DURATION: True,
        # mean, std, clip_a, clip_b
        # ConfigNetwork.FAV_EARLIEST_FEATURES = (8, 1, 5, 9)
        # ConfigNetwork.FAV_AVAILABILITY_FEATURES = (2, 1, 1, 4)
        # ConfigNetwork.PARKING_RATE_MIN = 1.50/60 # 1.50/h
        # ConfigNetwork.PARKING_RATE_MIN = 0.1*20/60
        # ,  # = rebalancing 1 min
        ConfigNetwork.PARKING_RATE_MIN: 0,  # = rebalancing 1 min
        # Saving
        ConfigNetwork.USE_SHORT_PATH: False,
        ConfigNetwork.SAVE_TRIP_DATA: False,
        ConfigNetwork.SAVE_FLEET_DATA: False,
        # Load 1st class probabilities dictionary
        ConfigNetwork.USE_CLASS_PROB: True,
        ConfigNetwork.ENABLE_RECHARGING: False,
        # PLOT ############################################### #
        ConfigNetwork.PLOT_FLEET_XTICKS_LABELS: [
            "",
            "6AM",
            "",
            "7AM",
            "",
            "8AM",
            "",
            "9AM",
            "",
            "10AM",
            "",
            "11AM",
            "",
            "12AM",
            "",
        ],
        ConfigNetwork.PLOT_FLEET_X_MIN: 0,
        ConfigNetwork.PLOT_FLEET_X_MAX: 84,
        ConfigNetwork.PLOT_FLEET_X_NUM: 15,
        ConfigNetwork.PLOT_FLEET_OMIT_CRUISING: False,
        ConfigNetwork.PLOT_DEMAND_Y_MAX: 3500,
        ConfigNetwork.PLOT_DEMAND_Y_NUM: 8,
        ConfigNetwork.PLOT_DEMAND_Y_MIN: 0,
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
