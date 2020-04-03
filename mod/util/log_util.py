import logging
import sys
import numpy as np

np.set_printoptions(precision=3)

FORMATTER_TERSE = logging.Formatter("%(message)s")

FORMATTER = logging.Formatter("%(asctime)s — %(levelname)s — %(message)s")

FORMATTER_VERBOSE = logging.Formatter(
    "%(asctime)s — %(name)s — %(levelname)s — %(message)s"
)

# NOTSET 0, DEBUG 10, INFO 20, WARNING 30, ERROR 40, CRITICAL 50
DEBUG = logging.DEBUG
INFO = logging.INFO
WARNING = logging.WARNING
NOTSET = logging.NOTSET
CRITICAL = logging.CRITICAL

# Dictionary of available levels
levels = {
    "INFO": INFO,
    "DEBUG": DEBUG,
    "WARNING": WARNING,
    "NOTSET": NOTSET,
    "CRITICAL": CRITICAL,
}
level_labels = {
    INFO: "INFO",
    DEBUG: "DEBUG",
    WARNING: "WARNING",
    NOTSET: "NOTSET",
    CRITICAL: "CRITICAL",
}

# Log options
LOG_WEIGHTS = "LOG_WEIGHTS"
LOG_VALUE_UPDATE = "LOG_VALUE_UPDATE"
LOG_DUALS = "LOG_DUALS"
LOG_FLEET_ACTIVITY = "LOG_FLEET_ACTIVITY"
LOG_STEP_SUMMARY = "LOG_STEP_SUMMARY"
LOG_COSTS = "LOG_COSTS"
LOG_SOLUTIONS = "LOG_SOLUTIONS"

LOG_ALL = "log_all"
LEVEL_FILE = "level_file"
LEVEL_CONSOLE = "level_console"
LOG_LEVEL = "log_level"
FORMATTER_FILE = "formatter_file"

LOG_MIP = "log_mip"
# If True, saves time details in file times.csv
LOG_TIMES = "log_times"
SAVE_PLOTS = "save_plots"
SAVE_DF = "save_df"

# Save all logs for all iterations
log_dict = dict()


def get_console_handler():
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(FORMATTER_TERSE)
    return console_handler


# Console is shared by all logs
ch = get_console_handler()


def get_file_handler(log_file, mode="w", formatter=FORMATTER_VERBOSE):
    file_handler = logging.FileHandler(log_file, mode=mode)
    file_handler.setFormatter(formatter)
    return file_handler


def create_logger(
    name,
    log_level,
    level_file,
    level_console,
    log_file,
    formatter_file=FORMATTER_VERBOSE,
):
    print(
        f"\n#### Creating logger..."
        f"\n       id = {name}"
        f"\n    level = {level_labels[log_level]}, "
        f"console = {level_labels[level_console]}, "
        f"file = {level_labels[level_file]}\n"
    )

    logger = logging.getLogger(name)
    logger.setLevel(log_level)

    # The console handler is shared by all logs
    ch.setLevel(level_console)
    logger.addHandler(ch)

    # Only add file handler if level == DEBUG
    if log_level == DEBUG:
        fh = get_file_handler(log_file, formatter=formatter_file)
        fh.setLevel(level_file)
        logger.addHandler(fh)

    logger.propagate = False

    return logger


def log_solution(name, decision_vars):

    try:

        logger_obj = log_dict[name]

        if logger_obj.LOG_SOLUTIONS:

            logger = logger_obj.logger

            logger.debug("")
            logger.debug(
                "##################################"
                " SOLUTIONS "
                "##################################"
            )

            decision_vars = sorted(
                list(decision_vars.items()),
                key=lambda d: (d[0][1], d[0][0], d[0][4], d[0][6]),
            )

            for d, var in decision_vars:
                if var.x > 0.0:
                    logger.debug(f"{format_tuple(d)} = {var.x:6.2f}")

    except Exception as e:
        print(f"Can't log solutions! Exception: {e}")


def format_tuple(d, item_len=4):
    fd = "|".join([f"{e:>4}" for e in d])
    return f"[{fd}]"


def log_costs(
    name,
    best_decisions,
    cost_func,
    post_cost_func,
    time_step,
    discount_factor,
    msg="",
    filter_decisions=set(),
    post_opt=False,
):

    try:

        logger_obj = log_dict[name]

        if logger_obj.LOG_COSTS:

            logger = logger_obj.logger

            overall_total = 0
            overall_post = 0
            overall_cost = 0

            logger.debug(
                f"######## LOG COSTS {msg} (decisions={len(best_decisions)}, time={time_step}) #########################"
            )

            decision_labels = [
                "ACTI",
                "POSI",
                "BATT",
                "CONT",
                "CART",
                "DEPO",
                "ORIG",
                "DEST",
                "USER",
                "COUN",
            ]

            if not post_opt:
                decision_labels = decision_labels[:-1]

            cost_label = (
                f"{'COST':>7} + {'DISC':>6}*{'POST':>12} = {'TOTAL':>12}"
            )
            decision_label = format_tuple(decision_labels)
            label = f"{decision_label} {cost_label}"
            logger.debug(label)
            logger.debug("-" * len(label))

            best_decisions = sorted(
                best_decisions, key=lambda d: (d[1], d[0], d[4], d[6])
            )

            for d in best_decisions:

                decision_type = d[0]

                if filter_decisions and decision_type not in filter_decisions:
                    continue

                if post_opt:
                    # Remove decision count
                    decision = d[:-1]
                else:
                    decision = d

                cost = cost_func(decision)
                overall_cost += cost

                post_cost, post_state = post_cost_func(time_step, decision)
                overall_post += post_cost

                total = cost + discount_factor * post_cost

                logger.debug(
                    f"{format_tuple(d)} {cost:>7.2f} + {discount_factor:>6.2f}*{post_cost:>12.6f} = {total:>12.6f}"
                )

                overall_total += total

            logger.debug(
                f"Overall total = {overall_total:>6.2f} ({overall_cost:>6.2f} + {discount_factor:>6.2f}*{overall_post:>12.6f}[{discount_factor*overall_post:>12.6f}])"
            )

    except Exception as e:
        print(f"Can't log costs! Exception: {e}")


def delete(name):
    try:
        log_dict[name].handlers = []
        del log_dict[name]
    except:
        print(f"Can't delete log '{name}'")


def log_fleet_activity(
    name, step, skip_steps, step_log, filter_status=[], msg=""
):

    try:

        logger_obj = log_dict[name]

        if skip_steps > 0 and step % skip_steps == 0:
            logger = logger_obj.logger

            if logger_obj.LOG_STEP_SUMMARY:

                logger.debug("")
                logger.debug(
                    "----------------------------------------"
                    f" Fleet status ({msg})"
                    "----------------------------------------"
                )

                logger.info(step_log.info())

            if logger_obj.LOG_FLEET_ACTIVITY:

                car_status_list = step_log.env.get_car_status_list(
                    filter_status=filter_status
                )

                for c in car_status_list:
                    logger.debug(c)

            # if log_config_dict[la.SAVE_PLOTS]:
            #     stats_summary = amod.get_fleet_stats_summary()
            #     statuses = ", ".join(
            #         [
            #             f"{Car.status_label_dict[status_code]}: {status_count:>4}"
            #             for status_code, status_count in stats_summary.items()
            #         ]
            #     )

            #     logger.info(
            #         f"## {step+1:>3} - revenue = {revenue:>8.2f}, "
            #         f"serviced = {len(serviced):>4}, "
            #         f"rejected = {len(rejected):>4}, "
            #         f"outstanding = {len(outstanding):>4} "
            #         f"####### {statuses}"
            #     )

            #     car_log_str = amod.print_fleet_stats(
            #         filter_status=[Car.ASSIGN]
            #     )
            #     logger.debug(car_log_str)

    except Exception as e:
        print(f"Can't log fleet activity! Exception: {e}")


def log_weights(name, state, weights, value_vector, value_estimation):

    try:
        logger_obj = log_dict[name]

        if logger_obj.LOG_WEIGHTS:

            logger = logger_obj.logger

            logger.debug(
                f"State={tuple([f'{str(e):4}' for e in state])}, "
                f"weights={[f'{w:7.3f}' for w in weights]}, "
                f"values={[f'{w:7.3f}' for w in value_vector]}, "
                f"estimation={value_estimation:6.2f}"
            )

    except Exception as e:
        print(f"Can't log weights! Exception: {e}")


def log_update_values_smoothed(name, t, level_update_list, values):

    try:
        logger_obj = log_dict[name]

        if logger_obj.LOG_VALUE_UPDATE:

            logger = logger_obj.logger

            logger.debug(
                "  ############ Updating value functions "
                f"(method=smoothed, time={t:>4}) ################"
            )

            keys = sorted(level_update_list.keys(), key=lambda x: (x[0], x[1]))

            previous_g = 0
            previous_g_time = 0
            count_values = 0
            count_locations = 0

            logger.debug(
                f"  *************************************** "
                f"Time({previous_g_time}) Location({previous_g}) "
                f"***************************************"
            )

            for k in keys:
                list_two_floating = [
                    float(f"{e:6.2f}") for e in level_update_list[k]
                ]

                t_g, g, a_g = k

                g_time, t_level = t_g

                (
                    pos,
                    battery,
                    (g_contract, contract_duration),
                    (g_cartype, car_type),
                    (g_carorigin, car_origin),
                ) = a_g

                if g_time != previous_g_time or g != previous_g:

                    previous_g = g
                    previous_g_time = g_time

                    logger.debug("")
                    logger.debug(
                        f"  ## Value count={count_values:>4}, "
                        f"Agg. locations={count_locations:>4}"
                    )

                    logger.debug(
                        f"*************************************** "
                        f"Time({previous_g_time}) Location({previous_g}) "
                        f"***************************************"
                    )

                    count_values = 0
                    count_locations = 0

                count_locations += 1
                count_values += len(level_update_list[k])

                logger.debug(
                    f"    - vf={values[t_g][g][a_g]:6.2f}, "
                    f"time({g_time})={t_level}, "
                    f"location({g})={pos:>4}, "
                    f"battery={battery}, "
                    f"contract({g_contract})={contract_duration}, "
                    f"car({g_cartype})={car_type}, "
                    f"origin({g_carorigin})={car_origin}, "
                    f"values={list_two_floating}"
                )

            logger.debug(
                f"    values={count_values:>4}, "
                f"agg_locations={count_locations}"
            )

    except Exception as e:
        print(f"Can't log value function update! Exception: {e}")


def log_update_values(name, t, values):

    try:
        logger_obj = log_dict[name]

        if logger_obj.LOG_VALUE_UPDATE:

            logger = logger_obj.logger

            logger.debug(
                "  ############ Updating value functions "
                f"(method=smoothed, time={t:>4}) ################"
            )

            for g, state_data in enumerate(values):

                logger.debug(f"\n############## {g}")

                for a_g, data in state_data.items():
                    (
                        t_g,
                        pos_g,
                        battery_g,
                        contract_duration_g,
                        car_type_g,
                        car_origin_g,
                    ) = a_g

                    logger.debug(
                        # f"    - vf={data[VF]:6.2f}, "
                        f"time={t_g}, "
                        f"location={pos_g:>4}, "
                        f"battery={battery_g}, "
                        f"contract={contract_duration_g}, "
                        f"car={car_type_g}, "
                        f"origin={car_origin_g}, "
                        # f"values={data}"
                        f" [VF = {data[0]:6.2f}, "
                        f"COUNT = {data[1]:>6}, "
                        f"TRANSIENT_BIAS = {data[2]:6.2f}, "
                        f"VARIANCE_G = {data[3]:6.2f}, "
                        f"STEPSIZE_FUNC = {data[4]:6.2f}, "
                        f"LAMBDA_STEPSIZE = {data[5]:6.2f}]"
                    )

    except Exception as e:
        print(f"Can't log value function update! Exception: {e}")


def log_attribute_cars_dict(name, attribute_cars_dict, msg=""):
    try:
        logger_obj = log_dict[name]

        if logger_obj.LOG_ATTRIBUTE_CARS:

            logger = logger_obj.logger

            # Header
            logger.debug("")
            logger.debug(
                f"  # ATTRIBUTE CAR COUNT {msg} ################################"
            )
            header = ["POSI", "BATT", "CONT", "CART", "DEPO"]
            logger.debug(f"    - {format_tuple(header)} = {'COUNT':>10}")

            car_attributes = sorted(
                list(attribute_cars_dict.items()), key=lambda d: (d[0][0])
            )

            # Car attribute list
            for k, v in car_attributes:
                car_count = len(v)
                logger.debug(f"    - {format_tuple(k)} = {car_count}")

    except Exception as e:
        print(f"Can't log car attributes! Exception: {e}")


def log_duals(name, duals, msg=""):
    """Log dictionary of car flow tuples associated to duals
    
    Parameters
    ----------
    name : str
        Logger name saved in log_dict
    duals : dict
        Dictionary of duals (float) associated to car_flow tuples
    msg : str, optional
        Message to be shown in log header, by default ""
    """

    try:
        logger_obj = log_dict[name]

        if logger_obj.LOG_DUALS:

            logger = logger_obj.logger

            header = ["POSI", "BATT", "CONT", "CART", "DEPO"]

            logger.debug("")
            logger.debug(f"  # DUALS {msg} ################################")
            logger.debug(f"    - {format_tuple(header)} = {'DUAL':>10}")
            equal_zero = 0

            duals = sorted(list(duals.items()), key=lambda d: (d[0][0]))

            for k, v in duals:
                if v == 0:
                    equal_zero += 1
                else:
                    logger.debug(f"    - {format_tuple(k)} = {v:10.5f}")
            logger.debug(
                f"  * {len(duals):>4} duals extracted ({equal_zero:>4} are zero)."
            )

    except Exception as e:
        print(f"Can't log duals! Exception: {e}")


class LogAux:
    def __init__(
        self,
        logger_name,
        log_level,
        level_file,
        level_console,
        log_file,
        formatter_file=FORMATTER_VERBOSE,
        LOG_WEIGHTS=True,
        LOG_VALUE_UPDATE=True,
        LOG_DUALS=True,
        LOG_FLEET_ACTIVITY=True,
        LOG_STEP_SUMMARY=True,
        LOG_COSTS=True,
        LOG_SOLUTIONS=True,
        LOG_ATTRIBUTE_CARS=True,
        log_all=False,
        log_mip=False,
        log_times=False,
        save_plots=False,
        save_df=False,
    ):

        self.LOG_SOLUTIONS = LOG_SOLUTIONS or log_all
        self.LOG_WEIGHTS = LOG_WEIGHTS or log_all
        self.LOG_VALUE_UPDATE = LOG_VALUE_UPDATE or log_all
        self.LOG_DUALS = LOG_DUALS or log_all
        self.LOG_FLEET_ACTIVITY = LOG_FLEET_ACTIVITY or log_all
        self.LOG_STEP_SUMMARY = LOG_STEP_SUMMARY or log_all
        self.LOG_COSTS = LOG_COSTS or log_all
        self.LOG_ATTRIBUTE_CARS = LOG_ATTRIBUTE_CARS or log_all
        self.logger = create_logger(
            logger_name,
            log_level,
            level_file,
            level_console,
            log_file,
            formatter_file=formatter_file,
        )


def get_logger(
    name,
    log_level=INFO,
    level_file=DEBUG,
    level_console=INFO,
    log_file="traces.log",
    formatter_file=FORMATTER_VERBOSE,
    LOG_WEIGHTS=False,
    LOG_VALUE_UPDATE=False,
    LOG_DUALS=False,
    LOG_FLEET_ACTIVITY=False,
    LOG_STEP_SUMMARY=False,
    LOG_COSTS=False,
    LOG_SOLUTIONS=True,
    LOG_ATTRIBUTE_CARS=True,
    log_all=False,
    log_mip=False,
    log_times=False,
    save_plots=False,
    save_df=False,
):
    try:
        return log_dict[name].logger

    except:
        logger = LogAux(
            name,
            log_level,
            level_file,
            level_console,
            log_file,
            formatter_file=formatter_file,
            LOG_WEIGHTS=LOG_WEIGHTS,
            LOG_VALUE_UPDATE=LOG_VALUE_UPDATE,
            LOG_DUALS=LOG_DUALS,
            LOG_FLEET_ACTIVITY=LOG_FLEET_ACTIVITY,
            LOG_STEP_SUMMARY=LOG_STEP_SUMMARY,
            LOG_COSTS=LOG_COSTS,
            LOG_SOLUTIONS=LOG_SOLUTIONS,
            LOG_ATTRIBUTE_CARS=LOG_ATTRIBUTE_CARS,
            log_all=log_all,
            log_mip=log_mip,
            log_times=log_times,
            save_plots=save_plots,
            save_df=save_df,
        )
        log_dict[name] = logger

        return log_dict[name].logger
