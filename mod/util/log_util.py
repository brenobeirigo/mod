import logging
import sys
import numpy as np

np.set_printoptions(precision=3)

FORMATTER_TERSE = logging.Formatter("%(message)s")

FORMATTER = logging.Formatter("%(asctime)s — %(levelname)s — %(message)s")

FORMATTER_VERBOSE = logging.Formatter(
    "%(asctime)s — %(name)s — %(levelname)s — %(message)s"
)

DEBUG = logging.DEBUG
INFO = logging.INFO
WARNING = logging.WARNING

# Save all logs for all iterations
log_dict = dict()


def get_console_handler():
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(FORMATTER_TERSE)
    return console_handler


# Console is shared by all logs
ch = get_console_handler()


def get_file_handler(log_file, mode="w"):
    file_handler = logging.FileHandler(log_file, mode=mode)
    file_handler.setFormatter(FORMATTER)
    return file_handler


def create_logger(name, level_file, level_console, log_file):

    logger = logging.getLogger(name)
    logger.setLevel(level_file)

    # ch = get_console_handler()
    ch.setLevel(level_console)
    logger.addHandler(ch)

    fh = get_file_handler(log_file)
    fh.setLevel(level_file)
    logger.addHandler(fh)

    logger.propagate = False

    return logger


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

            logger.debug(
                f"{[f'{e:>4}' for e in decision_labels]} "
                f"{'COST':>7} + {'DISC':>6}*{'POST':>7} = {'TOTAL':>7}"
            )

            sorted(best_decisions, key=lambda d: (d[0], d[4]))

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

                post_cost = post_cost_func(time_step, decision)
                overall_post += post_cost

                total = cost + discount_factor * post_cost

                logger.debug(
                    f"{[f'{e:>4}' for e in d]} {cost:>7.2f} + {discount_factor:>6.2f}*{post_cost:>7.2f} = {total:>7.2f}"
                )

                overall_total += total

            logger.debug(
                f"Overall total = {overall_total:>6.2f} ({overall_cost:>6.2f} + {discount_factor:>6.2f}*{overall_post:>6.2f}[{discount_factor*overall_post:>6.2f}])"
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

        if logger_obj.LOG_FLEET_ACTIVITY:

            logger = logger_obj.logger

            if skip_steps > 0 and step % skip_steps == 0:

                logger.debug("")
                logger.debug(
                    "----------------------------------------"
                    f" Fleet status ({msg})"
                    "----------------------------------------"
                )

                logger.debug(step_log.info())

                car_status_list = step_log.env.get_car_status_list(
                    filter_status=filter_status
                )

                for c in car_status_list:
                    logger.debug(c)

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


def log_duals(name, duals, msg=""):

    try:
        logger_obj = log_dict[name]

        if logger_obj.LOG_DUALS:

            logger = logger_obj.logger

            dual_labels = ["POSI", "BATT", "CONT", "CART", "DEPO"]

            logger.debug("")
            logger.debug(f"  # DUALS {msg} ################################")
            logger.debug(
                f"    - {[f'{e:>4}' for e in dual_labels]} = {'DUAL':>7}"
            )
            equal_zero = 0
            for k, v in duals.items():
                if int(v) == 0:
                    equal_zero += 1
                else:
                    logger.debug(f"    - {[f'{e:>4}' for e in k]} = {v:7.2f}")
            logger.debug(
                f"  * {len(duals):>4} duals extracted ({equal_zero:>4} are zero)."
            )

    except Exception as e:
        print(f"Can't log duals! Exception: {e}")


class LogAux:
    def __init__(
        self,
        logger_name,
        level_file,
        level_console,
        log_file,
        LOG_WEIGHTS=True,
        LOG_VALUE_UPDATE=True,
        LOG_DUALS=True,
        LOG_FLEET_ACTIVITY=True,
        LOG_COSTS=True,
        log_all=False,
    ):

        self.LOG_WEIGHTS = LOG_WEIGHTS or log_all
        self.LOG_VALUE_UPDATE = LOG_VALUE_UPDATE or log_all
        self.LOG_DUALS = LOG_DUALS or log_all
        self.LOG_FLEET_ACTIVITY = LOG_FLEET_ACTIVITY or log_all
        self.LOG_COSTS = LOG_COSTS or log_all
        self.logger = create_logger(
            logger_name, level_file, level_console, log_file
        )


def get_logger(
    name, level_file=DEBUG, level_console=INFO, log_file="traces.log"
):
    try:
        return log_dict[name].logger

    except:
        logger = LogAux(name, level_file, level_console, log_file)
        log_dict[name] = logger

        return log_dict[name].logger

