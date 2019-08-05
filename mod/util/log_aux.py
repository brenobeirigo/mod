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

LOG_WEIGHTS = True
LOG_VALUE_UPDATE = True
LOG_DUALS = True
LOG_FLEET_ACTIVITY = True
LOG_COSTS = True


def log_costs(name, best_decisions, cost_func, post_cost_func, time_step, discount_factor, agg_level, penalize_rebalance, msg=""):
    
    # TODO this log is mixing up with the previous log
    if LOG_COSTS:
        overall_total = 0
        overall_post = 0
        overall_cost = 0
        
        logger = logging.getLogger(name)

        logger.debug(f"######## LOG COSTS {msg} (decisions={len(best_decisions)}, time={time_step}) #########################")

        for d in best_decisions:

            decision = d[:-1]

            cost = cost_func(decision)
            overall_cost += cost

            post_cost = post_cost_func(
                time_step,
                decision,
                level=agg_level,
                penalize_rebalance=penalize_rebalance,
            )

            overall_post += post_cost

            total = cost + discount_factor * post_cost

            logger.debug(
                f"{d} {cost:6.2f} + {discount_factor:.2f}*{post_cost:6.2f} = {total:6.2f}"
            )

            overall_total += total

        logger.debug(f"Overall total = {overall_total} ({overall_cost} + {discount_factor}*{overall_post}[{discount_factor*overall_post}])")


def log_fleet_activity(
    name, step, skip_steps, step_log, filter_status=[], msg=""
):
    if LOG_FLEET_ACTIVITY:

        logger = logging.getLogger(name)

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


def get_console_handler():
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(FORMATTER_TERSE)
    return console_handler


def get_file_handler(log_file, mode="w"):
    file_handler = logging.FileHandler(log_file, mode=mode)
    file_handler.setFormatter(FORMATTER)
    return file_handler


def log_weights(name, state, weights, value_vector, value_estimation):

    if LOG_WEIGHTS:

        logger = logging.getLogger(name)

        logger.debug(
            f"State={tuple([f'{str(e):4}' for e in state])}, "
            f"weights={[f'{w:7.3f}' for w in weights]}, "
            f"values={[f'{w:7.3f}' for w in value_vector]}, "
            f"estimation={value_estimation:6.2f}"
        )


def log_update_values_smoothed(name, t, level_update_list, values):

    if LOG_VALUE_UPDATE:

        logger = logging.getLogger(name)

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

            pos, battery, (g_contract, contract_duration), (
                g_cartype,
                car_type,
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
                f"g({g_time})={t_level}, "
                f"location({g})={pos:>4}, "
                f"battery={battery}, "
                f"contract({g_contract})={contract_duration}, "
                f"car({g_cartype})={car_type}, "
                f"values={list_two_floating}"
            )

        logger.debug(
            f"    values={count_values:>4}, "
            f"agg_locations={count_locations}"
        )


def log_duals(name, duals):

    if LOG_DUALS:

        logger = logging.getLogger(name)

        logger.debug("")
        logger.debug("  # DUALS ################################")
        equal_zero = 0
        for k, v in duals.items():
            if int(v) == 0:
                equal_zero += 1
            else:
                logger.debug(f"    - {k} = {v:6.2f}")
        logger.debug(
            f"  * {len(duals):>4} duals extracted ({equal_zero:>4} are zero)."
        )


def get_logger(
    logger_name, level_file=DEBUG, level_console=INFO, log_file="traces.log"
):
    logger = logging.getLogger(logger_name)

    # If logger was not configured, add handlers
    if not len(logger.handlers):

        logger = logging.getLogger(logger_name)
        logger.setLevel(level_file)

        ch = get_console_handler()
        ch.setLevel(level_console)
        logger.addHandler(ch)

        fh = get_file_handler(log_file, mode="a")
        fh.setLevel(level_file)
        logger.addHandler(fh)

        logger.propagate = False

    return logger

