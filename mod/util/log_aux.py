import logging
import sys

FORMATTER = logging.Formatter(
    "%(asctime)s — %(name)s — %(levelname)s — %(message)s"
)

DEBUG = logging.DEBUG
INFO = logging.INFO

LOG_WEIGHTS = False
LOG_VALUE_UPDATE = False
LOG_DUALS = False


def get_console_handler():
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(FORMATTER)
    return console_handler


def get_file_handler(log_file, mode="w"):
    file_handler = logging.FileHandler(log_file, mode=mode)
    file_handler.setFormatter(FORMATTER)
    return file_handler


def log_weights(name, weights, value_vector, value_estimation):

    if LOG_WEIGHTS:

        logger = logging.getLogger(name)

        logger.debug(
            f"weights={weights}, values={value_vector}, "
            f"estimation={value_estimation}"
        )


def log_update_values_smoothed(name, t, level_update_list, values):

    if LOG_VALUE_UPDATE:

        logger = logging.getLogger(name)

        logger.debug(f"  ############ Time step = {t:>4} ################")
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

            pos, battery, contract_duration, car_type = a_g

            if g_time != previous_g_time or g != previous_g:

                logger.debug("")
                logger.debug(
                    f"  ## Value count={count_values:>4}, "
                    f"Agg. locations={count_locations}"
                )

                logger.debug(
                    f"  *************************************** "
                    f"Time({previous_g_time}) Location({previous_g}) "
                    f"***************************************"
                )

                count_values = 0
                previous_g = g
                previous_g_time = g_time
                count_locations = 0

            count_locations += 1
            count_values += len(level_update_list[k])

            logger.debug(
                f"    - vf={values[t_g][g][a_g]:6.2f}, "
                f"g({g_time})={t_level}, "
                f"location({g})={pos:>4}, "
                f"battery={battery}, "
                f"contract={contract_duration}, "
                f"car={car_type}, "
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

