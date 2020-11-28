import os
import random
import sys

# Adding project folder to import modules
root = os.getcwd().replace("\\", "/")
sys.path.append(root)

from mod.env.config import ConfigNetwork
import mod.util.file_util as fileutil

import mod.env.network as nw
import mod.env.adp.alg.value_iteration as alg

# Reproducibility of the experiments
random.seed(1)


def get_network_configuration():
    # Pull graph info
    (
        region,
        label,
        node_count,
        center_count,
        edge_count,
        region_type,
    ) = nw.query_info()
    info = (
        "##############################################################"
        f"\n### Region: {region} G(V={node_count}, E={edge_count})"
        f"\n### Center count: {center_count}"
        f"\n### Region type: {region_type}"
    )
    level_id_count_dict = {
        int(level): (i + 1, count)
        for i, (level, count) in enumerate(center_count.items())
    }
    level_id_count_dict[0] = (0, node_count)
    config_network = {
        # -------------------------------------------------------- #
        # NETWORK ################################################ #
        # -------------------------------------------------------- #
        ConfigNetwork.NAME: label,
        ConfigNetwork.REGION: region,
        ConfigNetwork.NODE_COUNT: node_count,
        ConfigNetwork.EDGE_COUNT: edge_count,
        ConfigNetwork.CENTER_COUNT: center_count,
    }
    print(level_id_count_dict)

    return config_network


if __name__ == "__main__":

    try:
        folder = sys.argv[1]
        filename = sys.argv[2]

        filepath_method_config = folder + filename
        filepath_log = folder + "base_log.json"

        base_config = fileutil.read_json_file(filepath_method_config)
        log_config = fileutil.read_json_file(filepath_log)

        start_config_dict = fileutil.join_dictionary_keys(base_config)

        start_config = ConfigNetwork()
        start_config.update(start_config_dict)

        config_network = get_network_configuration()
        start_config.update(config_network)
        start_config.make_int_keys(start_config)

        vi = alg.ValueIteration(None, start_config, log_config_dict=log_config)
        vi.init()

    except Exception as e:
        print("Exception: ", e)
