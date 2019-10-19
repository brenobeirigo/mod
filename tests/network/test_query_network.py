import os
import sys
from pprint import pprint

# Adding project folder to import modules
root = os.getcwd().replace("\\", "/")
sys.path.append(root)

import mod.env.network as nw

if __name__ == "__main__":
    # Only first
    for n in [1, 4, 100]:

        aggregated, distances = nw.query_aggregated_centers(n)
        print(f"\n##### LEVEL INTERVAL [0, {n-1}]" " #######################")

        for g, nodes in enumerate(aggregated):
            print(f"level={g:>2} / unique node ids={len(set(nodes))}")

    print("\n##### POINT LIST #########################################")
    point_list = nw.query_point_list(4, projection="GPS", step=60)
    print(point_list)

    print("\n##### NEIGHBORS ##########################################")
    for p in point_list:
        print(f"##### id = {p.id:>3} #####")
        for i, (dist, id_level) in enumerate(p.level_ids_dic.items()):
            if i == 0:
                continue
            neighbors = nw.query_neighbor_zones(id_level, dist)
            print(f"{dist} - neighbors={len(neighbors)}")

            print(
                "distances=",
                [f"{nw.get_distance(p.id, n):.2f}" for n in neighbors],
            )
