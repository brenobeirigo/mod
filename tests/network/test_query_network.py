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

        aggregated = nw.get_aggregated_centers(n)
        print(f"\n##### LEVEL INTERVAL [0, {n-1}]" " #######################")

        for g, nodes in enumerate(aggregated):
            print(f"level={g:>2} / unique node ids={len(set(nodes))}")

    print("\n##### POINT LIST #########################################")
    point_list = nw.get_point_list_map(4)
    print(point_list)

    print("\n##### NEIGHBORS ##########################################")
    for p in point_list:
        print(f"##### id = {p.id:>3} #####")
        for hops in range(0, 10):
            print(
                f"{hops} - neighbors={len(nw.get_neighbors_map(p.id, hops))}"
            )
