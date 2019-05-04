import os
import sys
from pprint import pprint

# Adding project folder to import modules
root = os.getcwd().replace("\\", "/")
sys.path.append(root)

from mod.env.network import get_aggregated_zones, get_cell_level_zone_ids

def test_get_aggregated_zones():

    for rows in [1,2,10,20, 100, 128, 256]:
        for cols in [1,2,10,20, 100, 128, 256]:
            for n_levels in [1, 2, 3, 4, 5]:
                print("\n\n####### AGGREGATED ZONES")
                print(f'##### ROWS={rows} - COLS={cols} - LEVELS={n_levels}')
                pprint(get_aggregated_zones(rows, cols, n_levels))
                
                print("\n\n####### CELL LEVEL ZONE IDS")
                print(f'##### ROWS={rows} - COLS={cols} - LEVELS={n_levels}')
                pprint(get_cell_level_zone_ids(rows, cols, n_levels))


if __name__ == "__main__":
    test_get_aggregated_zones()
    
