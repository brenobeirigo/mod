import os
import sys
from pprint import pprint
import numpy as np

# Adding project folder to import modules
root = os.getcwd().replace("\\", "/")
sys.path.append(root)

from mod.env.ml import get_state
from mod.env.amod import Amod
from mod.env.config import ConfigStandard
from mod.env.trip import get_random_trips


c=ConfigStandard()
c.update({'FLEET_SIZE':10, 'ROWS':10, 'COLS':10})
amod = Amod(c)

# Trips
min_trips, max_trips = 3, 10

for time_step in range(10):
    print(f"#### TIME STEP {time_step} ############################### ")
    trips = get_random_trips(
        amod.points,
        time_step,
        min_trips,
        max_trips
    )
    
    print("\n### Cars")
    pprint(amod.cars)

    print("\n### Trips")
    pprint(trips)

    print(f"\n# Tuple state space representation (per level)")
    for g in range(amod.config.aggregation_levels):
        print(f'\n####### Level {g:>03} #############################')

        state, cars_per_attribute, trips_per_attribute = (
            get_state(time_step, amod.cars, trips, level=g)
        )

        print('State:', state)
        
        print('\n\n#### Cars per attribute #####################')
        pprint(cars_per_attribute)
        
        print('\n\n#### Trips per attribute ####################')
        pprint(trips_per_attribute)

    # contrib, serviced, rejected = method.fcfs2(amod, trips, 1)
    # print(contrib, serviced, rejected)

    # amod.start_action_space_sparse()
    # action_tuple = amod.action_space.get_action_tuple(amod.cars)
    # print('\nAction tuple:', action_tuple)
    # print('Hash action tuple:', hash(tuple(action_tuple)))

    # print("\nIndividual car actions:")
    # for a in action_tuple:
    #     ActionSpace.read_single_action_tuple(a)
        