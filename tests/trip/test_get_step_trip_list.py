import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import random
import pandas as pd
import numpy as np
from collections import defaultdict
from datetime import timedelta, datetime
import math
import pprint

# Adding project folder to import modules
root = os.getcwd().replace("\\", "/")
sys.path.append(root)

import mod.env.config as conf
import mod.env.trip as trip

folder_tripdata = "C:/Users/LocalAdmin/OneDrive/leap_forward/street_network_server/tenv/data/out/manhattan-island-new-york-city-new-york-usa/tripdata/ids/"
path = folder_tripdata + "tripdata_ids_2011-04-12_000000_2011-04-12_235959.csv"

step = 0.5

step_trips = trip.get_step_trip_list(
    path, step=step, resize_factor=1, earliest_step=600, max_steps=480
)

for s, trips in enumerate(step_trips):
    print(f"#### {s:03} = {s*step}")
    pprint.pprint(trips)
