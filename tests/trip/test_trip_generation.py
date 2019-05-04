import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Adding project folder to import modules
root = os.getcwd().replace("\\", "/")
sys.path.append(root)

from mod.env.amod import Amod, StepLog
from mod.env.config import Config, ConfigStandard
from mod.env.match import fcfs
from mod.env.trip import get_random_trips