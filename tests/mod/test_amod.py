import os
import sys
from pprint import pprint

# Adding project folder to import modules
root = os.getcwd().replace("\\", "/")
sys.path.append(root)

from mod.env.amod import Amod
from mod.env.config import Config, ConfigStandard

c=ConfigStandard()
c.update({'FLEET_SIZE':10, 'ROWS':10, 'COLS':10})
amod = Amod(c)
amod.print_environment()