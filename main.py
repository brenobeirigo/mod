import mod.env.amod as amod
from mod.env.config import Config
from pprint import pprint

config = Config()
print(config)

env = amod.Amod(config)

env.print_environment()