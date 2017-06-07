import os
import logging

datapath = os.path.dirname(__file__)
datapath = os.path.join(datapath, '..', 'caldb-inputdata')

default_verbose = 1

logging.basicConfig(level="INFO")
