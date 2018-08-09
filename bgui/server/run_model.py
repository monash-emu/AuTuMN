import os
import sys
from server.handler import run_model

fname = sys.argv[1]
params_fname = os.path.abspath(fname)
run_model(params_fname)