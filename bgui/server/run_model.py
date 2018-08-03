import os
import sys
from server.handler import run_model
if __name__ == "__main__":
    params_fname = os.path.abspath(sys.argv[1])
    run_model(params_fname)