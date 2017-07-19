#!/bin/bash
# Assumes postgres and redis databases are already running
# (check with "ps -ef | grep redis")
# Version: 2016sep01

cd `dirname $0` # Make sure we're in the bin folder
python run_server.py 3000
