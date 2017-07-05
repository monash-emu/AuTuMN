#!/bin/bash
# Rebuilds the client, so JavaScript changes become visible.
# Version: 2016sep01

cd `dirname $0` # Make sure we're in this folder
cd ../client
source clean_dev_build.sh
