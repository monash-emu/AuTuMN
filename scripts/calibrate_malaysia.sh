#!/bin/bash

#SBATCH --job-name=malaysia_autumn_cal
#SBATCH --account=sh30

#SBATCH --time=16:00:00

#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=4096

#SBATCH --cpus-per-task=8

cd /projects/sh30/users/dshipman/AuTuMN

conda activate malaysia310

python scripts/calibrate_malaysia.py

