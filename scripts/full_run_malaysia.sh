#!/bin/bash

#SBATCH --job-name=malaysia_autumn_full
#SBATCH --account=sh30

#SBATCH --time=03:00:00

#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=4096

#SBATCH --cpus-per-task=8

cd /projects/sh30/users/dshipman/AuTuMN

conda activate malaysia310

python scripts/full_run_malaysia.py

