#!/bin/bash
# Usage: sbatch slurm-openmp-job-script
# Prepared By: Kai Xi,  Apr 2015
#              help@massive.org.au

# NOTE: To activate a SLURM option, remove the whitespace between the '#' and 'SBATCH'

# To give your job a name, replace "MyJob" with an appropriate name
#SBATCH --job-name=test_job


# To set a project account for credit charging, 
# SBATCH --account=pmosp


# Request CPU resource for a openmp job, suppose it is a 12-thread job
#SBATCH --ntasks=1
# SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8

# Memory usage (MB)
#SBATCH --mem-per-cpu=12000

# Set your minimum acceptable walltime, format: day-hours:minutes:seconds
# SBATCH --time=0-00:30:00


# To receive an email when job completes or fails
#SBATCH --mail-user=romain.ragonnet@monash.edu
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL


# Set the file for output (stdout)
# SBATCH --output=MyJob-%j.out

# Set the file for error log (stderr)
# SBATCH --error=MyJob-%j.err


# Use reserved node to run job when a node reservation is made for you already
# SBATCH --reservation=reservation_name


# Command to run a openmp job
# Set OMP_NUM_THREADS to the same value as: --cpus-per-task=12
source ../../miniconda/bin/activate
conda activate summer2

python test_job.py


