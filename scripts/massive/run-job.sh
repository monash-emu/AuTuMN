#!/bin/bash
# Run Slurm batch script
set -e
if [[ ! -f "./batch.sh" ]]
then
    echo "Error: expected batch script batch.sh to exist. Run setup.sh to generate the script."
    exit 1
fi
echo -e "\n>>> Requesting Slurm batch job..."
sbatch batch.sh

echo -e "\n>>> Job request submitted.\n>>> To view job status: squeue -j <JOBID>"
