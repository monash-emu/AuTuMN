#!/bin/bash
# Run Slurm batch script
set -e
if [[ ! -f "./batch.sh"]]
    echo "Error: expected batch script batch.sh to exist. Run setup.sh to generate the script."
fi
echo -e "\n>>> Requesting Slurm batch job..."
sbatch batch.sh
echo -e "\n>>> Job request submitted."
