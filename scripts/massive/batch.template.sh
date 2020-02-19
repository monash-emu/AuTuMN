#!/bin/bash
# Template for our batch script - see https://slurm.schedmd.com/sbatch.html
#SBATCH --job-name={job_name}
#SBATCH --array=0-{array_end}
#SBATCH --ntasks={num_tasks}
#SBATCH --cpus-per-task={cores_per_task}
#SBATCH --mem-per-cpu={mem_per_cpu}
#SBATCH --time={runtime}
#SBATCH --mail-user={notification_email}
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --output=logs/%A-%a-stdout.log
#SBATCH --error=logs/%A-%a-stderr.log
set -e
echo "Running job {job_name}, array task $SLURM_ARRAY_TASK_ID"

cd {job_dir}

# Load Python 3.6 module.
module purge
module load python/3.7.3-system

# Activate the virutal environment, which has our code installed.
. env/bin/activate

# Run the job
{job_command}
