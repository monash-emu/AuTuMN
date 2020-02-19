#!/bin/bash
# Sets up a new job in MASSIVE.
# Run this script on the MASSIVE login server (m3.massive.org.au)
# 
set -e
# Read the job name from the user
JOB_NAME_RAW=$1
if [[ -z "$JOB_NAME_RAW" ]]
then
    echo "Error: job name expected as first argument."
    exit 1
fi

# Check that the AuTuMN code base exists.
CODE_DIR=/project/sh30/autumn-repo
if [[ ! -d "$CODE_DIR" ]]
then
    echo "Error: expected AuTuMN project to be present at $CODE_DIR."
fi

# Format job name so it can be used as a filename
JOB_NAME=$(echo "$JOB_NAME_RAW" | iconv -t ascii//TRANSLIT | sed -r s/[^a-zA-Z0-9]+/-/g | sed -r s/^-+\|-+$//g | tr A-Z a-z)
DATE_STAMP=$(date +"%d-%m-%Y")

# Make the job folder
mkdir -p /projects/sh30/autumn-jobs
JOB_DIR="/projects/sh30/autumn-jobs/$JOB_NAME-$DATE_STAMP"
echo -e "\n>>> Setting up job folder $JOB_DIR."
mkdir -p $JOB_DIR

# Copy over Slurm batch job files.
cp ./scripts/massive/batch.template.sh $JOB_DIR
cp ./scripts/massive/config.yml $JOB_DIR
cp ./scripts/massive/run-job.sh $JOB_DIR

# Copy over AuTuMN code
echo -e "\n>>> Copying AuTuMN code from $CODE_DIR to $JOB_DIR."
cp -r $CODE_DIR/autumn $JOB_DIR
cp -r $CODE_DIR/applications $JOB_DIR
cp $CODE_DIR/requirements.txt $JOB_DIR

echo -e "\n>>> Setting up Python 3 environment in $JOB_DIR."
# Load Python 3.6 module.
module purge
module load python/3.7.3-system

# Create a virtual environment
cd $JOB_DIR
virtualenv -p python3 env
. env/bin/activate

# Install Python requirements
echo -e "\n>>> Installing Python requirements."
pip3 install -r requirements.txt

# Generate the job's batch script.
echo -e "\n>>> Generating job batch script."
python3 build-script.py $JOB_NAME
chmod +x batch.sh

echo -e "\n>>> Setup complete, job ready to run."
echo -e ">>> Run $JOB_DIR/run-job.sh to start the job."
