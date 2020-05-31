#!/bin/bash
set -e
RUN_TIME=30
NUM_CHAINS=30
CALIBRATION_NAME=malaysia

# Set up repo to run latest code.
cd ~/code
git pull
. ./env/bin/activate
pip install -r requirements.txt

# Run Malaysia calibration
touch pids.txt
mkdir -p logs
for i in $(seq 0 1 $NUM_CHAINS)
do
    echo "Starting chain $i"
    touch logs/run-$i.log
    nohup python3 -m apps calibrate $CALIBRATION_NAME $RUN_TIME $i &> logs/run-$i.log &
    echo $! >> pids.txt
done

echo "PIDS!"
cat pids.txt

# And umm...

rm -f pids
rm -rf logs
rm -rf data/covid_malaysia/