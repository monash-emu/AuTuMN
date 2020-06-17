#!/bin/bash
RUN_NAME=$1
rm -rf data/calibration_outputs
aws s3 cp --recursive s3://autumn-data/$RUN_NAME/data/calibration_outputs data/calibration_outputs
