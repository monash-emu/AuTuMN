#!/bin/bash
RUN_NAME=$1
rm -rf data/full_model_runs
aws s3 cp --recursive s3://autumn-data/$RUN_NAME/data/full_model_runs data/full_model_runs
