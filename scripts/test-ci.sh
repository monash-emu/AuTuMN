#!/bin/bash
# Run tests the same way they get run in GitHub Actions
export GITHUB_ACTION="1"
set -e
pytest --workers 7 -W ignore -vv -m "not run_models and not calibrate_models and not mixing_optimisation and not benchmark"
pytest --workers 7 -W ignore -vv -m run_models
pytest --workers 7 -W ignore -vv -m calibrate_models
pytest --workers 7 -W ignore -vv -m mixing_optimisation
