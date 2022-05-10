# To generate all the figures and tables for the manuscript

## Preliminary steps
1. Put the PBI database containing all percentile outputs for all scenarios in "apps/tuberculosis/regions/marshall_islands/outputs/pbi_databases". 
2. Put a copy of the same PBI database in "data/outputs/calibrate/tuberculosis/marshall-islands/yyyy-mm-dd/" (to allow Step 3)
3. Put a copy of the file "priors-1.yml" in the same folder as for step 2. This file can be generated locally by running a short calibration.
4. Previously there was a step here to create a dashboard, but we now use notebooks instead, so this won't work
5. Put the csv file into "apps/tuberculosis/regions/marshall_islands/outputs/parameter_posteriors".


## Generating the output files
Run the following command from a prompt: python -m apps plotrmi

All the figures and tables will be available from "apps/tuberculosis/regions/marshall_islands/outputs/figures". 

# Scenario list
The scenarios indexing should follow the definitions presented in ""apps/tuberculosis/regions/marshall_islands/README.md".
