# Autumn PowerBI Data

Autumn has a calibration process that produces a bunch of data that we display in PowerBI. Model outputs and calibration data are stored in a SQLite database, which is ingested by PowerBI.

This document describes the data that is produced by Autumn and used by PowerBI.
Here we describe the database tables and what's in them. The data described here is similar to the output databases used in Autumn, but it is PowerBI-specific.

### MCMC Run

The MCMC run table tracks the progress of the calibration process. It contains metadata on each iteration. See [here](./schemas/mcmc_run.sql) for the table schema.

### MCMC Params

The MCMC params table tracks the parameters used for each iteration of the calibration. See [here](./schemas/mcmc_params.sql) for the table schema.

### PowerBI Outputs

This table contains the model's compartment values for the maximum likelihood (MLE) parameter set from the calibration. This is the "best fit" model. See [here](./schemas/powerbi_outputs.sql) for the table schema.

### Derived Outputs

This table contains additional values, calculated from the outputs of the MLE model. See [here](./schemas/derived_outputs.sql) for the table schema.

### Uncertainty

This table contains uncertainty estimates around some derived outputs, which are computed from all of the accepted MCMC iterations. See [here](./schemas/uncertainty.sql) for the table schema.

### Build

A simple table that stores the unique build key. See [here](./schemas/build.sql)  

### Scenario

This table contains metadata on the scenario. The description and start time.
See [here](./schemas/scenario.sql) 

### Calibration

This table contains calibation targets used in the model. See [here](./schemas/calibration.sql) 

