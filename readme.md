  
AuTuMN  
======  

This project is a modelling framework used by the [AuTuMN tuberculosis modelling project](http://www.tb-modelling.com/index.php). It provides a set of Python models that modularises the development of dynamic transmission models and allows a pluggable API to other system. Applied to tuberculosis.

## TODO
- setup.py file to load all dependent modules
- document Bulgaria interventions properly in handbook
- the model would not run without age-stratification (detected when running Bulgaria)

## major outstanding tasks
- Bulgaria paper
- Bhutan inputs
- age-specific parameterisation
- mapping to DALYs, QALYs
- optimisation

## minor tasks
- simplify code for automatic detection of int_uncertainty start_time. Should use common method with optimisation start_dates
- in the adjust_treatment_outcomes_support method, only the "relative" approach accounts for baseline intervention coverage
    The "absolute" approach should be updated similarly in case we use it with a non-zero coverage at baseline.
- the platform currently loads all data files every single time we hit the run button from the GUY. we may want to run the file loading operations only once when launching the GUI 
