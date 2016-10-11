  
AuTuMN  
======  
  
This is a set of Python models that modularises the development of dynamic transmission models and allows a
pluggable API to other system. Applied to Tuberculosis.

# TODO
- philosophically, "consolidate" code to ensure all the structures are as robust, consistent and final as possible,
    - including:
        - remove unnecessary calculations during integration - including fraction calculations
        - consolidate calculation and collation of model outputs to occur mostly towards the end of model_runner
            execution (rather than at the start of outputs execution) - meaning that the outputs module would be
            just for producing output structure from alreday interpreted models
        - ensure code runs with defaults for countries without country-specific spreadsheets
- add epidemiological functionality, including:
    - get second strain incorporated properly (for Philippines application)
    - implement HIV with appropriate parameters
    - (new/retreatment stratification - possibly one day)
- debug the problem Emma found with storage of scenarios after uncertainty


