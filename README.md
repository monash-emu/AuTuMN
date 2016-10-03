  
AuTuMN  
======  
  
This is a set of Python models that modularises the development of dynamic transmission models and allows a
pluggable API to other system. Applied to Tuberculosis.

# TODO
- philosophically, "consolidate" code to ensure all the structures are as robust, consistent and final as possible,
    - including:
        - convert cost code to being consistent with epi data structures
        - remove unnecessary calculations during integration - including fraction calculations
        - discuss optimal GUI with Eike - to make sure we're not heading in the wrong direction using Tkinter
        - discuss optimisation with Eike - to make sure we can iron out Romain's bugs
- add epidemiological functionality, including:
    - better structure for maintaining comorbidity (i.e. risk group) populations at target values
    - get second strain incorporated properly (for Philippines application)
    - implement HIV with appropriate parameters
    - (new/retreatment stratification - possibly one day)



