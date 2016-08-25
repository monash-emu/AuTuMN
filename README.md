  
AuTuMN  
======  
  
This is a set of Python models that modularises the development of dynamic transmission models and allows a
pluggable API to other system. Applied to Tuberculosis.

# Files
  
- model.py: underlying transmission dynamic model.  
- plotting.py: plotting results of the model  
- spreadsheet.py: reads data from input.xlsx  


# TODO
  
# James TODO:
- Xpert improvement in diagnostic algorithm
- allow scenarios to be scaled-up over varying time periods
- population stratification for at-risk groups (partially done)
- would be nice to include assertion check of age-stratified variables
    to ensure that each age group starts where the last one left off


# Bigger outstanding tasks TODO
- incorporate all the analysis tools, including plotting, report writing, etc. into one
    big analysis object (I think this is probably the more elegant way to have this coded)
- GUI
- Automatic calibration
- new/retreatment stratification (possibly)




