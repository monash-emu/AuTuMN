  
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
- might speed things up to have spreadsheet reader save a data object of
    some sort, as reading is now taking 20+ seconds at each run
- Xpert improvement in diagnostic algorithm
- allow scenarios to be scaled-up over varying time periods
- population stratification for at-risk groups (partially done)
- would be nice to include assertion check of age-stratified variables
    to ensure that each age group starts where the last one left off
- make the analysis more consistently object-oriented, following the lead
    of Romain's object oriented spreadsheet writer

# Bigger outstanding tasks TODO
- incorporate all the analysis tools, including plotting, report writing, etc. into one
    big analysis object (I think this is probably the more elegant way to have this coded)
- GUI
- Automatic calibration
- new/retreatment stratification (possibly)




