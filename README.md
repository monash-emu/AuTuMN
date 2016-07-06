  
AuTuMN  
======  
  
This is a set of Python models that modularises the development of dynamic transmission models and allows a
pluggable API to other system. Applied to Tuberculosis.

# Files
  
- model.py: underlying transmission dynamic model.  
- plotting.py: plotting results of the model  
- spreadsheet.py: reads data from input.xlsx  
- settings: parameters for model  

# TODO  
  
# James TODO:
- still one bug in spreadsheet.py: parse_row is very similar between
    ControlPanelReader and FixedParametersReader, but I haven't yet
    been able to rationalise these back to one function (that
    ControlPanelReader can inherit from FixedParametersReader)
- Xpert improvement in diagnostic algorithm
- allow scenarios to be scaled-up over varying time periods
- population stratification for at-risk groups
- new/retreatment stratification (possibly)


## Compartments  
  
susceptible_unvac  
susceptible_vac  
susceptible_treated  
latent_early  
latent_late  
  
stratas over   
    - organ [smearpospulm, smearnegpulm, extrapulm]  
    - strain [xdr, dr, mdr]  
    - comorbidity [hiv, diabetes, nocomorb]
active   
detect  
missed  
treatment_infect  
treatment_noninfet

