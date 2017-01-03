  
AuTuMN  
======  
  
This is a set of Python models that modularises the development of dynamic transmission models and allows a
pluggable API to other system. Applied to tuberculosis.

# TODO
- uncertainty for economics
- intervention uncertainty
- automatic calibration
- update last two sheet readers - all done except:
    - latest GTB reader doesn't have prevalence data
        - seems very strange, but needs to be sorted
    - treatment outcomes reader seems unchanged and bugs when updated
        - also strange because this is should be mutually inconsistent
    - (World Bank demographic data is current, as 2014 seems to be the last year with data available still)
- check whether treatment outcomes can be derived from all patients, rather than from smear-positive
    (and check whether this is currently happening)
- tidy the tidy_axis method of outputs itself
- run through formatting of curve.py to make it more James-like

