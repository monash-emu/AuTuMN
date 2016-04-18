  
AuTuMN  
======  
  
This is a set of Python models that modularizes the development of dynamic transmission models and allows a pluggable API to other system. Applied to Tuberculosis.  
  
  
# Files  
  
- model.py: underlying transmission dynamic model.  
- plotting.py: plotting results of the model  
- spreadsheet.py: reads data from input.xlsx  
- settings: parameters for model  
  
  
# TODO  
  
- link to optima optimization
- fit to Phillippines output
- Link to macroeconomic model  
- make cost-coverage curves from data  
- feed cost-coverage curves into model

# Bosco TODO:
- change set_scaleup_var to set_scaleup_param to allow complex combination of scaleups
- curve fitting to data
- search for BOSCO comments in the source code

# James TODO:
- work out an approach to allowing a lower case detection rate and higher force of infection
- scale-up BCG vaccination (entry flow)
- MCMC incorporation
- new/retreatment stratification
- age stratification

# Current TB dynamic transmission model  
  
## Compartments  
  
susceptible_unvac  
susceptible_vac  
susceptible_treated  
latent_early  
latent_late  
  
stratas over   
    - organ [smearpospulm, smearnegpulm, extrapulm]  
    - strain [xdr, dr, mdr]  
    - comoribidty [hiv, diabetes, nocomorb]  
active   
detect  
missed  
treatment_infect  
traetment_noninfet  
  
## Parameters  
  
rate_birth = 26 / 1000 * total population  
rate_death = 69.4 / 1000  
tb_n_contact = 10 to 70 range (very broadly gamma distributed)  
  
ratio_force_smearpospulm = 1.  
ratio_force_smearnegpulm = multiplier_relative_infectiousness_smearneg  
ratio_force_extrapulm = 0.  
  
tb_rate_early_active = proportion_early_progression / timeperiod_early_latent (divide up by thirds for each pulmstatus)  
tb_rate_late_active = rate_late_progression  
tb_rate_stabilise = (1 - proportion_early_progression) / timeperiod_early_latent  
tb_rate_recover = (1 - proportion_casefatality_active_untreated_smearpos) / timeperiod_activeuntreated  
tb_rate_death = proportion_casefatality_active_untreated_smearpos / timeperiod_activeuntreated  
  
program_prop_vac = .8  
program_prop_unvac = 1 - program_prop_vac  
program_rate_detect %country  
program_rate_faildetect %country  
program_rate_start_treatment %country  
program_rate_sick_of_waiting %country  
program_rate_completion_infect = 1 / (2 / 52)  
program_rate_default_infect %country  
program_rate_death_infect %country  
  
program_rate_completion_noninfect %country  
program_rate_default_noninfect %country  
program_rate_death_noninfect %country  
  
## Auxiliary variables  
  
equilibrium - 1% fractional for active compartments over 5 years  
prevalence - sum(not(susceptible and latent)) / population * 100,000  
incidence - sum(latent -> active*) / population * 100,000  
mortality - sum(flows to tb_death) / population * 100,000  
  
## Philippines test set   
- population 2014 = 99M
- guesstimates  
    - latent = 30%  
    - n_contact = 10-60  
- who data: http://www.who.int/tb/country/data/download/en/  
    - incidence 2014 = 288 (254–324)  
    - prevalence 2014 = 417 (367–471)  
    - mortality 2014 = 10 (9.1–11)  
  
## Cost function integration  
  
rate_birth: rate_pop_birth  
rate_death: rate_pop_death  
  
program_prop_vac: vaccination program  
program_prop_unvac  
program_rate_detect  
program_rate_faildetect  
program_rate_start_treatment  
program_rate_sick_of_waiting  
program_rate_completion_infect: Outcome: rate of Tx completion (Spreadsheet: Cost_functions_Tan.xlxs)  
program_rate_default_infect  
program_rate_death_infect  
  
program_rate_completion_noninfect  
program_rate_default_noninfect  
program_rate_death_noninfect  
  
  
  
