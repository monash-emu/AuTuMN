
AuTuMN
======

A dynamic population model of TB transmission.

This is a set of Python models that modularizes the development of population transmission models and allows a pluggable API to other system. In particular, this is designed to interface with the Optima suite of software.


# Files

- model.py: Underlying transmission dynamic model. Running this won't show you anything in the console window 
- plotting.py: plotting results of the model
- spreadsheet.py: This reads data in input.xlsx and import into model.py. Again it will needed to reflect the changes we will make to model.py and consequently input.xlsx; and I am happy to do this as well. 

# TODO

- Get Bosco to teach us Github - @done
- Get Dr Bosco Ph to teach DrDr Jim and Dr Tan Github @done
- Start programming in Github @done
- Change current working code for simple model over to object-oriented / something close to the final "look" of the code @done
- Start developing transmission model
- Very simple model @done
- Link transmission model to Excel inputs spreadsheet - Tan will lead
- Link to minimisation algorithm - will have some sub-sections @done
- Link to minimisation algorithm written by UNSW Optima
- Develop Bayesian / machine learning code for refining model inputs @done
- Run code for example country
- Link to macroeconomic model
- Write reports 
- Write scientific papers 
Created on Wed Nov 11 18:07:45 2015


1. MAKE IT PRODUCE GRAPHICAL CURVES
2. WHEN HAVING A SPENDING AMOUNT, USE THE COST-COVERAGE CURVE TO GET THE CORRESPONDING COVERAGE LEVEL
3. USE THE COVERAGE LEVEL VALUE FROM (2), USE THE COVERAGE-OUTCOME CURVE TO GET THE CORRESPONDING OUTCOME VALUE
4. USE THE OUTCOME VALUE FROM 3 AND FEED IT INTO JAMES' TRANSMISSION DYNAMIC MODEL
5. ADD MORE COMPLEXITY: MULTIPLE INTERVENTIONS, SAME INTERVENTIONS AFFECTING DIFFERENT OUTCOMES, COSTVERAGE LEVEL FOR EVERY GIVEN YEAR ETC.


# Model type (stage 4)

For this model, we focus on a model with pulmonary status, and complex disease state and treatment.


## Compartments

susceptible_unvac
susceptible_vac
susceptible_treated
latent_early
latent_late

-> permute over smearpospulm, smearnegpulm, extrapulm
active_smearpospulm
detect_smearpospulm
faildetect_smearpospulm
treatment_infect_smearpospulm
treatment_noninfect_smearpospulm


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
incidence - sum(latent -> active*) / population * 100,000
prevalence - sum(not(susceptible and latent)) / population * 100,000
mortality - sum(flows to tb_death) / population * 100,000



# Philippines test set (2014)

- 2013 - 98 million susceptible
- active 0.4% 400 in 100,000
- latent guesstimate - 30%
- vary n_tbfixed_contact 10-60
- who data: http://www.who.int/tb/country/data/download/en/

- incidence 2014 = 288 (254–324)
- prevalence 2014 = 417 (367–471)
- mortality 2014 = 10 (9.1–11)

- total population = 99,000,000 in 2014




# Cost function integration

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


# Useful references

WHO data: http://www.who.int/tb/country/data/download/en/


