from applications.covid_19.covid_calibration import *
from numpy import linspace

country = 'philippines'
PAR_PRIORS, TARGET_OUTPUTS = get_priors_and_targets(country, 'deaths', 2)



# get rid of time in the params to calibrate
del PAR_PRIORS[1]

# start_date = 11/3/2020 (day 70) for first item of the death_counts list

target_to_plots = {'infection_deathXall': {'times': TARGET_OUTPUTS[0]['years'], 'values': [[d] for d in TARGET_OUTPUTS[0]['values']]}}
print(target_to_plots)

run_calibration_chain(120, 1, country, PAR_PRIORS, TARGET_OUTPUTS)
