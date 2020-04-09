from applications.covid_19.covid_calibration import *
from numpy import linspace

country = 'Australia'

# start_date = 16/3/2020 (day 75) for first item of the case_counts list
case_counts = [9, 2, 5, 9, 17, 11, 10, 31, 26, 21, 28, 32, 42, 28, 25, 55, 39, 41]
data_times = linspace(75, 75 + 17, num=18)
nb_time_points = 18
case_counts = case_counts[:nb_time_points]
data_times = data_times[:nb_time_points].tolist()

target_to_plots = {'notifications': {'times': data_times, 'values': [[d] for d in case_counts]}}
print(target_to_plots)
PAR_PRIORS = [
    {"param_name": "contact_rate", "distribution": "uniform", "distri_params": [.1, 4.]},
    {"param_name": "start_time", "distribution": "uniform", "distri_params": [-30, 69]}
]

TARGET_OUTPUTS = [
    {"output_key": "notifications",
     "years": data_times,
     "values": case_counts,
     "loglikelihood_distri": 'poisson'}
]

run_calibration_chain(120, 1, country, PAR_PRIORS, TARGET_OUTPUTS)
