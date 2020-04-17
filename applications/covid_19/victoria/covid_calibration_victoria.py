from applications.covid_19.covid_calibration import *
from numpy import linspace

country = 'victoria'

# start_date = 11/3/2020 (day 71) for first item of the case_counts list
case_counts = [1, 1, 3, 1, 0, 4, 8, 7, 9, 12, 13, 22, 19, 18, 16,
               22, 40, 33, 33, 49, 34, 36, 43, 16, 23, 23, 18, 15, 10,
               7, 15, 2, 1, 3]
case_counts = [float(c) for c in case_counts]
data_times = linspace(71, 71 + len(case_counts) - 1, num=len(case_counts))
nb_time_points = len(case_counts)
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
