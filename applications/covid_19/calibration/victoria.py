from .base import run_calibration_chain
from numpy import linspace

country = 'victoria'

########  local transmission only
# start_date = 11/3/2020 (day 71) for first item of the case_counts list
# case_counts = [1, 1, 3, 1, 0, 4, 8, 7, 9, 12, 13, 22, 19, 18, 16,
#                22, 40, 33, 33, 49, 34, 36, 43, 16, 23, 23, 18, 15, 10,
#                7, 15, 2, 1, 3]
# case_counts = [float(c) for c in case_counts]
# data_times = linspace(71, 71 + len(case_counts) - 1, num=len(case_counts))
# nb_time_points = len(case_counts)
# case_counts = case_counts[:nb_time_points]
# data_times = data_times[:nb_time_points].tolist()

# #######  all cases
data_times = [67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92,
              93, 94, 95, 96, 97, 98, 99, 100, 101, 102]
               34, 11, 21, 24, 19, 15, 27, 7]

target_to_plots = {'notifications': {'times': data_times, 'values': [[d] for d in case_counts]}}
print(target_to_plots)
PAR_PRIORS = [
    {"param_name": "contact_rate", "distribution": "uniform", "distri_params": [.1, .5]},
    #{"param_name": "start_time", "distribution": "uniform", "distri_params": [-30, 69]}
]

TARGET_OUTPUTS = [
    {"output_key": "notifications",
     "years": data_times,
     "values": case_counts,
     "loglikelihood_distri": 'poisson'}
]


run_calibration_chain(120, 1, country, PAR_PRIORS, TARGET_OUTPUTS, mode='lsm')
