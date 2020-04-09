from applications.covid_19.covid_calibration import *
from numpy import linspace

country = 'Philippines'

# start_date = 11/3/2020 (day 70) for first item of the death_counts list
death_counts = [3, 3, 4, 4, 5, 2, 11, 4, 10, 11, 8, 10, 15, 7, 7, 12, 8, 10, 12, 9, 8, 6, 3, 2, 5, 1]
data_times = linspace(70, 70 + 25, num=26)
nb_time_points = 10
death_counts = death_counts[:nb_time_points]
data_times = data_times[:nb_time_points].tolist()

PAR_PRIORS = [
    {"param_name": "contact_rate", "distribution": "uniform", "distri_params": [.1, 4.]},
    {"param_name": "start_time", "distribution": "uniform", "distri_params": [-30, 69]}
]

TARGET_OUTPUTS = [
    {"output_key": "infection_deathsXall",
     "years": data_times,
     "values": death_counts,
     "loglikelihood_distri": 'poisson'}
]

run_calibration_chain(120, 1, country, PAR_PRIORS, TARGET_OUTPUTS)
