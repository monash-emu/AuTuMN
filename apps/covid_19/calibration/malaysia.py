from apps.covid_19.calibration.base import run_calibration_chain

country = "malaysia"

PAR_PRIORS = [
    {'param_name': 'contact_rate', 'distribution': 'uniform', 'distri_params': [0.010, 0.040]},
    {'param_name': 'start_time', 'distribution': 'uniform', 'distri_params': [10., 40.]}
]

# notification data, provided by the country
notification_times = [63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134]
notification_counts = [7, 14, 5, 28, 10, 6, 18, 12, 20, 9, 45, 35, 190, 125, 120, 117, 110, 130, 153, 123, 212, 106, 172, 235, 130, 159, 150, 156, 140, 142, 208, 217, 150, 179, 131, 170, 156, 109, 118, 184, 153, 134, 170, 85, 110, 69, 54, 84, 36, 57, 50, 71, 88, 51, 38, 40, 31, 94, 57, 69, 105, 122, 55, 30, 45, 39, 68, 54, 67, 70, 16, 37]

# ICU data (prev / million pop), provided by the country
icu_times = [71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134]
icu_counts = [0.06179533237608244, 0.09269299856412365, 0.12359066475216487, 0.15448833094020611, 0.27807899569237093, 0.3707719942564946, 0.3707719942564946, 0.46346499282061826, 0.6179533237608245, 0.8033393208890718, 1.143213648957525, 1.4212926446498961, 1.7611669727183494, 1.977450636034638, 1.390394978461855, 1.390394978461855, 1.6684739741542258, 2.255529631727009, 2.255529631727009, 2.9043806216758745, 2.9043806216758745, 3.1515619511802044, 3.244254949744328, 3.3369479483084517, 3.0588689526160806, 3.0588689526160806, 3.1515619511802044, 2.8425852892997923, 2.348222630291133, 2.2246319655389675, 2.131938966974844, 2.2246319655389675, 2.0392459684107207, 2.0392459684107207, 1.853859971282473, 1.7302693065303083, 1.7302693065303083, 1.5757809755901022, 1.5139856432140197, 1.4212926446498961, 1.390394978461855, 1.3285996460857723, 1.3285996460857723, 1.2977019798977312, 1.2668043137096898, 1.1123159827694837, 1.1123159827694837, 1.143213648957525, 1.1123159827694837, 1.235906647521649, 1.1123159827694837, 1.143213648957525, 0.9578276518292779, 0.8342369870771129, 0.8651346532651542, 0.7415439885129892, 0.6797486561369068, 0.5870556575727832, 0.5561579913847419, 0.5561579913847419, 0.5561579913847419, 0.6179533237608245, 0.494362659, 0.494362659]

TARGET_OUTPUTS = [
    {
        "output_key": "notifications",
        "years": notification_times,
        "values": notification_counts,
        "loglikelihood_distri": "poisson",
    },
    {
        "output_key": "prevXlateXclinical_icuXamong",
        "years": icu_times,
        "values": icu_counts,
        "loglikelihood_distri": "poisson",
    }
]

MULTIPLIERS = {'prevXlateXclinical_icuXamong': 1.e6}

# __________  For the grid-based calibration approach
# define a grid of parameter values. The posterior probability will be evaluated at each node
par_grid = [
    {"param_name": "contact_rate", 'lower': .018, 'upper': .023, 'n': 6},
    {"param_name": "start_time", 'lower': 25., 'upper': 35., 'n': 11},
]


def run_mys_calibration_chain(max_seconds: int, run_id: int):
    run_calibration_chain(max_seconds, run_id, country, PAR_PRIORS, TARGET_OUTPUTS, mode='grid_based',
                          _grid_info=par_grid, _run_extra_scenarios=False, _multipliers=MULTIPLIERS)
    # run_calibration_chain(max_seconds, run_id, country, PAR_PRIORS, TARGET_OUTPUTS, mode='autumn_mcmc',
    #                       _run_extra_scenarios=False, _multipliers=MULTIPLIERS)


if __name__ == "__main__":
    run_mys_calibration_chain(2 * 60 * 60, 1)  # first argument only relevant for autumn_mcmc mode (time limit in seconds)
