from apps.covid_19.calibration import base
from autumn.constants import Region
from apps.covid_19.mixing_optimisation.utils import (
    get_prior_distributions_for_opti,
    get_target_outputs_for_opti,
    get_weekly_summed_targets,
)


country = Region.FRANCE

PAR_PRIORS = get_prior_distributions_for_opti()
# TARGET_OUTPUTS = get_target_outputs_for_opti(country, data_start_time=50, update_jh_data=False)
TARGET_OUTPUTS = [{'output_key': 'notifications', 'years': [50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 109, 110, 111, 113, 114, 115, 116, 117, 118, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 134, 135, 137, 138, 139, 140, 141, 142, 143, 145, 147, 148, 149, 150, 151, 152, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 177, 180, 181, 182, 183, 184], 'values': [0, 0, 0, 0, 0, 2, 4, 20, 19, 43, 30, 61, 13, 81, 92, 276, 296, 177, 83, 575, 497, 0, 1380, 808, 30, 2134, 1019, 1391, 1828, 1741, 1670, 1736, 3838, 2448, 2929, 3922, 3809, 4611, 2599, 4376, 7578, 4861, 2116, 5233, 4267, 1873, 3912, 3777, 3881, 4286, 4342, 3114, 26843, 3665, 4959, 3216, 12471, 1979, 4902, 2380, 2733, 2311, 1610, 1656, 555, 3742, 3016, 671, 0, 1212, 296, 614, 1049, 3530, 575, 1278, 440, 269, 452, 728, 731, 607, 33, 314, 874, 641, 237, 339, 319, 307, 157, 3265, 507, 1800, 235, 101, 328, 678, 552, 529, 293, 98, 335, 397, 358, 564, 393, 291, 68, 227, 311, 201, 569, 436, 5, 325, 382, 1336, 1680, 264, 612, 473, 446], 'loglikelihood_distri': 'negative_binomial'}, {'output_key': 'infection_deathsXall', 'years': [50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 137, 138, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 179, 180, 181, 182, 183, 184, 185], 'values': [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 2, 3, 2, 8, 0, 14, 15, 0, 31, 12, 0, 57, 0, 0, 95, 207, 112, 112, 186, 240, 231, 365, 299, 319, 292, 418, 499, 880, 984, 1120, 1053, 518, 833, 1417, 541, 1341, 987, 635, 561, 574, 745, 1436, 753, 760, 642, 391, 546, 525, 544, 516, 389, 369, 242, 437, 367, 427, 289, 218, 166, 135, 304, 330, 274, 177, 243, 79, 70, 263, 347, 81, 349, 104, 579, 131, 108, 83, 74, 43, 33, 90, 73, 66, 65, 52, 57, 31, 28, 107, 81, 43, 46, 31, 13, 53, 84, 23, 27, 28, 24, 7, 29, 109, 28, 28, 14, 14, 6, 21, 57, 9, 19, 25, 0, 32, 27, 17, 14, 18, 0], 'loglikelihood_distri': 'negative_binomial'}]

# Use weekly counts
for target in TARGET_OUTPUTS:
    target['years'], target['values'] = get_weekly_summed_targets(target['years'], target['values'])

MULTIPLIERS = {}


def run_calibration_chain(max_seconds: int, run_id: int):
    base.run_calibration_chain(
        max_seconds, run_id, country, PAR_PRIORS, TARGET_OUTPUTS, mode="autumn_mcmc",
    )


if __name__ == "__main__":
    for i in range(len(TARGET_OUTPUTS)):
        print(TARGET_OUTPUTS[i]['output_key'])
        print(TARGET_OUTPUTS[i]['years'])
        print([[v] for v in TARGET_OUTPUTS[i]['values']])
        print()

    run_calibration_chain(
        30, 1
    )  # first argument only relevant for autumn_mcmc mode (time limit in seconds)
