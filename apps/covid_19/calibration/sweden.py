from apps.covid_19.calibration import base
from autumn.constants import Region
from apps.covid_19.mixing_optimisation.utils import (
    get_prior_distributions_for_opti,
    get_target_outputs_for_opti,
    get_weekly_summed_targets,
)


country = Region.SWEDEN

PAR_PRIORS = get_prior_distributions_for_opti()
# TARGET_OUTPUTS = get_target_outputs_for_opti(country, data_start_time=50, update_jh_data=False)
TARGET_OUTPUTS = [{'output_key': 'notifications', 'years': [50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185], 'values': [0, 0, 0, 0, 0, 0, 1, 5, 0, 5, 2, 1, 6, 14, 59, 7, 60, 42, 45, 107, 145, 99, 215, 147, 61, 81, 87, 89, 160, 200, 124, 171, 112, 240, 240, 314, 229, 378, 253, 328, 407, 512, 621, 563, 312, 387, 376, 487, 726, 722, 544, 466, 332, 465, 497, 482, 613, 676, 606, 563, 392, 545, 682, 751, 812, 610, 463, 286, 695, 681, 790, 428, 562, 235, 404, 495, 702, 705, 642, 656, 401, 348, 602, 637, 673, 625, 470, 466, 234, 422, 724, 649, 637, 379, 271, 384, 597, 648, 639, 749, 637, 429, 272, 775, 2214, 1080, 1056, 948, 843, 403, 791, 890, 1474, 1396, 1247, 683, 769, 940, 1239, 1481, 2033, 515, 257, 84, 1905, 1487, 1566, 1247, 0, 0, 2530, 784, 1241, 947, 780, 0], 'loglikelihood_distri': 'negative_binomial'}, {'output_key': 'infection_deathsXall', 'years': [50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185], 'values': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 3, 1, 3, 1, 5, 4, 1, 4, 11, 26, 15, 28, 0, 5, 36, 34, 59, 69, 50, 15, 28, 76, 114, 96, 106, 77, 17, 12, 20, 114, 170, 130, 67, 111, 29, 40, 185, 172, 84, 131, 40, 2, 80, 81, 107, 124, 67, 16, 10, 90, 85, 87, 99, 135, 45, 5, 31, 57, 147, 69, 117, 28, 5, 19, 45, 88, 40, 54, 67, 6, 31, 96, 95, 46, 84, 45, 0, 8, 65, 74, 20, 77, 17, 3, 35, 23, 78, 19, 40, 20, 0, 17, 48, 102, 12, 52, 4, 2, 11, 39, 48, 21, 50, 0, 0, 30, 23, 37, 41, 9, 0], 'loglikelihood_distri': 'negative_binomial'}]

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
