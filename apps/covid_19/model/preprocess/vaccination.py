import numpy as np


def get_vacc_roll_out_function(params):

    # Get vaccination parameters
    coverage = params.coverage
    start_time = params.start_time
    end_time = params.end_time
    duration = end_time - start_time
    assert end_time >= start_time
    assert 0. <= coverage <= 1.

    # Calculate the vaccination rate from the coverage and the duration of the program
    vaccination_rate = \
        -np.log(1. - coverage) / \
        duration

    def get_vaccination_rate(time):
        return vaccination_rate if start_time < time < end_time else 0.

    return get_vaccination_rate
