import numpy as np


def get_vacc_roll_out_function(params):

    # Get vaccination parameters
    coverage = params.coverage
    start_time = params.start_time
    end_time = params.end_time
    duration = end_time - start_time
    assert end_time >= start_time
    assert 0. <= coverage <= 1.

    # Get the function to represent the vaccination program
    def get_vaccination_rate(time):
        if start_time < time < end_time:
            vaccination_rate = \
                -np.log(1. - coverage) / \
                duration
            return vaccination_rate
        else:
            return 0.

    return get_vaccination_rate
