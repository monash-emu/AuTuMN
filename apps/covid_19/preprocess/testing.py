import numpy as np


def convert_tests_to_prop(tests, maximum_detection, shape_parameter):
    return maximum_detection * (1.0 - np.exp(-shape_parameter * tests))


def convert_exponent_to_value(exponent):
    return 10 ** exponent
