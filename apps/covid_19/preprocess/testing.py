import numpy as np

def convert_tests_to_prop(tests, maximum_detection, shape_parameter):
    return maximum_detection * (1. - np.exp(-shape_parameter * tests))