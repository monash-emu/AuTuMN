from apps.covid_19.matrices import *


def test_plot_mixing_adjustments():

    _mixing_params = {'other_locations_times': [77.0, 81.0, 92.0, 106.0], 'other_locations_values': [1.0, 0.67, 0.33, 0.0],
                      'work_times': [77.0, 81.0, 92.0, 106.0], 'work_values': [1.0, 0.67, 0.33, 0.0],
                      'school_times': [70.0, 75.0], 'school_values': [1.0, 0.]}
    _npi_effectiveness_range = {'work': [0.5, 0.8], 'other_locations': [0.5, 0.8], 'school': [.7, .9]}

    plot_mixing_params_over_time(_mixing_params, _npi_effectiveness_range)
