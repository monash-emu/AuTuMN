# method to handle covid-specific parameters that are used during automatic calibration
# but encapsulated in dictionaries

def update_dict_params_for_calibration(params):
    """
    Update some specific parameters that are stored in a dictionary but are updated during calibration.
    For example, we may want to update params['default']['compartment_periods']['incubation'] using the parameter
    ['default']['compartment_periods_incubation']
    :param params: dict
        contains the model parameters
    :return: the updated dictionary
    """

    if 'n_imported_cases_final' in params:
        params['data']['n_imported_cases'][-1] = params['n_imported_cases_final']

    for location in ['school', 'work', 'home', 'other_locations']:
        if 'npi_effectiveness_' + location in params:
            params['npi_effectiveness'][location] = params['npi_effectiveness_' + location]

    for comp_type in ['incubation', 'infectious', 'late', 'hospital_early', 'hospital_late',
                      'icu_early', 'icu_late']:
        if 'compartment_periods_' + comp_type in params:
            params['compartment_periods'][comp_type] = params['compartment_periods_' + comp_type]

    return params
