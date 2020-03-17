

def add_agegroup_breaks(parameters):
    """
    Algorithmically add breakpoints for agegroup from request for the end year and the agegroup width
    """
    if 'agegroup_breaks' in parameters:
        parameters['all_stratifications']['agegroup'] = \
            [str(i_break) for
             i_break in list(range(0,
                                   parameters['agegroup_breaks'][0],
                                   parameters['agegroup_breaks'][1]))
             ]
    return parameters
