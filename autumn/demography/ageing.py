

def add_agegroup_breaks(params):
    """
    Algorithmically add breakpoints for agegroup from request for the end year and the agegroup width
    """
    if 'agegroup_breaks' in params['default']:
        params['default']['age_breaks'] = \
            [str(i_break) for
             i_break in list(range(0,
                                   params['default']['agegroup_breaks'][0],
                                   params['default']['agegroup_breaks'][1]))
             ]
    return params
