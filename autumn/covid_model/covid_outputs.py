

def find_incidence_outputs(parameters):
    last_exposed = \
        'exposed_' + str(parameters['n_exposed_compartments']) if \
            parameters['n_exposed_compartments'] > 1 else \
            'exposed'
    first_infectious = \
        'infectious_1' if \
            parameters['n_infectious_compartments'] > 1 else \
            'infectious'
    return {
        'incidence':
            {'origin': last_exposed,
             'to': first_infectious,
             'origin_condition': '',
             'to_condition': ''
             }
    }
