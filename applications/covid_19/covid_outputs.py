from autumn.constants import Compartment


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


def create_request_stratified_incidence_covid(requested_stratifications, strata_dict, n_exposed, n_infectious):
    """
    Create derived outputs for disaggregated incidence
    """
    out_connections = {}
    origin_compartment = Compartment.EXPOSED if n_exposed < 2 else 'exposed_' + str(n_exposed)
    to_compartment = Compartment.INFECTIOUS if n_infectious < 2 else Compartment.INFECTIOUS + '_1'
    for stratification in requested_stratifications:
        for stratum in strata_dict[stratification]:
            out_connections['incidenceX' + stratification + '_' + stratum] \
                = {
                'origin': origin_compartment,
                'to': to_compartment,
                'origin_condition': '',
                'to_condition': stratification + '_' + stratum,
            }
    return out_connections
