from autumn.constants import Compartment


def find_incidence_outputs(parameters):
    last_presympt = \
        'presympt_' + str(parameters['n_presympt_compartments']) if \
            parameters['n_presympt_compartments'] > 1 else \
            'presympt'
    first_infectious = \
        'infectious_1' if \
            parameters['n_infectious_compartments'] > 1 else \
            'infectious'
    return {
        'incidence':
            {'origin': last_presympt,
             'to': first_infectious,
             'origin_condition': '',
             'to_condition': ''
             }
    }


def create_request_stratified_incidence_covid(requested_stratifications, strata_dict, presympt, n_infectious):
    """
    Create derived outputs for disaggregated incidence
    """
    out_connections = {}
    origin_compartment = 'presympt' if presympt < 2 else 'presympt_' + str(presympt)
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
